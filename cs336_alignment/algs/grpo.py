import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn
import wandb
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel

try:
    from vllm import SamplingParams
except ImportError:
    class SamplingParams:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("vllm is required for sampling and evaluation workflows.")

from cs336_alignment.algs.utils import (
    get_response_log_probs,
    log_generation,
    masked_mean,
    tokenize_prompt_and_output,
)
from cs336_alignment.base_config import BaseConfig
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.utils import (
    clear_memory,
    compute_response_masked_mean,
    get_ctx,
    load_dataset,
    print_color,
    print_rich_dict,
    to_float,
)
from cs336_alignment.vllm_utils import (
    generate_responses,
    load_policy_into_vllm_instance,
)

REWARD_FN_MAP = {"r1_zero_reward_fn": r1_zero_reward_fn, "question_only_reward_fn": question_only_reward_fn}


def iter_grpo_batch_indices(
    num_samples: int,
    train_batch_size: int,
    epochs_per_rollout_batch: int,
) -> list[torch.Tensor]:
    all_batches = []
    for _ in range(epochs_per_rollout_batch):
        epoch_indices = torch.randperm(num_samples)
        for start in range(0, num_samples, train_batch_size):
            all_batches.append(epoch_indices[start : start + train_batch_size])
    return all_batches


@dataclass
class GRPOTrainConfig(BaseConfig):
    n_grpo_cur_steps: int = 200
    rollout_batch_size: int = 256
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    group_size: int = 8

    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    reward_fn: Literal["r1_zero_reward_fn"] = "r1_zero_reward_fn"
    cliprange: float = 0.2
    norm_by_std: bool = True

    # Optimizer hyperparameters
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip"
    betas: tuple = field(default=(0.9, 0.95))
    weight_decay: float = 0.0
    max_lr: float = 5e-6
    max_grad_norm: float = 1.0

    # Sampling hyperparameters
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4
    sampling_top_p: float = 1.0
    sampling_stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])

    # Others
    mixed_precision_training: bool = True
    eval_interval: int = 5
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    def __post_init__(self):
        self.run_name = f"grpo_dataset({self.dataset_name})_prompt({self.prompt_template_path.split('/')[-1]})_reward({self.reward_fn})_loss_type({self.loss_type})"

        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        self.micro_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size


## ------ Advantage Computation ------ ##
def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool = True,
):
    formatted_rewards = []
    answer_correct_rewards = []
    rewards = []
    for response, true_answer in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, true_answer)
        rewards.append(reward_info["reward"])
        formatted_rewards.append(reward_info["format_reward"])
        answer_correct_rewards.append(reward_info["answer_reward"])

    advs = []
    for i in range(0, len(rewards), group_size):
        group_rewards = rewards[i : i + group_size]
        group_rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        group_mean = torch.mean(group_rewards_tensor)
        if normalized_by_std:
            group_std = torch.std(group_rewards_tensor) + advantage_eps
            normalized_rewards = (group_rewards_tensor - group_mean) / group_std
        else:
            normalized_rewards = group_rewards_tensor - group_mean
        advs.extend(normalized_rewards.tolist())

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    normalized_rewards_tensor = torch.tensor(advs, dtype=torch.float32)
    formatted_rewards_tensor = torch.tensor(formatted_rewards, dtype=torch.float32)
    answer_correct_rewards_tensor = torch.tensor(answer_correct_rewards, dtype=torch.float32)

    meta_info = {
        "reward_mean": float(rewards_tensor.mean().item()),
        "reward_std": float(rewards_tensor.std().item()),
        "format_reward_mean": float(formatted_rewards_tensor.mean().item()),
        "answer_reward_mean": float(answer_correct_rewards_tensor.mean().item()),
        "advantage_mean": float(normalized_rewards_tensor.mean().item()),
        "advantage_std": float(normalized_rewards_tensor.std().item()),
    }

    return normalized_rewards_tensor, rewards_tensor, meta_info


## ------ Loss Computation ------ ##
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    """
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float = 0.2,
):
    """
    Compute the GRPO-CLIP loss at every token.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped_loss = -advantages * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = -advantages * clipped_ratio

    loss = torch.max(unclipped_loss, clipped_loss)
    metadata = {}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None,
    advantages: torch.Tensor | None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float = 0.2,
):
    """
    Compute the policy-gradient loss at every token.
    """

    if loss_type == "no_baseline" and raw_rewards is not None:
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
        return loss, metadata
    elif loss_type == "reinforce_with_baseline" and advantages is not None:
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
        return loss, metadata

    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None:
            raise ValueError("Advantages and old_log_probs must be provided for grpo_clip loss.")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        return loss, metadata

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


## ------ Dataset Utils ------ ##
def sample_batch_questions(
    prompts: list[str],
    answers: list[str],
    batch_size: int,
    group_size: int = 8,
) -> tuple[list[str], list[str]]:
    index = random.sample(range(len(prompts)), k=batch_size)
    sampled_prompts = [prompts[i] for i in index]
    sampled_answers = [answers[i] for i in index]

    batch_prompts = []
    batch_answers = []
    for p, a in zip(sampled_prompts, sampled_answers):
        batch_prompts.extend([p] * group_size)
        batch_answers.extend([a] * group_size)

    return batch_prompts, batch_answers


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the GRPO loss over microbatches for training.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    masked_loss = masked_mean(
        tensor=loss,
        mask=response_mask,
        dim=-1,
    )

    masked_loss = masked_loss.mean()
    masked_loss = masked_loss / gradient_accumulation_steps
    masked_loss.backward()

    return masked_loss, metadata


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_config: GRPOTrainConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        self.model = model
        self.train_config = train_config
        self.device = device
        self.dataset_dir_base = dataset_dir_base

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            use_fast=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=train_config.betas,
            lr=self.train_config.max_lr,
            weight_decay=self.train_config.weight_decay,
        )
        self.ctx = get_ctx(
            use_mixed=self.train_config.mixed_precision_training,
            device=device,
        )

        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_prompts, train_cots, train_answers = load_dataset(
            os.path.join(dataset_dir, "train.jsonl"),
            prompt_template_path=self.train_config.prompt_template_path,
        )
        self.train_prompts = train_prompts
        self.train_answers = train_answers

        test_prompts, test_cots, test_answers = load_dataset(
            os.path.join(dataset_dir, "test.jsonl"),
            prompt_template_path=self.train_config.prompt_template_path,
        )
        self.test_prompts = test_prompts
        self.test_true_answers = test_answers

        self.checkpoint_path = os.path.join(
            train_config.checkpoint_dir,
            f"grpo_{train_config.model_name.split('/')[-1]}_{train_config.dataset_name}_{train_config.reward_fn}_loss({train_config.loss_type})",
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_config.to_json(
            os.path.join(self.checkpoint_path, "train_config.json"),
        )
        self.sampling_params = SamplingParams(
            temperature=self.train_config.sampling_temperature,
            max_tokens=self.train_config.sampling_max_tokens,
            top_p=self.train_config.sampling_top_p,
            min_tokens=self.train_config.sampling_min_tokens,
            include_stop_str_in_output=True,
            stop=self.train_config.sampling_stop_tokens,
        )

        self.reward_fn = REWARD_FN_MAP[self.train_config.reward_fn]

        self.grpo_cur_step = 0

    @torch.no_grad()
    def evaluate(self, vllm=None):
        print_color(f"Evaluating GRPO model on test dataset at step {self.grpo_cur_step}", color="magenta")

        overview = evaluate_responses(
            vllm=vllm,
            prompts=self.test_prompts,
            answers=self.test_true_answers,
            sampling_params=self.sampling_params,
        )

        print_color("Evaluation Overview", color="magenta")
        print_rich_dict(overview)
        return overview

    @torch.no_grad()
    def sample_responses(
        self,
        vllm=None,
        num_samples: int = 5,
    ):
        print_color(f"Sampling {num_samples} responses from GRPO model...", color="cyan")

        index = random.sample(range(len(self.test_prompts)), k=num_samples)
        prompts = [self.test_prompts[i] for i in index]
        true_answers = [self.test_true_answers[i] for i in index]

        out = log_generation(
            prompts=prompts,
            true_answers=true_answers,
            reward_fn=self.reward_fn,
            model=self.model,
            tokenizer=self.tokenizer,
            vllm=vllm,
            sampling_params=self.sampling_params,
        )

        print_rich_dict(out)

    def grpo_train_step(
        self,
        vllm,
    ):
        print_color(f"Sampling batch of {self.train_config.rollout_batch_size} questions...", color="green")
        sample_prompts, sample_answers = sample_batch_questions(
            self.train_prompts,
            self.train_answers,
            self.train_config.n_prompts_per_rollout_batch,
            self.train_config.group_size,
        )

        # Load old policy into VLLM
        # This will be done at the end of the previous training step while do the evaluation

        print_color("Generating rollout responses...", color="green")
        rollout_responses = generate_responses(vllm, sample_prompts, self.sampling_params)

        tokenized = tokenize_prompt_and_output(
            sample_prompts,
            rollout_responses,
            self.tokenizer,
        )

        print_color("Computing rewards...", color="green")
        repeated_ground_truths = sample_answers
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=self.reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=self.train_config.group_size,
            advantage_eps=self.train_config.advantage_eps,
            normalized_by_std=self.train_config.norm_by_std,
        )

        print_color("Computing old log probabilities...", color="green")
        input_ids = tokenized["input_ids"].to(self.device, non_blocking=True)
        labels = tokenized["labels"].to(self.device, non_blocking=True)
        response_mask = tokenized["response_mask"].to(self.device, non_blocking=True)
        ave_length = response_mask.sum(dim=1).float().mean().item()

        old_log_probs = []
        self.model.eval()
        with torch.no_grad():
            for i in trange(0, input_ids.size(0), self.train_config.micro_batch_size):
                batch_input_ids = input_ids[i : i + self.train_config.micro_batch_size]
                batch_labels = labels[i : i + self.train_config.micro_batch_size]

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=batch_input_ids,
                        labels=batch_labels,
                        return_token_entropy=False,
                    )
                    batch_log_probs = policy_outputs["log_probs"]

                old_log_probs.append(batch_log_probs.cpu())
        old_log_probs = torch.cat(old_log_probs, dim=0)
        self.model.train()

        advantages = advantages.to(self.device)
        raw_rewards = raw_rewards.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        train_batches = iter_grpo_batch_indices(
            num_samples=input_ids.size(0),
            train_batch_size=self.train_config.train_batch_size,
            epochs_per_rollout_batch=self.train_config.epochs_per_rollout_batch,
        )

        print_color(f"Performing {len(train_batches)} training steps...", color="green")
        batch_loss = 0.0
        token_entropy_avg = 0.0
        for train_step, batch_indices in enumerate(train_batches, start=1):
            micro_batches = [
                batch_indices[start : start + self.train_config.micro_batch_size]
                for start in range(0, len(batch_indices), self.train_config.micro_batch_size)
            ]
            micro_batches = [micro for micro in micro_batches if len(micro) > 0]

            for micro_indices in trange(
                len(micro_batches),
                desc="Microbatches",
            ):
                current_indices = micro_batches[micro_indices]
                micro_input_ids = input_ids[current_indices]
                micro_labels = labels[current_indices]
                micro_response_mask = response_mask[current_indices]
                micro_advantages = advantages[current_indices]
                micro_raw_rewards = raw_rewards[current_indices]
                micro_old_log_probs = old_log_probs[current_indices]

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=micro_input_ids,
                        labels=micro_labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = policy_outputs["log_probs"]
                    token_entropy = policy_outputs["token_entropy"]
                    token_entropy_masked = compute_response_masked_mean(token_entropy, micro_response_mask)

                micro_loss, micro_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=micro_response_mask,
                    gradient_accumulation_steps=len(micro_batches),
                    loss_type=self.train_config.loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=self.train_config.cliprange,
                )

                batch_loss += to_float(micro_loss)
                token_entropy_avg += to_float(token_entropy_masked) / len(micro_batches)

            print_color(
                f"GRPO Step {self.grpo_cur_step} | Train Step {train_step}/{len(train_batches)} | "
                f"Batch Loss: {batch_loss:.4f} | Avg Token Entropy: {token_entropy_avg:.4f}",
                color="green",
            )
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        del input_ids, labels, response_mask, old_log_probs
        clear_memory()

        return {
            "train/batch_loss": batch_loss,
            "train/token_entropy_avg": token_entropy_avg,
            "train/grad_norm": grad_norm,
            "train/ave_length": ave_length,
            **{f"train/{key}": value for key, value in metadata.items()},
        }

    def train(
        self,
        vllm,
    ):
        for _ in range(
            self.train_config.n_grpo_cur_steps,
        ):
            self.grpo_cur_step += 1

            print_color(
                f"\n=== GRPO Training Step {self.grpo_cur_step} / {self.train_config.n_grpo_cur_steps} ===",
                color="magenta",
            )

            log_dict = self.grpo_train_step(
                vllm,
            )

            print_color("Loading current policy into VLLM instance...", color="green")
            load_policy_into_vllm_instance(
                self.model,
                vllm,
            )

            if self.grpo_cur_step % self.train_config.eval_interval == 0:
                clear_memory()

                self.sample_responses(vllm=vllm, num_samples=3)
                out = self.evaluate(vllm)
                log_dict["eval/answer_accuracy"] = out["answer_accuracy"]
                log_dict["eval/answer_correct"] = out["answer_correct"]
                log_dict["eval/format_correct"] = out["format_correct"]
                log_dict["eval/formatted_but_answer_wrong"] = out["formatted_but_answer_wrong"]
                log_dict["eval/reward_1"] = out["reward_1"]

            if self.train_config.wandb_logging:
                wandb.log(log_dict, step=self.grpo_cur_step)
