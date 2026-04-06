from types import SimpleNamespace

import pytest
import torch

import train_ei
import train_grpo
from cs336_alignment.algs.grpo import GRPOTrainer, iter_grpo_batch_indices
from cs336_alignment.utils import compute_response_masked_mean, get_model_loading_kwargs


def test_iter_grpo_batch_indices_cover_all_samples_each_epoch():
    torch.manual_seed(0)
    batches = iter_grpo_batch_indices(
        num_samples=10,
        train_batch_size=4,
        epochs_per_rollout_batch=2,
    )

    assert [len(batch) for batch in batches] == [4, 4, 2, 4, 4, 2]

    epoch_1 = torch.cat(batches[:3]).tolist()
    epoch_2 = torch.cat(batches[3:]).tolist()
    assert sorted(epoch_1) == list(range(10))
    assert sorted(epoch_2) == list(range(10))
    assert epoch_1 != epoch_2


def test_compute_response_masked_mean_uses_only_response_tokens():
    token_entropy = torch.tensor([[1.0, 100.0, 1000.0], [2.0, 200.0, 2000.0]])
    response_mask = torch.tensor([[False, True, False], [False, False, True]])

    out = compute_response_masked_mean(token_entropy, response_mask)

    assert torch.isclose(out, torch.tensor((100.0 + 2000.0) / 2))


def test_get_model_loading_kwargs_cpu():
    kwargs = get_model_loading_kwargs(torch.device("cpu"))

    assert kwargs["device_map"] == "cpu"
    assert kwargs["torch_dtype"] == torch.float32
    assert "attn_implementation" not in kwargs


def test_grpo_train_skips_wandb_when_disabled(monkeypatch):
    trainer = object.__new__(GRPOTrainer)
    trainer.train_config = SimpleNamespace(
        n_grpo_cur_steps=1,
        eval_interval=10,
        wandb_logging=False,
    )
    trainer.grpo_cur_step = 0
    trainer.model = object()

    called = {"grpo_train_step": 0}

    def fake_grpo_train_step(vllm):
        called["grpo_train_step"] += 1
        return {"train/batch_loss": 1.0}

    trainer.grpo_train_step = fake_grpo_train_step
    trainer.sample_responses = lambda *args, **kwargs: None
    trainer.evaluate = lambda *args, **kwargs: {"answer_accuracy": 1.0}

    monkeypatch.setattr("cs336_alignment.algs.grpo.load_policy_into_vllm_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr("cs336_alignment.algs.grpo.wandb.log", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("wandb.log should not be called when wandb_logging is false")
    ))

    trainer.train(vllm=object())
    assert called["grpo_train_step"] == 1


def _make_dummy_model():
    class DummyModel:
        def to(self, device):
            return self

    return DummyModel()


def test_train_ei_saves_final_checkpoint(monkeypatch, tmp_path):
    save_calls = []

    class FakeTrainer:
        def __init__(self, model, train_config, device):
            self.model = model
            self.optimizer = object()
            self.checkpoint_path = str(tmp_path / "ei-checkpoints")
            self.sft_cur_step = 7

        def train(self, vllm=None):
            return None

    config = SimpleNamespace(model_name="dummy-model", seed=0, wandb_logging=False)

    monkeypatch.setattr(train_ei, "EITrainer", FakeTrainer)
    monkeypatch.setattr(train_ei.EITrainConfig, "from_json", lambda path: config)
    monkeypatch.setattr(train_ei, "init_vllm", lambda **kwargs: object())
    monkeypatch.setattr(train_ei, "get_device", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(train_ei, "seed_everything", lambda seed: None)
    monkeypatch.setattr(train_ei.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(train_ei.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: _make_dummy_model())
    monkeypatch.setattr(train_ei, "save_model_checkpoint", lambda **kwargs: save_calls.append(kwargs))

    train_ei.main()

    assert len(save_calls) == 1
    assert save_calls[0]["cur_step"] == 7
    assert save_calls[0]["checkpoint_path"].endswith("checkpoint_final.pt")


def test_train_grpo_saves_final_checkpoint(monkeypatch, tmp_path):
    save_calls = []

    class FakeTrainer:
        def __init__(self, model, train_config, device):
            self.model = model
            self.optimizer = object()
            self.checkpoint_path = str(tmp_path / "grpo-checkpoints")
            self.grpo_cur_step = 5

        def train(self, vllm=None):
            return None

    config = SimpleNamespace(model_name="dummy-model", seed=0, wandb_logging=False)

    monkeypatch.setattr(train_grpo, "GRPOTrainer", FakeTrainer)
    monkeypatch.setattr(train_grpo.GRPOTrainConfig, "from_json", lambda path: config)
    monkeypatch.setattr(train_grpo, "init_vllm", lambda **kwargs: object())
    monkeypatch.setattr(train_grpo, "get_device", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(train_grpo, "seed_everything", lambda seed: None)
    monkeypatch.setattr(train_grpo.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(train_grpo.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: _make_dummy_model())
    monkeypatch.setattr(train_grpo, "save_model_checkpoint", lambda **kwargs: save_calls.append(kwargs))

    train_grpo.main()

    assert len(save_calls) == 1
    assert save_calls[0]["cur_step"] == 5
    assert save_calls[0]["checkpoint_path"].endswith("checkpoint_final.pt")
