import torch
import torch.nn.functional as F


def _response_log_prob_sum(
    lm: torch.nn.Module,
    tokenizer,
    prompt: str,
    response: str,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("Prompt and response must each tokenize to at least one token.")

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=next(lm.parameters()).device).unsqueeze(0)
    labels = torch.tensor(full_ids[1:], dtype=torch.long, device=input_ids.device).unsqueeze(0)

    logits = lm(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    response_start = len(prompt_ids) - 1
    return log_probs[:, response_start:].sum(dim=-1).squeeze(0)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    chosen_logp = _response_log_prob_sum(lm, tokenizer, prompt, response_chosen)
    rejected_logp = _response_log_prob_sum(lm, tokenizer, prompt, response_rejected)
    ref_chosen_logp = _response_log_prob_sum(lm_ref, tokenizer, prompt, response_chosen)
    ref_rejected_logp = _response_log_prob_sum(lm_ref, tokenizer, prompt, response_rejected)

    preference_logit = beta * ((chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp))
    return -F.logsigmoid(preference_logit)
