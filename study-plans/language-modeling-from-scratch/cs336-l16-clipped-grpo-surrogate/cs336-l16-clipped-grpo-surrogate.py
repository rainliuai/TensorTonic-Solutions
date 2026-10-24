import torch

def clipped_grpo_surrogate(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
    advantages: torch.Tensor, token_mask: torch.Tensor,
    clip_epsilon: int | float = 0.2,
) -> dict:
    """
    Returns a dict: loss and clipped_fraction (scalars), per_sequence_surrogate (tensor).
    """
    r = torch.exp(torch.clip(new_log_probs - old_log_probs, -60, 60))
    unclipped = r * advantages.unsqueeze(-1)
    clipped = torch.clip(r, 1-clip_epsilon, 1+clip_epsilon) * advantages.unsqueeze(-1)

    clipped_fraction = ((clipped != unclipped) & token_mask).sum() / token_mask.sum().clamp_min(1e-6)
    
    s = torch.minimum(unclipped, clipped)
    per_sequence_surrogate = (s * token_mask).sum(dim=1) / (token_mask.sum(dim=1)).clamp_min(1e-6)
    loss = - (s * token_mask).sum() / token_mask.sum().clamp_min(1e-6)
    return {"loss": loss, "per_sequence_surrogate":per_sequence_surrogate, "clipped_fraction":clipped_fraction,}
