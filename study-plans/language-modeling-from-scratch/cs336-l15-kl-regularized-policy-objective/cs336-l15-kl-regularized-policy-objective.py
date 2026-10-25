import torch

def kl_regularized_policy_objective(
    policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor,
    rewards: torch.Tensor, response_mask: torch.Tensor, beta: int | float,
) -> dict:
    """
    Returns a dict: objective (scalar), penalized_rewards and sampled_divergences (float32 tensors).
    """
    sampled_divergences = (response_mask * (policy_log_probs - reference_log_probs)).sum(dim = -1) #B,
    penalized_rewards = rewards - beta * sampled_divergences
    objective = penalized_rewards.mean()
    return {"objective": objective, "penalized_rewards": penalized_rewards, "sampled_divergences": sampled_divergences}
    
