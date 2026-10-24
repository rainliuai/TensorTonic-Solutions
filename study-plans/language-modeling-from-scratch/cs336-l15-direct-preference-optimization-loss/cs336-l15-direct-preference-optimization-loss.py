import torch
import torch.nn.functional as F

def direct_preference_optimization_loss(
    policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
    beta: int | float, label_smoothing: int | float = 0.0,
) -> dict:
    """
    Returns a dict: loss, per_example_loss, chosen_rewards, rejected_rewards, preference_accuracy (tensors).
    """
    chosen_rewards = beta * (policy_chosen - reference_chosen)
    rejected_rewards = beta * (policy_rejected - reference_rejected)
    z = chosen_rewards - rejected_rewards
    per_example_loss = - (1 - label_smoothing) * F.logsigmoid(z) - label_smoothing * F.logsigmoid(-z)
    loss = per_example_loss.mean()
    preference_accuracy = (chosen_rewards > rejected_rewards).to(policy_chosen.dtype).mean()

    return {"loss": loss, "per_example_loss":per_example_loss, "chosen_rewards": chosen_rewards, 
           "rejected_rewards": rejected_rewards, "preference_accuracy": preference_accuracy}
    
    
