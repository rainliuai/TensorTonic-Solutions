import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    return ((np.array(epsilon) - np.array(epsilon_pred)) ** 2).mean()