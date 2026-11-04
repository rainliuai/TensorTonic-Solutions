import numpy as np

def reverse_step(x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """
    alpha = 1 - np.array(betas)
    alpha_bar = np.cumprod(np.array(alpha))
    alpha_bar_t = alpha_bar[t-1]
    alpha_t = alpha[t-1]
    beta_t = betas[t-1]
    mu_t = 1 / np.sqrt(alpha_t) * (np.array(x_t) - beta_t / np.sqrt(1-alpha_bar_t) * np.array(epsilon_pred))
    if t > 1:
        return mu_t + np.sqrt(beta_t) * np.array(z)
    return mu_t