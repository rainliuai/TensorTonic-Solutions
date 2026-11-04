import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    n = len(betas)
    alpha_bar = np.cumprod(np.array([1] * n) - np.array(betas))
    return np.round(alpha_bar, 6)

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]
    x_t = np.array(x_0) * np.sqrt(alpha_bar_t) + np.array(epsilon) * np.sqrt(1 - alpha_bar_t)
    return x_t.tolist()