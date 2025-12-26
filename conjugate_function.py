import numpy as np
import torch

EPS = 1e-12

def log_u_star_neg_mu(mu):
    """
    u^*(-mu) for log utility.
    Args:
        mu    : Tensor of nonnegative multipliers
        eps   : numerical safety
    Returns:
        Tensor with u^*(-mu)
    """
    return -1.0 - torch.log(mu + EPS)
    
def piecewise_log_u_star_neg_mu(mu):  # returns u^*(-mu)
    # if mu < 1:  -1 - log(mu)
    # else:       0.5*mu^2 - 2*mu + 0.5
    return torch.where(
        mu < 1.0,
        -1.0 - torch.log(mu + 1e-12),
        0.5 * mu * mu - 2.0 * mu + 0.5,
    )
    
def frac_u_star_neg_mu(mu, alpha):
    """
    u^*(-mu) for fractional utility.
    Args:
        mu    : Tensor of nonnegative multipliers
        alpha : fairness parameter (>0)
        eps   : numerical safety
    Returns:
        Tensor with u^*(-mu)
    """
    exp_term = (alpha - 1.0) / alpha
    return (alpha * torch.pow(mu + EPS, exp_term) - 1.0) / (1.0 - alpha)

def piecewise_frac_u_star_neg_mu(mu, alpha):
    """
    u^*(-mu) for piecewise_frac with s = exp(-alpha).
    Args:
        mu    : Tensor of nonnegative multipliers
        alpha : fairness parameter (>0)
        eps   : numerical safety
    Returns:
        Tensor with u^*(-mu)
    """
    s = float(np.exp(-alpha))  # e^{-α}
    
    right_mask = (mu < 1.0)

    # Right branch (0<mu<1): u^*(-mu) = [α * mu^{(α-1)/α} - 1] / (1-α)
    exp_term = (alpha - 1.0) / alpha
    right_val = (alpha * torch.pow(mu, exp_term) - 1.0) / (1.0 - alpha)

    # Left branch (mu>=1): u^*(-mu) = ((mu - 2 + s)^2)/(2*(1-s)) - 1.5 + 0.5*s
    left_val = ((mu - (2.0 - s))**2) / (2.0 * (1.0 - s)) - 1.5 + 0.5 * s

    return torch.where(right_mask, right_val, left_val)

