from enum import Enum
import torch
import torch.nn.functional as F


class FDivergence(str, Enum):
    KL = "KL"
    CHI = "Chi"
    SOFT_CHI = "SoftChi"
    DUAL_DICE = "DualDICE"


def f(x: torch.Tensor, f_divergence: FDivergence, eps: float = 1e-10) -> torch.Tensor:
    x = torch.as_tensor(x)

    if f_divergence == FDivergence.KL:
        return x * torch.log(x + eps)

    elif f_divergence == FDivergence.CHI:
        return 0.5 * (x - 1.0) ** 2

    elif f_divergence == FDivergence.SOFT_CHI:
        # KL region when x < 1, Chi-square region otherwise
        return torch.where(
            x < 1.0,
            x * torch.log(x + eps) - x + 1.0,
            0.5 * (x - 1.0) ** 2,
        )

    elif f_divergence == FDivergence.DUAL_DICE:
        return (2.0 / 3.0) * torch.abs(x) ** 1.5

    else:
        raise ValueError(f"Unknown f-divergence: {f_divergence}")


def f_derivative_inverse(
    y: torch.Tensor, f_divergence: FDivergence, t: float = 1.0
) -> torch.Tensor:
    y = torch.as_tensor(y)

    if f_divergence == FDivergence.KL:
        # f'(x) = log(x) + 1  ->  (f')^{-1}(y) = exp(y - 1)
        return torch.exp(y - 1.0)

    elif f_divergence == FDivergence.CHI:
        # f'(x) = x - 1  ->  (f')^{-1}(y) = y + 1
        return y + 1.0

    elif f_divergence == FDivergence.SOFT_CHI:
        # piecewise inverse
        return torch.where(y < 0.0, torch.exp(y), y + 1.0)

    elif f_divergence == FDivergence.DUAL_DICE:
        raise ValueError(f"(f')⁻¹ does not exist for {f_divergence}.")

    else:
        raise ValueError(f"Unknown f-divergence: {f_divergence}")


def state_action_ratio(
    nu: torch.Tensor,
    next_nu: torch.Tensor,
    rewards: torch.Tensor,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
) -> torch.Tensor:
    """
    Compute state-action ratio weights:
      w*(s,a) = max(0, (f')^{-1}((r + γ ν(s') - ν(s)) / α))
    """

    e = rewards + discount * next_nu - nu
    w = f_derivative_inverse(e / alpha, f_divergence)
    return F.relu(w)

