from logging import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from divergence import f, FDivergence, f_derivative_inverse
from network import DiscretePolicy, CriticTime
from torch.optim.lr_scheduler import CosineAnnealingLR

EPS = 1e-12

class MuNetwork(nn.Module):
    def __init__(self, config, learnable=True):
        super().__init__()
        self.learnable = learnable
        if self.learnable:
            self.theta = nn.Parameter(torch.full((config.reward_dim,), 1.0))
            self.register_buffer("mu_constant", None)
        else:
            self.theta = None
            self.register_buffer("mu_constant", torch.full((config.reward_dim,), 1.0))
            
    def forward(self):
        if self.learnable:
            return F.softplus(self.theta)
        else:
            return self.mu_constant

def piecewise_log_u_star_neg_mu(mu):  # returns u^*(-mu)
    # if mu < 1:  -1 - log(mu)
    # else:       0.5*mu^2 - 2*mu + 0.5
    return torch.where(
        mu < 1.0,
        -1.0 - torch.log(mu + 1e-12),
        0.5 * mu * mu - 2.0 * mu + 0.5,
    )
    
# def scaled_piecewise_log_u_star_neg_mu(mu, c=100.0):
#     mu_s = mu / c  # 입력 스케일링
#     base = torch.where(
#         mu_s < 1.0,
#         -1.0 - torch.log(mu_s),
#         0.5 * mu_s * mu_s - 2.0 * mu_s + 0.5,
#     )
#     return c * base

# def log_u_star_neg_mu(mu ,c):
#     """
#     u^*(-mu) for log utility.
#     Args:
#         mu    : Tensor of nonnegative multipliers
#         c     : scaling constant
#         eps   : numerical safety
#     Returns:
#         Tensor with u^*(-mu)
#     """
#     # return -1.0 - torch.log(mu + EPS)
#     return -c - c * torch.log((mu + EPS)/c)

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

# def scaled_piecewise_frac_u_star_neg_mu(mu, alpha, c=100.0):
#     s = float(np.exp(-alpha))
#     mu_s = mu / c
#     exp_term = (alpha - 1.0) / alpha

#     right_mask = (mu_s < 1.0)  # 분기점이 mu < c로 스케일됨
#     right_val = (alpha * torch.pow(mu_s, exp_term) - 1.0) / (1.0 - alpha)
#     left_val  = ((mu_s - (2.0 - s))**2) / (2.0 * (1.0 - s)) - 1.5 + 0.5 * s

#     return c * torch.where(right_mask, right_val, left_val)


class FiniteAET(nn.Module):
    """
    Finite-horizon (time-dependent ν), ν(s,H)=0.
    """
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.alpha = config.alpha
        self.f_div = config.f_divergence  # e.g., FDivergence.KL

        # -----------------
        # Networks
        # -----------------
        self.policy = DiscretePolicy(   # time-dependent policy
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim,
            horizon=config.horizon,
            time_embed_dim=getattr(config, "time_embed_dim", 8),
        ).to(device)

        # time-dependent ν
        self.nu = CriticTime(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            horizon=config.horizon,
            time_embed_dim=getattr(config, "time_embed_dim", 8),
            layer_norm=config.layer_norm,
        ).to(device)
        
        if config.fair_alpha == 0.0: # ESR case
            self.mu = MuNetwork(config, learnable=False).to(device)
            self.mu_optim = None
        else: # SER case
            self.mu = MuNetwork(config, learnable=True).to(device)
            self.mu_optim = optim.Adam(self.mu.parameters(), lr=config.mu_lr)

        # -----------------
        # Optimizers
        # -----------------
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.nu_optim = optim.Adam(self.nu.parameters(), lr=config.nu_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        
        self.step = 0

    def train(self, batch):
        return self.train_step(batch)

    # -----------------
    # Loss functions
    # -----------------

    def nu_loss_fn(self, states, next_states, timesteps, next_timesteps, rewards, initial_states):
        nu_vals   = self.nu(states, timesteps)               # [B]
        next_nu   = self.nu(next_states, next_timesteps)     # [B]
        init_nu = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]
        
        mu = self.mu()  # [R]
        e         = torch.matmul(rewards, mu).squeeze(-1) + next_nu - nu_vals    # [B]

        # w* = (f')^{-1}(e/α); w >= 0
        state_action_ratio = f_derivative_inverse(e / self.alpha, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)

        f_vals = f(state_action_ratio, self.f_div)

        # loss
        loss_1 = init_nu.mean()
        loss_2 = (state_action_ratio * e - self.alpha * f_vals).mean()
        if self.config.fair_alpha == 0.0:
            loss_3 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            # loss_3 = log_u_star_neg_mu(mu).sum()
            loss_3 = piecewise_log_u_star_neg_mu(mu).sum()
            # loss_3 = scaled_piecewise_log_u_star_neg_mu(mu, c=100.0).sum()
        else:
            # loss_3 = frac_u_star_neg_mu(mu, self.config.fair_alpha).sum()
            loss_3 = piecewise_frac_u_star_neg_mu(mu, self.config.fair_alpha).sum()
            # loss_3 = scaled_piecewise_frac_u_star_neg_mu(mu, self.config.fair_alpha, c=100.0).sum()
        # ----- gradient penalty -----
        gp_coeff = getattr(self.config, "nu_grad_penalty_coeff", 0.0)
        
        if gp_coeff > 0.0:
            eps = torch.rand(states.shape[0], 1, device=states.device)
            states_inter = eps * states + (1 - eps) * next_states
            states_inter.requires_grad_(True)
            nu_out = self.nu(states_inter, timesteps)
            grads = torch.autograd.grad(
                outputs=nu_out,
                inputs=states_inter,
                grad_outputs=torch.ones_like(nu_out),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            nu_grad_penalty = (grads.norm(dim=-1) ** 2).mean()
        else:
            nu_grad_penalty = 0.0

        nu_loss = loss_1 + loss_2 + loss_3 + gp_coeff * nu_grad_penalty
        return nu_loss, (e, nu_grad_penalty, loss_1, loss_2, loss_3, nu_vals.mean(), next_nu.mean(), f_vals.mean())

    def policy_loss_fn(self, states, actions, rewards, next_states, timesteps, next_timesteps):
        dist = self.policy(states, timesteps)  # [B, A]
        log_probs = dist.log_prob(actions)                    # [B]

        nu_val     = self.nu(states, timesteps)               # [B, D]
        next_nu_val= self.nu(next_states, next_timesteps)     # [B, D]
        
        mu = self.mu()  # [R]
        e = torch.matmul(rewards, mu).squeeze(-1) + next_nu_val - nu_val  # [B, D]

        state_action_ratio = f_derivative_inverse(e / self.alpha, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)  # 음수는 0으로
        w = state_action_ratio.detach()  # [B, D]
        
        # w = w / (w.mean() + 1e-8)
        policy_loss = -(w * log_probs).mean()
        return policy_loss, w

    # -----------------
    # Training step
    # -----------------
    def train_step(self, batch):
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)     # [B,1] or [B]
        timesteps      = batch["timesteps"].to(self.device)   # [B,1] long
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states    = batch["initial_states"].to(self.device)   # [B,1] float

        # ---- Update ν and μ ----
        self.nu_optim.zero_grad()
        if self.mu_optim is not None:
            self.mu_optim.zero_grad()

        nu_loss, (e, nu_grad_penalty, init_loss, advantage_loss, utill_loss, nu_vals_mean, next_nu_mean, f_vals_mean) = self.nu_loss_fn(
            states, next_states, timesteps, next_timesteps, rewards, initial_states
        )
        nu_loss.backward()
        self.nu_optim.step()
        if self.mu_optim is not None:
            self.mu_optim.step()
        self.nu_scheduler.step()

        # ---- Update policy ----
        self.policy_optim.zero_grad()
        policy_loss, w = self.policy_loss_fn(
            states, actions, rewards, next_states, timesteps, next_timesteps
        )
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        self.step += 1

        return {
            "policy_loss": float(policy_loss.item()),
            "nu_loss": float(nu_loss.item()),
            "w_mean": float(w.mean().item()),
            "w_std": float(w.std().item()),
            "w_max": float(w.max().item()),
            "w_min": float(w.min().item()),
            "e_mean": float(e.mean().item()),
            "e_std": float(e.std().item()),
            "e_min": float(e.min().item()),
            "e_max": float(e.max().item()),
            "nu_grad_penalty": float(nu_grad_penalty.item()) if isinstance(nu_grad_penalty, torch.Tensor) else 0.0,
            "init_loss": float(init_loss.item()),
            "advantage_loss": float(advantage_loss.item()),
            "utill_loss": float(utill_loss.item()),
            "nu_vals_mean": float(nu_vals_mean.item()),
            "next_nu_mean": float(next_nu_mean.item()),
            "mu_0": float(self.mu().detach()[0].item()),
            "mu_1": float(self.mu().detach()[1].item()),
            "f_vals_mean": float(f_vals_mean.detach().item()),
        }

    # -----------------
    # Save / Load
    # -----------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))