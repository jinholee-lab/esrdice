from logging import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from divergence import f, FDivergence, f_derivative_inverse
from network import DiscretePolicy, DiscretePolicyMultiHead, CriticTime, CriticTimeVector
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

class FiniteAET_double(nn.Module):
    """
    Finite-horizon (time-dependent ν), ν(s,H)=0.
    """
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence

        self.lambda_1 = config.lambda_1 # λ_1 (f-divergence constraint)
        self.lambda_2 = config.lambda_2 # λ_2 (f-divergence constraint)

        # Networks
        self.policy = DiscretePolicyMultiHead(   # time-dependent policy
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim,
            horizon=config.horizon
        ).to(device)


        # time-dependent ν
        self.nu_1 = CriticTimeVector(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            horizon=config.horizon,
            layer_norm=config.layer_norm,
        ).to(device)

        self.nu_2 = CriticTimeVector(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            horizon=config.horizon,
            layer_norm=config.layer_norm,
        ).to(device)

        if config.fair_alpha == 0.0: # ESR case
            self.mu_1 = MuNetwork(config, learnable=False).to(device)
            self.mu_2 = MuNetwork(config, learnable=False).to(device)
            self.mu_optim = None
        else: # SER case
            self.mu_1 = MuNetwork(config, learnable=True).to(device)
            self.mu_2 = MuNetwork(config, learnable=True).to(device)
            self.mu_optim = optim.Adam(list(self.mu_1.parameters()) + list(self.mu_2.parameters()), lr=config.mu_lr)

        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.nu_optim = optim.Adam(list(self.nu_1.parameters()) + list(self.nu_2.parameters()), lr=config.nu_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        
        self.step = 0

    def train(self, batch):
        return self.train_step(batch)
    
    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        if mask.dtype != torch.bool:
            mask = mask > 0
        m = mask.to(x.dtype)
        den = m.sum().clamp_min(eps)
        return (x * m).sum() / den
    
    # Loss functions
    def nu_loss_fn(self, states, next_states, timesteps, next_timesteps, rewards, initial_states):
        lambda_f_div = self.lambda_1

        nu_curr = self.nu_1(states)
        nu_next = self.nu_1(next_states)
        nu_t   = nu_curr[:, :-1]
        nu_tp1 = nu_next[:,  1:] 
        init_nu = self.nu_1(initial_states, torch.zeros_like(timesteps))  # [B]
        
        mu_1 = self.mu_1()  # [R]
        scalar = (rewards @ mu_1)
        e_nonflat      = scalar[:, None] + nu_tp1 - nu_t    # [B]
        e = e_nonflat.reshape(-1)

        # w = (f')^{-1}(e/λ); w >= 0
        state_action_ratio = f_derivative_inverse(e / lambda_f_div, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)

        # f(w)
        f_vals = f(state_action_ratio, self.f_div)

        # loss
        loss_1 = init_nu.mean()
        loss_2 = (state_action_ratio * e - lambda_f_div * f_vals).mean()
        if self.config.fair_alpha == 0.0:
            loss_3 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_3 = piecewise_log_u_star_neg_mu(mu_1).sum()
        else:
            loss_3 = piecewise_frac_u_star_neg_mu(mu_1, self.config.fair_alpha).sum()

        with torch.no_grad():
            tau = getattr(self.config, "mask_threshold", 0.0)
            mask = (state_action_ratio > tau) 

        lambda_f_div = self.lambda_2

        nu_curr_2 = self.nu_2(states)
        nu_next_2 = self.nu_2(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu_2(initial_states, torch.zeros_like(timesteps))  # [B]
        
        mu_2 = self.mu_2()  # [R]
        scalar_2 = (rewards @ mu_2)
        e_nonflat_2      = scalar_2[:, None] + nu_tp1_2 - nu_t_2    # [B]
        e_2 = e_nonflat_2.reshape(-1)
        state_action_ratio_2 = f_derivative_inverse(e_2 / lambda_f_div, self.f_div)
        state_action_ratio_2 = torch.nn.functional.relu(state_action_ratio_2)

        f_vals_2 = f(state_action_ratio_2, self.f_div)

        loss_1_2 = init_nu_2.mean()
        loss_2_2 = self.masked_mean(state_action_ratio_2 * e_2 - lambda_f_div * f_vals_2, mask)
        if self.config.fair_alpha == 0.0:
            loss_3_2 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_3_2 = piecewise_log_u_star_neg_mu(mu_2).sum()
        else:
            loss_3_2 = piecewise_frac_u_star_neg_mu(mu_2, self.config.fair_alpha).sum()

        nu_loss = loss_1 + loss_2 + loss_3
        nu_loss = nu_loss + loss_1_2 + loss_2_2 + loss_3_2

        return nu_loss, (e, loss_1, loss_2, loss_3, nu_t.mean(), nu_tp1.mean(), f_vals.mean().detach())


    def policy_loss_fn(self, states, actions, rewards, next_states):
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B,H]

        lambda_f_div_1 = self.lambda_1
        nu_curr_1 = self.nu_1(states)       # [B, H+1]
        nu_next_1 = self.nu_1(next_states)  # [B, H+1]
        nu_t_1   = nu_curr_1[:, :-1]        # [B, H]
        nu_tp1_1 = nu_next_1[:, 1:]
        
        mu_1 = self.mu_1()
        r_scalar_1 = rewards @ mu_1 # [B, H]
        e_1 = r_scalar_1[:, None] + nu_tp1_1 - nu_t_1

        state_action_ratio_1 = f_derivative_inverse(e_1 / lambda_f_div_1, self.f_div)
        state_action_ratio_1 = torch.nn.functional.relu(state_action_ratio_1)
        with torch.no_grad():
            tau = getattr(self.config, "mask_threshold", 0.0)
            mask = (state_action_ratio_1 > tau)

        
        lambda_f_div_2 = self.lambda_2
        nu_curr_2 = self.nu_2(states)
        nu_next_2 = self.nu_2(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:]

        mu_2 = self.mu_2()  # [R]
        r_scalar_2 = (rewards @ mu_2)
        e_2      = r_scalar_2[:, None] + nu_tp1_2 - nu_t_2    # [B]
        
        state_action_ratio_2 = f_derivative_inverse(e_2 / lambda_f_div_2, self.f_div)
        state_action_ratio_2 = torch.nn.functional.relu(state_action_ratio_2)
        w = state_action_ratio_2.detach()

        # w = w / (w.mean() + 1e-8)
        policy_loss = -self.masked_mean(w * log_probs, mask).mean()
        return policy_loss, w * mask



    # Training step
    def train_step(self, batch):
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states    = batch["initial_states"].to(self.device)

        # Update ν and μ
        self.nu_optim.zero_grad()
        if self.mu_optim is not None:
            self.mu_optim.zero_grad()

        nu_loss, (e, init_loss, advantage_loss, util_loss,
                  nu_vals_mean, next_nu_mean, f_vals_mean) = self.nu_loss_fn(
            states, next_states, timesteps, next_timesteps, rewards, initial_states
        )
        nu_loss.backward()
        self.nu_optim.step()
        if self.mu_optim is not None:
            self.mu_optim.step()
        self.nu_scheduler.step()

        # Update policy
        self.policy_optim.zero_grad()
        policy_loss, w = self.policy_loss_fn(
            states, actions, rewards, next_states
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
            "init_loss": float(init_loss.item()),
            "advantage_loss": float(advantage_loss.item()),
            "util_loss": float(util_loss.item()),
            "nu_vals_mean": float(nu_vals_mean.item()),
            "next_nu_mean": float(next_nu_mean.item()),
            "f_vals_mean": float(f_vals_mean.item()),
        }

    # Save / Load
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))