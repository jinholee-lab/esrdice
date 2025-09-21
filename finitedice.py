import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from divergence import f, FDivergence, f_derivative_inverse
from network import CriticTime, DiscretePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR



class FiniteOptiDICE(nn.Module):
    """
    Finite-horizon OptiDICE (time-dependent ν), ν(s,H)=0.
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

        # -----------------
        # Optimizers
        # -----------------
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.nu_optim = optim.Adam(self.nu.parameters(), lr=config.nu_lr)
        # in FiniteOptiDICE.__init__
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0

    def train(self, batch):
        return self.train_step(batch)

    # -----------------
    # Loss functions
    # -----------------
    # def _e_nu(self, rewards, nu_vals, next_nu):
    #     # eν = (1/H) r + ν(s',t+1) - ν(s,t)

    #     return (1.0 / self.H) * rewards.squeeze(-1) + next_nu - nu_vals

    def nu_loss_fn(self, states, next_states, timesteps, next_timesteps, rewards, initial_states):
        nu_vals   = self.nu(states, timesteps)               # [B]
        next_nu   = self.nu(next_states, next_timesteps)     # [B]
        # e         = (1.0 / self.H) * rewards.squeeze(-1) + next_nu - nu_vals    # [B]
        e         = rewards.squeeze(-1) + next_nu - nu_vals    # [B]

        # w* = (f')^{-1}(e/α); w >= 0
        state_action_ratio = f_derivative_inverse(e / self.alpha, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)  # 음수는 0으로

        # 초기항: E[ν(s,0)] ≈ 평균( initial_states * ν(s,0) ) / (mean normalizer)
        init_nu = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]
        # init_term = init_nu.mean() / self.H  # scalar
        init_term = init_nu.mean()
        
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

        # dual min-only: E[ w* e - α f(w*) ] + init_term
        loss_term = (state_action_ratio * e - self.alpha * f(state_action_ratio, self.f_div)).mean()
        nu_loss = loss_term + gp_coeff * nu_grad_penalty + init_term

        return nu_loss, (e.detach(), state_action_ratio.detach(), 
                         nu_grad_penalty.detach(), init_term.detach(), 
                         loss_term.detach(), nu_vals.mean().detach(),
                         next_nu.mean().detach())

    def policy_loss_fn(self, states, actions, rewards, next_states, timesteps, next_timesteps):
        dist = self.policy(states, timesteps)  # [B, A]
        log_probs = dist.log_prob(actions)                    # [B]

        nu_val     = self.nu(states, timesteps)               # [B]
        next_nu_val= self.nu(next_states, next_timesteps)     # [B]
        # e      = (1.0 / self.H) * rewards.squeeze(-1) + next_nu_val - nu_val  # [B]
        e      = rewards.squeeze(-1) + next_nu_val - nu_val  # [B]

        state_action_ratio = f_derivative_inverse(e / self.alpha, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)  # 음수는 0으로
        w = state_action_ratio.detach()  # [B]
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

        # ---- Update ν ----
        self.nu_optim.zero_grad()
        nu_loss, (e, w_star, nu_grad_penalty, init_term, loss_term, nu_vals_mean, next_nu_mean) = self.nu_loss_fn(
            states, next_states, timesteps, next_timesteps, rewards, initial_states
        )
        nu_loss.backward()
        self.nu_optim.step()
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
            "w_star_mean": float(w_star.mean().item()),
            "w_star_std": float(w_star.std().item()),
            "w_star_min": float(w_star.min().item()),
            "w_star_max": float(w_star.max().item()),
            "nu_grad_penalty": float(nu_grad_penalty.item()) if isinstance(nu_grad_penalty, torch.Tensor) else 0.0,
            "init_term": float(init_term.item()),
            "loss_term": float(loss_term.item()),
            "nu_vals_mean": float(nu_vals_mean.item()),
            "next_nu_mean": float(next_nu_mean.item()),
        }

    # -----------------
    # Save / Load
    # -----------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))