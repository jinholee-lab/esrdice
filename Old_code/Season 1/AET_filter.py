from logging import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from divergence import f, FDivergence, f_derivative_inverse
from network import MuNetwork, DiscretePolicyMultiHead, CriticTimeVector
from torch.optim.lr_scheduler import CosineAnnealingLR
from conjugate_function import log_u_star_neg_mu, piecewise_log_u_star_neg_mu, frac_u_star_neg_mu, piecewise_frac_u_star_neg_mu
from utility import Utility

class FiniteAET_filter(nn.Module):
    def __init__(self, config, filter_threshold, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.filter_threshold = filter_threshold
        self.filter_f_div_threshold = config.f_divergence_filter
        self.fair_alpha = config.fair_alpha

        self.register_buffer("moving_k", torch.full((config.reward_dim,), 0.0).to(device))        # 기대 성능 벡터 k의 추세
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.register_buffer("pruning_rate", torch.full((1,), 0.0).to(device)) # Pruning rate의 추세

        self.ser_utility = Utility(
            kind=config.utility_kind,
            alpha= config.fair_alpha
            )
        
        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)
        
        # --- 2. Trainable parameters ---
        self.mu = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_lambda_f_div = nn.Parameter(torch.tensor(float(np.log(config.lambda_init))).to(device))

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---

        self.ema_alpha = 0.9 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        if config.fair_alpha != 0.0:
            params_nu += list(self.mu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        # Dual Optimizer는 Primal보다 10배 느리게 설정 (Stability)
        self.dual_optim = optim.Adam([self.log_lambda_f_div], lr=config.nu_lr) #*0.1
        self.dual_scheduler = CosineAnnealingLR(self.dual_optim, T_max=config.num_steps, eta_min=1e-6)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0

    def update_nu_mu(self, states, next_states, timesteps, rewards, initial_states):
        # Multipliers 얻기
        beta = torch.exp(self.log_lambda_f_div)
        mu = self.mu()

        # Bellman Flow 관련 계산
        nu_curr = self.nu(states)
        nu_next = self.nu(next_states)
        nu_t = nu_curr[:, :-1]
        nu_tp1 = nu_next[:, 1:]
        init_nu = self.nu(initial_states, torch.zeros_like(timesteps))
        
        weighted_reward = (rewards @ mu) # [B, H]
        e = weighted_reward[:, None] + nu_tp1 - nu_t # [B, H]

        # w = (f')^-1(e/lambda)
        w_notflat = torch.nn.functional.relu(f_derivative_inverse(e / beta, self.f_div)) # [B, H]
        w = w_notflat.reshape(-1) # [B * H]
        f_vals = f(w, self.f_div)

        expanded_reward = rewards.unsqueeze(1).expand(-1, self.H, -1)
        k = (w_notflat.unsqueeze(-1) * expanded_reward).mean(dim=(0, 1))

        # Loss 구성
        loss_1 = init_nu.mean()
        loss_2 = (w * (e.reshape(-1)) - beta * f_vals).mean()

        # Fair RL 관련 loss_3 (기존)
        if self.config.fair_alpha == 0.0:
            loss_3 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_3 = piecewise_log_u_star_neg_mu(mu).sum()
        else:
            loss_3 = piecewise_frac_u_star_neg_mu(mu, self.config.fair_alpha).sum()

        # Total nu_loss
        nu_loss = loss_1 + loss_2 + loss_3
        
        # 6. 정적 필터링 성능 모니터링을 위한 메트릭 추출
        with torch.no_grad():
            pruning_rate = (w == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return nu_loss.item(), w_notflat.detach(), k.detach(), f_vals.mean().detach(), pruning_rate.item()

    def update_meta_parameters(self, k, f_div, pruning_rate):
        """
        EMA Buffer를 갱신하고, Satisficing(lambda) 및 Robustness(beta) 파라미터를 업데이트함
        """
        beta = torch.exp(self.log_lambda_f_div)

        with torch.no_grad():
            # (1) 기대 성능 벡터 k 추적 (Weighted Returns)
            self.moving_k = self.ema_alpha * self.moving_k + (1 - self.ema_alpha) * k
            self.moving_f_div = self.ema_alpha * self.moving_f_div + (1 - self.ema_alpha) * f_div
            self.pruning_rate = self.ema_alpha * self.pruning_rate + (1 - self.ema_alpha) * pruning_rate
        
        # u(k) 계산: Nash Social Welfare (alpha=1.0) 기준 [cite: 59, 215]
        # moving_k가 0 이하로 떨어지지 않게 안정화 (특히 chi-square 필터링 시 주의)
        current_u = self.ser_utility(self.moving_k, keep_dims=False)
        # Meta-Loss Formulation:
        # L = (1 - lambda) * C + lambda * u(k) + beta * (f_div - delta)
        # B <= C 제약과 D_f <= delta 제약이 반영됨 [cite: 68, 121]
        loss_div = beta * (self.filter_f_div_threshold - self.moving_f_div)
        
        total_meta_loss = loss_div

        # 4. Optimizer Step (Two-timescale: nu_lr * 0.1)
        self.dual_optim.zero_grad()
        total_meta_loss.backward()
        self.dual_optim.step()
        self.dual_scheduler.step()

        with torch.no_grad():
            # beta >= 1e-5 제약을 위해 log(1e-5) ≈ -11.5129를 하한으로 설정.
            self.log_lambda_f_div.clamp_(min=np.log(1e-5))

        return {
            "ema_k": self.moving_k.detach().cpu().numpy().tolist(),
            "ema_f_div": self.moving_f_div.item(),
            "beta": beta.item(),
            "meta_loss": total_meta_loss.item(),
            "current_welfare": current_u.item(),
            "pruning_rate": self.pruning_rate.item()
        }

    def update_policy(self, states, actions, w_target):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -(w_target * log_probs).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss

    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        nu_loss, w_target, k, f_vals, pruning_rate = self.update_nu_mu(states, next_states, timesteps, rewards, initial_states)
        # --- 2. Update λ (f-divergence constraint) ---
        info_meta = self.update_meta_parameters(k, f_vals, pruning_rate)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target)

        self.step += 1

        log_dict = {
            "train/nu_loss": nu_loss,
            "train/f_div": f_vals,
            "train/pruning_rate": info_meta["pruning_rate"],
            "train/meta_loss": info_meta["meta_loss"],
            "train/beta": info_meta["beta"],
            "train/ema_f_div": info_meta["ema_f_div"],
            "train/utility": info_meta["current_welfare"]
        }

        # 벡터 k와 ema_k를 개별 목적별로 분리하여 저장 
        for i in range(len(k)):
            log_dict[f"k/obj_{i}"] = k[i].item()
            log_dict[f"ema_k/obj_{i}"] = info_meta["ema_k"][i] # 리스트인 경우 그대로 사용


        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class FiniteAET_after(nn.Module):
    def __init__(self, config, filter_path = None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.f_div_threshold = config.f_divergence_threshold
        self.fair_alpha = config.fair_alpha

        self.ser_utility = Utility(
            kind=config.utility_kind,
            alpha= config.fair_alpha
            )
        
        self.nu_filter = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.mu_filter = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_lambda_f_div_filter = nn.Parameter(torch.tensor(0.0).to(device))
        self.logit_lambda_sat_filter = nn.Parameter(torch.tensor(0.0).to(device))

        checkpoint = torch.load(filter_path, map_location=self.device)
        
        # 매핑 딕셔너리 (저장된 이름 -> 현재 클래스의 _1 이름)
        name_map = {
            'nu': 'nu_filter',
            'mu': 'mu_filter',
            'log_lambda_f_div': 'log_lambda_f_div_filter',
            'logit_lambda_sat': 'logit_lambda_sat_filter'
        }
        
        new_state_dict = self.state_dict()
        for old_name, new_name in name_map.items():
            prefix = old_name + "."
            new_prefix = new_name + "."
            
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    # 변수명 내부의 mu는 건드리지 않고, 최상위 객체 이름만 바꿉니다.
                    target_key = key.replace(prefix, new_prefix, 1) # 1번만 치환
                    if target_key in new_state_dict:
                        new_state_dict[target_key] = checkpoint[key]
                
        self.load_state_dict(new_state_dict)

        # Stage 1 파라미터 고정 (이게 중요합니다)
        for name, param in self.named_parameters():
            if '_filter' in name:
                param.requires_grad = False

        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)
        
        # --- 2. Trainable parameters ---
        self.mu = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_lambda_f_div = nn.Parameter(torch.tensor(float(np.log(config.lambda_init))).to(device))

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.ema_alpha = 0.0 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        if config.fair_alpha != 0.0:
            params_nu += list(self.mu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        self.dual_optim = optim.Adam([self.log_lambda_f_div], lr=config.nu_lr*0.1) 
        self.dual_scheduler = CosineAnnealingLR(self.dual_optim, T_max=config.num_steps, eta_min=1e-7)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0
    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        if mask.dtype != torch.bool:
            mask = mask > 0
        m = mask.to(x.dtype)
        den = m.sum().clamp_min(eps)
        return (x * m).sum() / den
    
    def update_nu_mu(self, states, next_states, timesteps, rewards, initial_states):
        lambda_f_div_filter = torch.exp(self.log_lambda_f_div_filter)
        logit_lambda_sat_filter = torch.exp(self.log_lambda_f_div_filter)
        
        nu_curr = self.nu_filter(states)
        nu_next = self.nu_filter(next_states)
        nu_t   = nu_curr[:, :-1]
        nu_tp1 = nu_next[:,  1:]
        
        if self.fair_alpha != 0.0:
            mu_1 = self.mu_filter() # [reward_dim]
        else:
            mu_1 = logit_lambda_sat_filter * self.mu_filter()
        
        scalar = (rewards @ mu_1)
        e_nonflat      = scalar[:, None] + nu_tp1 - nu_t    # [B]
        e = e_nonflat.reshape(-1)

        # w = (f')^{-1}(e/λ); w >= 0
        state_action_ratio = f_derivative_inverse(e / lambda_f_div_filter, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)

        with torch.no_grad():
            mask = (state_action_ratio > 0.0)

        lambda_f_div = torch.exp(self.log_lambda_f_div)

        nu_curr_2 = self.nu(states)
        nu_next_2 = self.nu(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]
        
        mu_2 = self.mu()  # [R]
        scalar_2 = (rewards @ mu_2)
        e_nonflat_2      = scalar_2[:, None] + nu_tp1_2 - nu_t_2    # [B]
        e_2 = e_nonflat_2.reshape(-1)
        state_action_ratio_2 = f_derivative_inverse(e_2 / lambda_f_div, self.f_div)
        state_action_ratio_2 = torch.nn.functional.relu(state_action_ratio_2)
        w = mask * state_action_ratio_2

        f_vals = f(w, self.f_div)

        init_loss = init_nu_2.mean()
        loss_1 = self.masked_mean((state_action_ratio_2 * e_2 - lambda_f_div * f_vals), mask)
        if self.config.fair_alpha == 0.0:
            loss_2 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_2 = piecewise_log_u_star_neg_mu(mu_2).sum()
        else:
            loss_2 = piecewise_frac_u_star_neg_mu(mu_2, self.config.fair_alpha).sum()

        nu_loss = init_loss + loss_1 + loss_2

        # For policy extraction
        state_action_ratio_2_nonflat = f_derivative_inverse(e_nonflat_2 / lambda_f_div, self.f_div)
        w_new_nonflat = torch.nn.functional.relu(state_action_ratio_2_nonflat)

        # mask_nonflat: [B, H] (Stage 1의 합격 여부)
        mask_nonflat = mask.reshape(states.shape[0], self.H)

        # 최종 가중치: Stage 1 마스킹 * Stage 2 유틸리티 가중치
        # 이 nonflat_w가 정책 학습(Weighted BC)의 직접적인 target이 됩니다.
        nonflat_w = (mask_nonflat * w_new_nonflat).detach() # [B, H]

        with torch.no_grad():
            f_div = self.masked_mean(f_vals, mask)
            pruning_rate = (w == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return f_div.detach(), nonflat_w, {
            "nu_loss": nu_loss.item(),
            "pruning_rate": pruning_rate.item()
        }
    
    def update_meta_parameters(self, f_div):
        """
        EMA Buffer를 갱신하고, Satisficing(lambda) 및 Robustness(beta) 파라미터를 업데이트함
        """
        beta = torch.exp(self.log_lambda_f_div)

        with torch.no_grad():
            # (1) f-divergence 추적 (Divergence of corrected distribution)
            self.moving_f_div = self.ema_alpha * self.moving_f_div + (1 - self.ema_alpha) * f_div
        
        loss_div = beta * (self.f_div_threshold - self.moving_f_div)
        total_meta_loss = loss_div

        # 4. Optimizer Step (Two-timescale: nu_lr * 0.1)
        self.dual_optim.zero_grad()
        total_meta_loss.backward()
        self.dual_optim.step()
        self.dual_scheduler.step()

        with torch.no_grad():
            # beta >= 1e-5 제약을 위해 log(1e-5) ≈ -11.5129를 하한으로 설정.
            self.log_lambda_f_div.clamp_(min=np.log(1e-7))

        return {
            "stage2/ema_f_div": self.moving_f_div.item(),
            "stage2/f_div_gap": (self.f_div_threshold - self.moving_f_div).item(), # 예산 여유분
            "stage2/beta": beta.item(),
            "stage2/meta_loss": total_meta_loss.item()
        }
    
    def update_policy(self, states, actions, w_target):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -(w_target * log_probs).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss
    
    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        f_vals, w_target, info_nu = self.update_nu_mu(states, next_states, timesteps, rewards, initial_states)
        # --- 2. Update λ (f-divergence constraint) ---
        info_meta = self.update_meta_parameters(f_vals)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target)

        self.step += 1

        log_dict = {
            "train/nu_loss": info_nu["nu_loss"],
            "train/f_div": f_vals,
            "train/meta_loss": info_meta["stage2/meta_loss"],
            "train/pruning_rate": info_nu["pruning_rate"],
            "train/beta": info_meta["stage2/beta"],
            "train/ema_f_div": info_meta["stage2/ema_f_div"],
            "train/f_div_gap" : info_meta["stage2/f_div_gap"]
        }
        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class FairDICE(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.f_div_threshold = config.f_divergence_threshold
        self.fair_alpha = config.fair_alpha

        self.ser_utility = Utility(
            kind=config.utility_kind,
            alpha= config.fair_alpha
            )
        
        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)
        
        # --- 2. Trainable parameters ---
        self.mu = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_constraint = nn.Parameter(torch.tensor(0.0).to(device))

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---
        self.register_buffer("moving_k", torch.full((config.reward_dim,), 0.0).to(device)) 
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.ema_alpha = 0.0 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        if config.fair_alpha != 0.0:
            params_nu += list(self.mu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        self.dual_optim = optim.Adam([self.log_lambda_f_div], lr=config.nu_lr*0.1) 
        self.dual_scheduler = CosineAnnealingLR(self.dual_optim, T_max=config.num_steps, eta_min=1e-7)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0

    def update_nu_mu(self, states, next_states, timesteps, rewards, initial_states):
        beta = torch.exp(self.log_lambda_f_div)
        eta = torch.exp(self.log_constraint)
        mu_2 = self.mu()

        nu_curr_2 = self.nu(states)
        nu_next_2 = self.nu(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]
        
        weighted_reward = (rewards @ mu_2) # [B, H]
        e = weighted_reward[:, None] + nu_tp1_2 - nu_t_2 # [B, H]

        e_augmented = e - eta * weighted_reward[:, None]

        w_notflat = torch.nn.functional.relu(f_derivative_inverse(e_augmented / beta, self.f_div)) # [B, H]
        w = w_notflat.reshape(-1) # [B * H]
        w_mean_check = w.mean()
        f_vals = f(w, self.f_div)

        expanded_reward = rewards.unsqueeze(1).expand(-1, self.H, -1)
        k = (w_notflat.unsqueeze(-1) * expanded_reward).mean(dim=(0, 1))

        init_loss = init_nu_2.mean()
        loss_1 = (w * (e_augmented.reshape(-1)) - beta * f_vals).mean()
        if self.config.fair_alpha == 0.0:
            loss_2 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_2 = piecewise_log_u_star_neg_mu(mu_2).sum()
        else:
            loss_2 = piecewise_frac_u_star_neg_mu(mu_2, self.config.fair_alpha).sum()

        nu_loss = init_loss + loss_1 + loss_2

        with torch.no_grad():
            f_div = f_vals.mean()
            pruning_rate = (w == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return f_div.detach(), w_notflat.detach(), k.detach(), {
            "nu_loss": nu_loss.item(),
            "pruning_rate": pruning_rate.item(),
            "w_mean": w_mean_check.item()
        }
    
    def update_meta_parameters(self, f_div, k):
        """
        EMA Buffer를 갱신하고, Satisficing(lambda) 및 Robustness(beta) 파라미터를 업데이트함
        """
        beta = torch.exp(self.log_lambda_f_div)

        with torch.no_grad():
            # (1) f-divergence 추적 (Divergence of corrected distribution)
            self.moving_f_div = self.ema_alpha * self.moving_f_div + (1 - self.ema_alpha) * f_div
        
        loss_div = beta * (self.f_div_threshold - self.moving_f_div)
        total_meta_loss = loss_div

        current_u = self.ser_utility(k, keep_dims=False)

        # 4. Optimizer Step (Two-timescale: nu_lr * 0.1)
        self.dual_optim.zero_grad()
        total_meta_loss.backward()
        self.dual_optim.step()
        self.dual_scheduler.step()

        with torch.no_grad():
            # beta >= 1e-5 제약을 위해 log(1e-5) ≈ -11.5129를 하한으로 설정.
            self.log_lambda_f_div.clamp_(min=np.log(1e-7))

        return {
            "stage2/ema_f_div": self.moving_f_div.item(),
            "stage2/f_div_gap": (self.f_div_threshold - self.moving_f_div).item(), # 예산 여유분
            "stage2/beta": beta.item(),
            "stage2/meta_loss": total_meta_loss.item(),
            "current_welfare": current_u.item()
        }
    
    def update_policy(self, states, actions, w_target):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -(w_target * log_probs).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss
    
    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        f_vals, w_target, k, info_nu = self.update_nu_mu(states, next_states, timesteps, rewards, initial_states)
        # --- 2. Update λ (f-divergence constraint) ---
        info_meta = self.update_meta_parameters(f_vals, k)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target)

        self.step += 1

        log_dict = {
            "train/nu_loss": info_nu["nu_loss"],
            "train/f_div": f_vals,
            "train/meta_loss": info_meta["stage2/meta_loss"],
            "train/pruning_rate": info_nu["pruning_rate"],
            "train/beta": info_meta["stage2/beta"],
            "train/ema_f_div": info_meta["stage2/ema_f_div"],
            "train/f_div_gap" : info_meta["stage2/f_div_gap"],
            "current_welfare" : info_meta["current_welfare"],
            "w_mean_check" : info_nu["w_mean"]
        }
        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class ESRDICE(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.f_div_filter = config.f_divergence_filter
        self.fair_alpha = config.fair_alpha

        self.ser_utility = Utility(
            kind=config.utility_kind,
            alpha= config.fair_alpha
            )
        
        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)
        
        # --- 2. Trainable parameters ---
        #self.log_lambda_f_div = nn.Parameter(torch.tensor(float(np.log(config.lambda_init))).to(device))
        self.log_constraint = nn.Parameter(torch.tensor(-1.0).to(device))

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---
        self.register_buffer("moving_k", torch.full((config.reward_dim,), 0.0).to(device)) 
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.ema_alpha = 0.0 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        self.dual_optim = optim.Adam([self.log_constraint], lr=config.nu_lr) 
        self.dual_scheduler = CosineAnnealingLR(self.dual_optim, T_max=config.num_steps, eta_min=1e-6)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0

    def update_nu_mu(self, states, next_states, timesteps, rewards, initial_states):
        beta = self.f_div_filter
        eta = torch.exp(self.log_constraint)

        nu_curr_2 = self.nu(states)
        nu_next_2 = self.nu(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]
        
        total_reward = rewards.sum(dim=-1)

        e = total_reward[:, None] + nu_tp1_2 - nu_t_2 # [B, H]

        e_augmented = e - eta * total_reward[:, None]

        w_notflat = torch.nn.functional.relu(f_derivative_inverse(e_augmented / beta, self.f_div)) # [B, H]
        w = w_notflat.reshape(-1) # [B * H]
        w_mean_check = w.mean()
        f_vals = f(w, self.f_div)

        expanded_reward = rewards.unsqueeze(1).expand(-1, self.H, -1)
        k = (w_notflat.unsqueeze(-1) * expanded_reward).mean()

        init_loss = init_nu_2.mean()
        loss_1 = (w * (e_augmented.reshape(-1)) - beta * f_vals).mean()

        nu_loss = init_loss + loss_1

        with torch.no_grad():
            f_div = f_vals.mean()
            pruning_rate = (w == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return f_div.detach(), w_notflat.detach(), k.detach(), {
            "nu_loss": nu_loss.item(),
            "pruning_rate": pruning_rate.item(),
            "w_mean": w_mean_check.item()
        }
    
    def update_meta_parameters(self, f_div, k):
        """
        EMA Buffer를 갱신하고, Satisficing(lambda) 및 Robustness(beta) 파라미터를 업데이트함
        """
        C_limit = 0.23
        beta = self.f_div_filter
        eta = torch.exp(self.log_constraint)


        with torch.no_grad():
            # (1) f-divergence 추적 (Divergence of corrected distribution)
            self.moving_f_div = self.ema_alpha * self.moving_f_div + (1 - self.ema_alpha) * f_div
        
        #loss_div = beta * (self.f_div_filter - self.moving_f_div)
        loss_constraint = -eta * (k.detach() - C_limit)

        total_meta_loss = loss_constraint

        current_u = self.ser_utility(k, keep_dims=False)

        # 4. Optimizer Step (Two-timescale: nu_lr * 0.1)
        self.dual_optim.zero_grad()
        total_meta_loss.backward()
        self.dual_optim.step()
        self.dual_scheduler.step()

        #with torch.no_grad():
            # beta >= 1e-5 제약을 위해 log(1e-5) ≈ -11.5129를 하한으로 설정.
            #self.log_lambda_f_div.clamp_(min=np.log(1e-7))

        return {
            "stage2/ema_f_div": self.moving_f_div.item(),
            "stage2/f_div_gap": (self.f_div_filter - self.moving_f_div).item(), # 예산 여유분
            "stage2/meta_loss": total_meta_loss.item(),
            "current_welfare": current_u.item(),
            "stage2/eta": eta.item()
        }
    
    def update_policy(self, states, actions, w_target):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -(w_target * log_probs).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss
    
    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        f_vals, w_target, k, info_nu = self.update_nu_mu(states, next_states, timesteps, rewards, initial_states)
        # --- 2. Update λ (f-divergence constraint) ---
        info_meta = self.update_meta_parameters(f_vals, k)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target)

        self.step += 1

        log_dict = {
            "train/nu_loss": info_nu["nu_loss"],
            "train/f_div": f_vals,
            "train/meta_loss": info_meta["stage2/meta_loss"],
            "train/pruning_rate": info_nu["pruning_rate"],
            "train/ema_f_div": info_meta["stage2/ema_f_div"],
            "train/f_div_gap" : info_meta["stage2/f_div_gap"],
            "current_welfare" : info_meta["current_welfare"],
            "w_mean_check" : info_nu["w_mean"],
            "train/eta": info_meta["stage2/eta"]
        }
        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class ESRDICE_after(nn.Module):
    def __init__(self, config, filter_path = None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.f_div_filter = config.f_divergence_filter
        self.f_div_threshold = config.f_divergence_threshold
        self.fair_alpha = config.fair_alpha
        
        self.nu_filter = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.log_constraint_filter = nn.Parameter(torch.tensor(-1.0).to(device))

        checkpoint = torch.load(filter_path, map_location=self.device)
        
        # 매핑 딕셔너리 (저장된 이름 -> 현재 클래스의 _1 이름)
        name_map = {
            'nu': 'nu_filter',
            'log_constraint': 'log_constraint_filter'
        }
        
        new_state_dict = self.state_dict()
        for old_name, new_name in name_map.items():
            prefix = old_name + "."
            new_prefix = new_name + "."
            
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    # 변수명 내부의 mu는 건드리지 않고, 최상위 객체 이름만 바꿉니다.
                    target_key = key.replace(prefix, new_prefix, 1) # 1번만 치환
                    if target_key in new_state_dict:
                        new_state_dict[target_key] = checkpoint[key]
                
        self.load_state_dict(new_state_dict)

        # Stage 1 파라미터 고정 (이게 중요합니다)
        for name, param in self.named_parameters():
            if '_filter' in name:
                param.requires_grad = False

        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.ema_alpha = 0.0 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0
    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        if mask.dtype != torch.bool:
            mask = mask > 0
        m = mask.to(x.dtype)
        den = m.sum().clamp_min(eps)
        return (x * m).sum() / den
    
    def update_nu_mu(self, states, next_states, timesteps, rewards, initial_states):
        beta = self.f_div_filter
        eta = torch.exp(self.log_constraint_filter)

        nu_curr_2 = self.nu_filter(states)
        nu_next_2 = self.nu_filter(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu_filter(initial_states, torch.zeros_like(timesteps))  # [B]
        
        total_reward = rewards.sum(dim=-1)

        e = total_reward[:, None] + nu_tp1_2 - nu_t_2 # [B, H]

        e_augmented = e - eta * total_reward[:, None]

        w_notflat = torch.nn.functional.relu(f_derivative_inverse(e_augmented / beta, self.f_div)) # [B, H]
        w = w_notflat.reshape(-1) # [B * H]

        with torch.no_grad():
            mask = (w > 0.0)

        lambda_f_div = self.f_div_threshold

        nu_curr_2 = self.nu(states)
        nu_next_2 = self.nu(next_states)
        nu_t_2   = nu_curr_2[:, :-1]
        nu_tp1_2 = nu_next_2[:,  1:] 
        init_nu_2 = self.nu(initial_states, torch.zeros_like(timesteps))  # [B]

        total_reward = rewards.sum(dim=-1)
        e_nonflat_2      = total_reward[:, None] + nu_tp1_2 - nu_t_2    # [B]
        e_2 = e_nonflat_2.reshape(-1)
        state_action_ratio_2 = f_derivative_inverse(e_2 / lambda_f_div, self.f_div)
        state_action_ratio_2 = torch.nn.functional.relu(state_action_ratio_2)
        w = mask * state_action_ratio_2

        f_vals = f(w, self.f_div)

        init_loss = init_nu_2.mean()
        loss_1 = self.masked_mean((state_action_ratio_2 * e_2 - lambda_f_div * f_vals), mask)

        nu_loss = init_loss + loss_1

        # For policy extraction
        state_action_ratio_2_nonflat = f_derivative_inverse(e_nonflat_2 / lambda_f_div, self.f_div)
        w_new_nonflat = torch.nn.functional.relu(state_action_ratio_2_nonflat)

        # mask_nonflat: [B, H] (Stage 1의 합격 여부)
        mask_nonflat = mask.reshape(states.shape[0], self.H)

        # 최종 가중치: Stage 1 마스킹 * Stage 2 유틸리티 가중치
        # 이 nonflat_w가 정책 학습(Weighted BC)의 직접적인 target이 됩니다.
        nonflat_w = (mask_nonflat * w_new_nonflat).detach() # [B, H]

        with torch.no_grad():
            f_div = self.masked_mean(f_vals, mask)
            pruning_rate = (w == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return f_div.detach(), nonflat_w, {
            "nu_loss": nu_loss.item(),
            "pruning_rate": pruning_rate.item()
        }
    
    def update_policy(self, states, actions, w_target):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -(w_target * log_probs).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss
    
    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        f_vals, w_target, info_nu = self.update_nu_mu(states, next_states, timesteps, rewards, initial_states)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target)

        self.step += 1

        log_dict = {
            "train/nu_loss": info_nu["nu_loss"],
            "train/f_div": f_vals,
            "train/pruning_rate": info_nu["pruning_rate"]
        }
        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class FiniteAET_postfilter(nn.Module):
    def __init__(self, config, filter_path = None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.H = config.horizon
        self.f_div = config.f_divergence
        self.filter_f_div_threshold = config.f_divergence_filter
        self.f_div_threshold = config.f_divergence_threshold
        self.fair_alpha = config.fair_alpha

        self.register_buffer("moving_k", torch.full((config.reward_dim,), 0.0).to(device))        # 기대 성능 벡터 k의 추세
        self.register_buffer("moving_f_div", torch.full((1,), 0.0).to(device)) # Divergence의 추세
        self.register_buffer("pruning_rate", torch.full((1,), 0.0).to(device)) # Pruning rate의 추세

        self.ser_utility = Utility(
            kind=config.utility_kind,
            alpha= config.fair_alpha
            )

        self.nu_filter = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.mu_filter = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_lambda_f_div_filter = nn.Parameter(torch.tensor(0.0).to(device))

        checkpoint = torch.load(filter_path, map_location=self.device)
        
        # 매핑 딕셔너리 (저장된 이름 -> 현재 클래스의 _1 이름)
        name_map = {
            'nu': 'nu_filter',
            'mu': 'mu_filter',
            'log_lambda_f_div': 'log_lambda_f_div_filter'
        }
        
        new_state_dict = self.state_dict()
        for old_name, new_name in name_map.items():
            prefix = old_name + "."
            new_prefix = new_name + "."
            
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    # 변수명 내부의 mu는 건드리지 않고, 최상위 객체 이름만 바꿉니다.
                    target_key = key.replace(prefix, new_prefix, 1) # 1번만 치환
                    if target_key in new_state_dict:
                        new_state_dict[target_key] = checkpoint[key]
                
        self.load_state_dict(new_state_dict)

        # Stage 1 파라미터 고정 (이게 중요합니다)
        for name, param in self.named_parameters():
            if '_filter' in name:
                param.requires_grad = False        

        # --- 1. Networks ---
        self.nu = CriticTimeVector(config.state_dim, config.hidden_dims, config.horizon, config.layer_norm).to(device)
        self.policy = DiscretePolicyMultiHead(config.state_dim, config.hidden_dims, config.action_dim, config.horizon).to(device)
        
        # --- 2. Trainable parameters ---
        self.mu = MuNetwork(config, learnable=(config.fair_alpha != 0.0)).to(device)
        self.log_lambda_f_div = nn.Parameter(torch.tensor(float(np.log(config.lambda_init))).to(device))

        # --- 3. EMA Buffers (Philosophy: Smoothing the Constraints) ---

        self.ema_alpha = 0.0 # 묵직하게 추세를 따라가도록 설정

        # --- 4. Optimizers (Two-timescale) ---
        params_nu = list(self.nu.parameters())
        if config.fair_alpha != 0.0:
            params_nu += list(self.mu.parameters())
        self.nu_optim = optim.Adam(params_nu, lr=config.nu_lr)
        self.nu_scheduler = CosineAnnealingLR(self.nu_optim, T_max=config.num_steps, eta_min=1e-6)
        # Dual Optimizer는 Primal보다 10배 느리게 설정 (Stability)
        self.dual_optim = optim.Adam([self.log_lambda_f_div], lr=config.nu_lr*0.1) #*0.1
        self.dual_scheduler = CosineAnnealingLR(self.dual_optim, T_max=config.num_steps, eta_min=1e-7)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_scheduler = CosineAnnealingLR(self.policy_optim, T_max=config.num_steps, eta_min=1e-6)

        self.step = 0

    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        if mask.dtype != torch.bool:
            mask = mask > 0
        m = mask.to(x.dtype)
        den = m.sum().clamp_min(eps)
        return (x * m).sum() / den
    
    def get_mask(self, states, next_states, rewards):
        lambda_f_div_filter = torch.exp(self.log_lambda_f_div_filter)
        
        nu_curr = self.nu_filter(states)
        nu_next = self.nu_filter(next_states)
        nu_t   = nu_curr[:, :-1]
        nu_tp1 = nu_next[:,  1:]

        mu_1 = self.mu_filter()
        
        scalar = (rewards @ mu_1)
        e_nonflat      = scalar[:, None] + nu_tp1 - nu_t    # [B, H]

        # w = (f')^{-1}(e/λ); w >= 0
        state_action_ratio = f_derivative_inverse(e_nonflat / lambda_f_div_filter, self.f_div)
        state_action_ratio = torch.nn.functional.relu(state_action_ratio)

        with torch.no_grad():
            mask = (state_action_ratio > 0.0)
        return mask # [B, H]

    def update_nu_mu(self, mask, states, next_states, timesteps, rewards, initial_states):
        # Multipliers 얻기
        beta = torch.exp(self.log_lambda_f_div)
        mu = self.mu()

        # Bellman Flow 관련 계산
        nu_curr = self.nu(states)
        nu_next = self.nu(next_states)
        nu_t = nu_curr[:, :-1]
        nu_tp1 = nu_next[:, 1:]
        init_nu = self.nu(initial_states, torch.zeros_like(timesteps))
        
        weighted_reward = (rewards @ mu) # [B, H]
        e = weighted_reward[:, None] + nu_tp1 - nu_t # [B, H]

        # w = (f')^-1(e/lambda)
        w_notflat = torch.nn.functional.relu(f_derivative_inverse(e / beta, self.f_div)) # [B, H]
        masked_w_notflat = mask * w_notflat # [B, H]
        masked_w_flat = masked_w_notflat.reshape(-1) # [B * H]
        f_vals = f(masked_w_flat, self.f_div)

        #expanded_reward = rewards.unsqueeze(1).expand(-1, self.H, -1)
        #k = (masked_w * expanded_reward).mean(dim=(0, 1))

        # Loss 구성
        loss_1 = init_nu.mean()
        loss_2 = self.masked_mean(masked_w_flat * (e.reshape(-1)) - beta * f_vals, masked_w_flat)

        # Fair RL 관련 loss_3 (기존)
        if self.config.fair_alpha == 0.0:
            loss_3 = torch.tensor(0.0, device=states.device)
        elif self.config.fair_alpha == 1.0:
            loss_3 = piecewise_log_u_star_neg_mu(mu).sum()
        else:
            loss_3 = piecewise_frac_u_star_neg_mu(mu, self.config.fair_alpha).sum()

        # Total nu_loss
        nu_loss = loss_1 + loss_2 + loss_3
        
        # 6. 정적 필터링 성능 모니터링을 위한 메트릭 추출
        with torch.no_grad():
            pruning_rate = (masked_w_flat == 0).float().mean() # 필터링되어 배제된 데이터의 비율
        
        self.nu_optim.zero_grad()
        nu_loss.backward()
        self.nu_optim.step()
        self.nu_scheduler.step()

        return nu_loss.item(), masked_w_notflat.detach(), f_vals.mean().detach(), pruning_rate.item()

    def update_meta_parameters(self, f_div, pruning_rate):
        """
        EMA Buffer를 갱신하고, Satisficing(lambda) 및 Robustness(beta) 파라미터를 업데이트함
        """
        beta = torch.exp(self.log_lambda_f_div)

        with torch.no_grad():
            # (1) 기대 성능 벡터 k 추적 (Weighted Returns)
            #self.moving_k = self.ema_alpha * self.moving_k + (1 - self.ema_alpha) * k
            self.moving_f_div = self.ema_alpha * self.moving_f_div + (1 - self.ema_alpha) * f_div
            self.pruning_rate = self.ema_alpha * self.pruning_rate + (1 - self.ema_alpha) * pruning_rate
        
        # u(k) 계산: Nash Social Welfare (alpha=1.0) 기준 [cite: 59, 215]
        # moving_k가 0 이하로 떨어지지 않게 안정화 (특히 chi-square 필터링 시 주의)
        #current_u = self.ser_utility(self.moving_k, keep_dims=False)
        # Meta-Loss Formulation:
        # L = (1 - lambda) * C + lambda * u(k) + beta * (f_div - delta)
        # B <= C 제약과 D_f <= delta 제약이 반영됨 [cite: 68, 121]
        loss_div = beta * (self.f_div_threshold - self.moving_f_div)
        
        total_meta_loss = loss_div

        # 4. Optimizer Step (Two-timescale: nu_lr * 0.1)
        self.dual_optim.zero_grad()
        total_meta_loss.backward()
        self.dual_optim.step()
        self.dual_scheduler.step()

        with torch.no_grad():
            # beta >= 1e-5 제약을 위해 log(1e-5) ≈ -11.5129를 하한으로 설정.
            self.log_lambda_f_div.clamp_(min=np.log(1e-7))

        return {
            "beta": beta.item(),
            "meta_loss": total_meta_loss.item(),
            "pruning_rate": self.pruning_rate.item()
        }

    def update_policy(self, states, actions, w_target, mask):
        """
        Satisficing 가중치 w를 사용하여 정책을 업데이트합니다.
        """
        logits = self.policy(states)
        logp_all = logits.log_softmax(dim=-1)
        idx = actions.view(-1, 1, 1).expand(-1, logp_all.size(1), 1)
        log_probs = torch.gather(logp_all, dim=-1, index=idx).squeeze(-1) # [B, H]

        # Weighted MLE Loss: 가중치가 높은(유익한) 샘플의 확률을 높임
        policy_loss = -self.masked_mean(w_target * log_probs, mask)
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        return policy_loss

    def train_step(self, batch):
        """
        1. Critic(nu), Utility(mu), Satisficing Multiplier 최적화
        2. Divergence constraint(lambda) 최적화
        3. Policy 최적화
        """
        states         = batch["states"].to(self.device)
        next_states    = batch["next_states"].to(self.device)
        actions        = batch["actions"].to(self.device)
        rewards        = batch["rewards"].to(self.device)
        timesteps      = batch["timesteps"].to(self.device)
        next_timesteps = batch["next_timesteps"].to(self.device)
        initial_states = batch["initial_states"].to(self.device)

        mask = self.get_mask(states, next_states, rewards)

        # nu_loss_fn은 우리가 정의한 Satisficing Lagrangian의 Dual을 계산함
        nu_loss, w_target, f_vals, pruning_rate = self.update_nu_mu(mask, states, next_states, timesteps, rewards, initial_states)
        # --- 2. Update λ (f-divergence constraint) ---
        info_meta = self.update_meta_parameters(f_vals, pruning_rate)

        # --- 3. Update Policy (Satisficing Objective 사용) ---
        info_policy = self.update_policy(states, actions, w_target, mask)

        self.step += 1

        log_dict = {
            "train/nu_loss": nu_loss,
            "train/f_div": f_vals,
            "train/pruning_rate": info_meta["pruning_rate"],
            "train/meta_loss": info_meta["meta_loss"],
            "train/beta": info_meta["beta"]
        }

        return log_dict

    # --- Persistence ---
    def save(self, path):
        """모든 네트워크 파라미터와 Satisficing 승수(logit_lambda_sat)를 저장합니다."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """저장된 체크포인트로부터 상태를 복구합니다."""
        self.load_state_dict(torch.load(path, map_location=self.device))