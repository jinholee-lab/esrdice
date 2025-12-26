
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# from divergence_torch import f, FDivergence, f_derivative_inverse

# -----------------
# Critic(s,t): ν(s,t)
# -----------------
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

class CriticTime(nn.Module):
    def __init__(self, state_dim, hidden_dims, horizon, time_embed_dim=8, layer_norm=False):
        super().__init__()
        self.horizon = horizon
        self.time_emb = nn.Embedding(horizon + 1, time_embed_dim)
        self.state_dim = state_dim
        self.time_embed_dim = time_embed_dim
        self.in_dim = state_dim + time_embed_dim
        dims = [self.in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, states, timesteps):
        t = timesteps.squeeze(-1).clamp(min=0, max=self.horizon)  # [B]
        emb = self.time_emb(t)                                    # [B, D]
        x = torch.cat([states, emb], dim=-1)
        out = self.net(x).squeeze(-1)                             # [B]
        # ★ ν(s,H)=0: t==H에서 0
        return out * (t < self.horizon).float()

class CriticTimeVector(nn.Module):
    """
    ν(s, 0..H)를 한 번에 출력.
    - 출력 shape: [B, H+1]
    - 경계조건: ν(s, H) = 0 (마지막 성분을 항상 0으로 마스킹)
    - timesteps를 넘기면 해당 시점의 스칼라를 반환 (기존 인터페이스 호환)
    """
    def __init__(self, state_dim, hidden_dims, horizon, layer_norm=False):
        super().__init__()
        assert isinstance(horizon, int) and horizon >= 0
        self.horizon = horizon

        dims = [state_dim] + hidden_dims + [horizon + 1]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # ν(s, H)=0을 강제하는 마스크 (학습되지 않음)
        mask = torch.ones(1, horizon + 1)
        mask[:, -1] = 0.0
        self.register_buffer("terminal_mask", mask)

    def forward(self, states, timesteps=None):
        """
        states: [B, state_dim]
        timesteps (optional): [B] or [B,1], 각 샘플의 t (0..H로 클램프)
        """
        out = self.net(states)                  # [B, H+1]
        out = out * self.terminal_mask          # 마지막 성분 0으로 고정

        if timesteps is None:
            return out                          # [B, H+1]

        # 편의: 특정 t의 값을 바로 반환 (기존 사용 코드 호환)
        t = timesteps.squeeze(-1).long().clamp(min=0, max=self.horizon)  # [B]
        return out.gather(1, t.unsqueeze(1)).squeeze(1)                  # [B]

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, horizon, time_embed_dim=8):
        super().__init__()
        self.time_emb = nn.Embedding(horizon+1, time_embed_dim)
        dims = [state_dim + time_embed_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.logits = nn.Linear(dims[-1], action_dim)

    def forward(self, states, timesteps):
        t = timesteps.squeeze(-1)
        emb = self.time_emb(t)
        x = torch.cat([states, emb], dim=-1)
        logits = self.logits(self.mlp(x))
        return Categorical(logits=logits)

class DiscretePolicyMultiHead(nn.Module):
    """
    시간 임베딩 없이, H개의 시점별 head를 가지는 policy.
    - 입력:  states [B, state_dim]
    - 출력: logits [B, H, action_dim]
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int],
        action_dim: int,
        horizon: int,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)

        # torso (공통 feature extractor)
        torso_layers = []
        dims = [state_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            torso_layers.append(nn.Linear(dims[i], dims[i+1]))
            if layer_norm:
                torso_layers.append(nn.LayerNorm(dims[i+1]))
            torso_layers.append(nn.ReLU())
            if dropout > 0:
                torso_layers.append(nn.Dropout(dropout))
        self.torso = nn.Sequential(*torso_layers) if torso_layers else nn.Identity()

        last_hidden = dims[-1] if hidden_dims else state_dim
        # 시점별 head: H × A 출력
        self.head = nn.Linear(last_hidden, horizon * action_dim)

        # 초기화 (optional)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight, gain=0.7)

    def forward(self, states, layout="time_last"):
        """
        states: [B, state_dim]
        layout:
          - "time_last"  -> [B, H, A]
          - "logits_time"-> [B, A, H]
        """
        B = states.size(0)
        h = self.torso(states)                  # [B, hidden]
        logits_flat = self.head(h)              # [B, H*A]
        logits = logits_flat.view(B, self.horizon, self.action_dim)  # [B, H, A]

        if layout == "logits_time":
            logits = logits.permute(0, 2, 1)    # [B, A, H]
        elif layout != "time_last":
            raise ValueError(f"Unknown layout: {layout}")
        return logits

    def get_dist(self, states, timesteps):
        """
        특정 timestep의 분포를 반환 (t ∈ [0, H-1])
        """
        logits = self.forward(states, layout="time_last")  # [B, H, A]
        t = timesteps.squeeze(-1).long().clamp(0, self.horizon - 1)
        logits_t = logits[torch.arange(states.size(0)), t]  # [B, A]
        return Categorical(logits=logits_t)