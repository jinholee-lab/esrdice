
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# from divergence_torch import f, FDivergence, f_derivative_inverse

# -----------------
# Critic(s,t): ν(s,t)
# -----------------
class CriticTime(nn.Module):
    def __init__(self, state_dim, hidden_dims, horizon, time_embed_dim=8, layer_norm=False):
        super().__init__()
        self.horizon = horizon
        self.time_emb = nn.Embedding(horizon + 1, time_embed_dim)

        in_dim = state_dim + time_embed_dim
        dims = [in_dim] + hidden_dims + [1]
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


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()
        dims = [state_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.logits = nn.Linear(dims[-1], action_dim)

    def forward(self, states):
        x = self.mlp(states)
        logits = self.logits(x)
        return Categorical(logits=logits)
