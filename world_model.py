import torch
import torch.nn as nn
from typing import Tuple


class WorldModel(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.delta = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.reward = nn.Linear(hidden, 1)
        self.done = nn.Linear(hidden, 1)

    def forward(
        self, 
        z: torch.Tensor, 
        a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)
        h = self.net(x)

        delta = self.delta(h)
        logvar = self.logvar(h).clamp(-5, 2)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z_next = z + delta + eps * std
        r = self.reward(h)
        d = torch.sigmoid(self.done(h))

        return z_next, logvar, r, d
