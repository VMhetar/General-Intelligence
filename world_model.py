import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, z_dim, action_dim, hidden=256):
        super().__init__()
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

    def forward(self, z, a):
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
