import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, state_dim)
        self.logvar = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)

        mean = self.mean(h)
        logvar = self.logvar(h).clamp(-5, 2)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        next_state = state + mean + eps * std
        return next_state, mean, logvar

def loss_fn(pred_mean, pred_logvar, target):
    inv_var = torch.exp(-pred_logvar)
    mse = (pred_mean - target) ** 2
    return (mse * inv_var + pred_logvar).mean()
