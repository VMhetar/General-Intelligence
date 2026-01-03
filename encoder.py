import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
