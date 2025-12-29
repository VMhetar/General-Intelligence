import torch
import torch.nn as nn

class IdentityEncoder(nn.Module):
    def forward(self, obs):
        return obs
