import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalUnderstanding(nn.Module):
    def __init__(self, num_causes, cause_dim, action_dim):
        super().__init__()
        self.causes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cause_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, cause_dim)
            ) for _ in range(num_causes)
        ])
    
    def forward(self, causes, action):
        next_causes = []
        for c_net, c in zip(self.causes, causes):
            next_causes.append(c + c_net(torch.cat([c, action], -1)))
        return next_causes
    
    def intervention(self, causes, action, intervene_idx, new_value):
        intervened_causes = causes.copy()
        intervened_causes[intervene_idx] = new_value
        return self.forward(intervened_causes, action)
    
    def causal_influence(self, causes, action, idx, epsilon = 1e-2):
        original_next = self.forward(causes, action)

        intervened_causes = causes.copy()
        intervened_causes[idx] = (
            intervened_causes[idx] + epsilon * torch.randn_like(intervened_causes[idx])
        )

        intervened_next = self.forward(intervened_causes, action)

        influence = 0.0
        
        for o, i in zip(original_next, intervened_next):
            influence += (o - i).norm()

        return influence.item()