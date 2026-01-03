import torch
import torch.nn as nn
from typing import List


class CausalUnderstanding(nn.Module):
    def __init__(self, num_causes: int, cause_dim: int, action_dim: int):
        super().__init__()

        self.num_causes = num_causes
        self.cause_dim = cause_dim
        self.action_dim = action_dim

        self.causes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cause_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, cause_dim)
            )
            for _ in range(num_causes)
        ])

    def forward(self, causes: List[torch.Tensor], action: torch.Tensor) -> List[torch.Tensor]:
        if len(causes) != self.num_causes:
            raise ValueError(f"Expected {self.num_causes} causes, got {len(causes)}")

        next_causes = []
        for c_net, c in zip(self.causes, causes):
            input_tensor = torch.cat([c, action], dim=-1)
            delta = c_net(input_tensor)
            next_causes.append(c + delta)

        return next_causes

    def intervention(
        self,
        causes: List[torch.Tensor],
        action: torch.Tensor,
        intervene_idx: int,
        new_value: torch.Tensor
    ) -> List[torch.Tensor]:
        if not (0 <= intervene_idx < len(causes)):
            raise IndexError(f"intervene_idx {intervene_idx} out of range [0, {len(causes)})")

        intervened_causes = [c.clone() for c in causes]
        intervened_causes[intervene_idx] = new_value.clone()

        return self.forward(intervened_causes, action)

    def causal_influence(
        self,
        causes: List[torch.Tensor],
        action: torch.Tensor,
        idx: int,
        epsilon: float = 1e-2
    ) -> float:
        if not (0 <= idx < len(causes)):
            raise IndexError(f"idx {idx} out of range [0, {len(causes)})")

        with torch.no_grad():
            original_next = self.forward(causes, action)

            intervened_causes = [c.clone() for c in causes]
            noise = torch.randn_like(intervened_causes[idx]) * epsilon
            intervened_causes[idx] = intervened_causes[idx] + noise

            intervened_next = self.forward(intervened_causes, action)

            total_influence = sum(
                (o - i).norm().item()
                for o, i in zip(original_next, intervened_next)
            )
            
            return total_influence / len(original_next)

    def influence_profile(
        self,
        causes: List[torch.Tensor],
        action: torch.Tensor,
        trials: int = 5
    ) -> List[float]:
        scores = []

        for i in range(len(causes)):
            trial_scores = [
                self.causal_influence(causes, action, i)
                for _ in range(trials)
            ]
            scores.append(sum(trial_scores) / len(trial_scores))

        return scores
    
    def prune_causes(
        self,
        influence_scores: List[float],
        threshold: float = 1e-3
    ) -> List[int]:
        return [
            i for i, score in enumerate(influence_scores)
            if score > threshold
        ]
