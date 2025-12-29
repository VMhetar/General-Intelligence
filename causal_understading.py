"""
This module learns, analyzes, and operates over latent causal factors
within a world model.

Each causal factor is represented as an independent neural module that
predicts its next state conditioned on its current state and the action.

The module supports:
- Explicit interventions (do-operations) on individual causes
- Counterfactual influence measurement
- Stability-based influence profiling
- Pruning of irrelevant or redundant causes

Assumptions:
- Causal factors are independent latent variables.
- Each cause evolves conditioned only on itself and the action.
- Cross-cause interactions are revealed via counterfactual influence.
- Meaningful causes emerge through repeated intervention and pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalUnderstanding(nn.Module):
    """
    This module learns to understand the set of causal factors in a world model.
    It models each cause as a separate neural network that predicts its next state
    Assumptions:
    - Causal factors are independent latent variables.
    - Each cause evolves conditioned only on itself and the action.
    - Cross-cause interactions are captured indirectly via influence.
    - Causes are discoverable via intervention and counterfactual impact.
    """

    def __init__(self, num_causes: int, cause_dim: int, action_dim: int):
        super().__init__()

        self.num_causes = num_causes
        self.cause_dim = cause_dim

        self.causes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cause_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, cause_dim)
            )
            for _ in range(num_causes)
        ])

    def forward(self, causes: List[torch.Tensor], action: torch.Tensor):
        """
        Args:
            causes: list of tensors, each shape [cause_dim]
            action: tensor shape [action_dim]

        Returns:
            next_causes: list of tensors, each shape [cause_dim]
        """
        next_causes = []

        for c_net, c in zip(self.causes, causes):
            delta = c_net(torch.cat([c, action], dim=-1))
            next_causes.append(c + delta)

        return next_causes

    def intervention(
        self,
        causes: List[torch.Tensor],
        action: torch.Tensor,
        intervene_idx: int,
        new_value: torch.Tensor
    ):
        """
        Performs a do-operation on a single causal slot.
        This function returns the predicted next causes after intervening.
        Args:
            causes: list of tensors, each shape [cause_dim]
            action: tensor shape [action_dim]
            intervene_idx: index of the cause to intervene on
            new_value: tensor shape [cause_dim], new value for the intervened cause
        """
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
        """
        Measures how much intervening on cause `idx` changes the future.
        Returns average counterfactual impact.
        Note:
            This measures counterfactual sensitivity, not ground-truth causality.
            True causal meaning emerges over repeated interventions and pruning.

        """
        original_next = self.forward(causes, action)

        intervened_causes = [c.clone() for c in causes]
        intervened_causes[idx] = (
            intervened_causes[idx]
            + epsilon * torch.randn_like(intervened_causes[idx])
        )

        intervened_next = self.forward(intervened_causes, action)

        influence = 0.0
        for o, i in zip(original_next, intervened_next):
            influence += (o - i).norm()

        influence /= len(original_next)
        return influence.item()

    def influence_profile(
        self,
        causes: List[torch.Tensor],
        action: torch.Tensor,
        trials: int = 5
    ) -> List[float]:
        """
        Computes stable influence estimates for each cause.
        """
        scores = []

        for i in range(len(causes)):
            trial_scores = []
            for _ in range(trials):
                trial_scores.append(
                    self.causal_influence(causes, action, i)
                )
            scores.append(
                torch.tensor(trial_scores).mean().item()
            )

        return scores
    
    def prune_causes(
        self,
        influence_scores: List[float],
        threshold: float = 1e-3
    ) -> List[int]:
        """
        Returns indices of causes worth keeping.
        """
        return [
            i for i, score in enumerate(influence_scores)
            if score > threshold
        ]
