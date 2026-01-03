import torch
from typing import List, Dict, Any

from world_model import WorldModel
from causal_understanding import CausalUnderstanding
from llm import interpret_world_model


class CausalAgent:
    def __init__(
        self,
        encoder,
        world_model: WorldModel,
        causal_model: CausalUnderstanding,
    ):
        self.encoder = encoder
        self.world_model = world_model
        self.causal_model = causal_model

        self.num_causes = causal_model.num_causes
        self.cause_dim = causal_model.cause_dim

        self.beliefs: List[Dict[str, Any]] = []

    def split_into_causes(self, z: torch.Tensor) -> List[torch.Tensor]:
        expected_dim = self.num_causes * self.cause_dim
        actual_dim = z.shape[-1]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Dimension mismatch: z has {actual_dim}, expected {expected_dim}"
            )

        causes = []
        for i in range(self.num_causes):
            start_idx = i * self.cause_dim
            end_idx = start_idx + self.cause_dim
            causes.append(z[..., start_idx:end_idx])
        
        return causes

    async def step(self, observation: torch.Tensor, action: torch.Tensor) -> Dict[str, Any]:
        z = self.encoder(observation)

        z_next, logvar, reward, done = self.world_model(z, action)

        causes = self.split_into_causes(z)

        influence_scores = self.causal_model.influence_profile(causes, action)

        active_causes = self.causal_model.prune_causes(influence_scores)

        interpretation = await interpret_world_model(self.world_model, z, action)

        belief = {
            "influence": influence_scores,
            "active_causes": active_causes,
            "interpretation": interpretation
        }
        self.beliefs.append(belief)

        return {
            "z_next": z_next,
            "reward": reward,
            "done": done,
            "logvar": logvar,
            "influence": influence_scores,
            "active_causes": active_causes,
            "interpretation": interpretation
        }
