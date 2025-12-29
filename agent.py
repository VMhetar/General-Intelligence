import torch
from typing import List

from world_model import WorldModel
from causal_understanding import CausalUnderstanding
from llm import interpret_world_model


class CausalAgent:
    """
    Orchestrates world prediction, causal analysis, and language interpretation.
    """

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

        # Optional long-term belief memory
        self.beliefs = []

    # --------------------------------------------------
    # Latent → causal slots
    # --------------------------------------------------
    def split_into_causes(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Splits latent vector z into causal slots.
        """
        assert (
            z.shape[-1] == self.num_causes * self.cause_dim
        ), "z_dim must equal num_causes * cause_dim"

        return [
            z[..., i * self.cause_dim : (i + 1) * self.cause_dim]
            for i in range(self.num_causes)
        ]

    # --------------------------------------------------
    # One cognition step
    # --------------------------------------------------
    async def step(self, observation, action):
        """
        Full cognitive step:
        encode → predict → analyze → interpret
        """

        # 1. Encode observation into latent state
        z = self.encoder(observation)

        # 2. Predict physics
        z_next, logvar, reward, done = self.world_model(z, action)

        # 3. Decompose into causal variables
        causes = self.split_into_causes(z)

        # 4. Measure causal influence
        influence_scores = self.causal_model.influence_profile(
            causes, action
        )

        # 5. Prune weak causes
        active_causes = self.causal_model.prune_causes(
            influence_scores
        )

        # 6. Interpret with language (no gradients, no physics)
        interpretation = await interpret_world_model(
            self.world_model, z, action
        )

        # 7. Store belief (optional)
        self.beliefs.append({
            "influence": influence_scores,
            "active_causes": active_causes,
            "interpretation": interpretation
        })

        return {
            "z_next": z_next,
            "reward": reward,
            "done": done,
            "logvar": logvar,
            "influence": influence_scores,
            "active_causes": active_causes,
            "interpretation": interpretation
        }
