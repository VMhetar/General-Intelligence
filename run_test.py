import torch
import asyncio

from toy_env import ToyPhysicsEnv
from encoder import IdentityEncoder
from world_model import WorldModel
from causal_understanding import CausalUnderstanding
from agent import CausalAgent


async def main():
    # ----- setup -----
    env = ToyPhysicsEnv(gravity=-9.8)

    encoder = IdentityEncoder()

    world_model = WorldModel(
        z_dim=2,
        action_dim=1
    )

    causal_model = CausalUnderstanding(
        num_causes=2,
        cause_dim=1,
        action_dim=1
    )

    agent = CausalAgent(
        encoder=encoder,
        world_model=world_model,
        causal_model=causal_model
    )

    # ----- run -----
    obs = env.reset()

    print("\n--- Running with normal gravity ---")
    for t in range(10):
        action = torch.tensor([0.0])  # no force
        result = await agent.step(obs, action)

        obs, _, done = env.step(action)

        print(f"t={t}")
        print(" influence:", result["influence"])
        print(" active causes:", result["active_causes"])
        print()

        if done:
            break

    # ----- change physics (THIS IS THE TEST) -----
    env.gravity = +9.8
    obs = env.reset()

    print("\n--- Gravity flipped ---")
    for t in range(10):
        action = torch.tensor([0.0])
        result = await agent.step(obs, action)

        obs, _, done = env.step(action)

        print(f"t={t}")
        print(" influence:", result["influence"])
        print(" active causes:", result["active_causes"])
        print()

        if done:
            break


asyncio.run(main())
