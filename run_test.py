import torch
import asyncio

from toy_env import ToyPhysicsEnv
from encoder import IdentityEncoder
from world_model import WorldModel
from causal_understanding import CausalUnderstanding
from agent import CausalAgent


async def main():
    torch.manual_seed(0)

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 1️⃣ Train the world model (CRUCIAL)
    # --------------------------------------------------
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    print("\n--- Training world model ---")
    for epoch in range(1000):
        obs = env.reset()

        for _ in range(10):
            action = torch.randn(1) * 0.1

            z = encoder(obs)
            z_next_pred, _, _, _ = world_model(z, action)

            obs_next, _, _ = env.step(action)
            z_true = encoder(obs_next)

            loss = ((z_next_pred - z_true) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = obs_next

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | loss = {loss.item():.6f}")

    # --------------------------------------------------
    # 2️⃣ Run with normal gravity
    # --------------------------------------------------
    obs = env.reset()

    print("\n--- Running with normal gravity ---")
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

    # --------------------------------------------------
    # 3️⃣ Flip gravity (THIS IS THE TEST)
    # --------------------------------------------------
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
