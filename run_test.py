import torch
import asyncio

from toy_env import ToyPhysicsEnv
from encoder import IdentityEncoder
from world_model import WorldModel
from causal_understanding import CausalUnderstanding
from agent import CausalAgent
from transfer_protocol import TransferProtocol 


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

    protocol = TransferProtocol(world_model, causal_model)

    # --------------------------------------------------
    # 1️. Train the world model (World A)
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
    # 2️. FREEZE KNOWLEDGE (CRITICAL)
    # --------------------------------------------------
    print("\n--- Freezing learned knowledge ---")
    protocol.freeze_all()
    protocol.report_trainable()

    # --------------------------------------------------
    # 3. Run with normal gravity (baseline)
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
    # 4. TRANSFER TEST: Flip gravity
    # --------------------------------------------------
    env.gravity = +9.8
    obs = env.reset()

    print("\n--- Gravity flipped (TRANSFER TEST) ---")
    broken_causes = None

    for t in range(10):
        action = torch.tensor([0.0])
        result = await agent.step(obs, action)

        obs, _, done = env.step(action)

        print(f"t={t}")
        print(" influence:", result["influence"])
        print(" active causes:", result["active_causes"])
        print()

        # Diagnose only once
        if t == 0:
            broken_causes = protocol.diagnose_broken_causes(
                result["influence"]
            )
            print(" Broken causes detected:", broken_causes)

        if done:
            break

    # --------------------------------------------------
    #  OPTIONAL: Selective repair 
    # --------------------------------------------------
    if broken_causes:
        print("\n--- Selective repair phase ---")
        protocol.unfreeze_single_cause(broken_causes[0])
        protocol.report_trainable()

        repair_optimizer = protocol.make_optimizer(lr=1e-3)

        for step in range(50):
            action = torch.randn(1) * 0.1

            z = encoder(obs)
            z_next_pred, _, _, _ = world_model(z, action)

            obs_next, _, _ = env.step(action)
            z_true = encoder(obs_next)

            loss = ((z_next_pred - z_true) ** 2).mean()

            repair_optimizer.zero_grad()
            loss.backward()
            repair_optimizer.step()

            obs = obs_next

        print(" Selective repair completed")


asyncio.run(main())
