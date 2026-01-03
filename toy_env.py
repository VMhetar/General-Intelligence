import torch
from typing import Tuple


class ToyPhysicsEnv:
    def __init__(self, gravity: float = -9.8):
        self.gravity = gravity
        self.state = torch.zeros(3)
        self.reset()

    def reset(self) -> torch.Tensor:
        self.state = torch.tensor([0.0, 0.0, self.gravity], dtype=torch.float32)
        return self.state.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        pos, vel, g = self.state

        force = action.item() if action.numel() == 1 else action[0].item()

        vel = vel + force + g * 0.1
        pos = pos + vel * 0.1

        self.state = torch.tensor([pos, vel, g], dtype=torch.float32)

        reward = -abs(pos.item())
        done = abs(pos.item()) > 10.0

        return self.state.clone(), reward, done
