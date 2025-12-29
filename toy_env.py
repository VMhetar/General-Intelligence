import torch

class ToyPhysicsEnv:
    def __init__(self, gravity=-9.8):
        self.gravity = gravity
        self.reset()

    def reset(self):
        self.state = torch.tensor([0.0, 0.0])  # [position, velocity]
        return self.state.clone()

    def step(self, action):
        # action = force
        pos, vel = self.state
        force = action.item()

        vel = vel + force + self.gravity * 0.1
        pos = pos + vel * 0.1

        self.state = torch.tensor([pos, vel])

        reward = -abs(pos)          # stay near 0
        done = abs(pos) > 10.0

        return self.state.clone(), reward, done
