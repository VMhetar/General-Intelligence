import torch
from typing import List, Optional


class TransferProtocol:
    def __init__(self, world_model, causal_model):
        self.world_model = world_model
        self.causal_model = causal_model

    def freeze_world_model(self):
        for p in self.world_model.parameters():
            p.requires_grad = False

    def freeze_causal_model(self):
        for p in self.causal_model.parameters():
            p.requires_grad = False

    def freeze_all(self):
        self.freeze_world_model()
        self.freeze_causal_model()

    def diagnose_broken_causes(
        self,
        influence_scores: List[float],
        threshold: float = 1e-2
    ) -> List[int]:
        return [
            i for i, score in enumerate(influence_scores)
            if score > threshold
        ]

    def dominant_cause(self, influence_scores: List[float]) -> Optional[int]:
        if not influence_scores:
            return None
        return int(torch.tensor(influence_scores).argmax().item())

    def unfreeze_single_cause(self, cause_idx: int):
        if not (0 <= cause_idx < len(self.causal_model.causes)):
            raise IndexError(
                f"cause_idx {cause_idx} out of range [0, {len(self.causal_model.causes)})"
            )

        for i, net in enumerate(self.causal_model.causes):
            for p in net.parameters():
                p.requires_grad = (i == cause_idx)

    def unfreeze_multiple_causes(self, cause_indices: List[int]):
        num_causes = len(self.causal_model.causes)
        
        for idx in cause_indices:
            if not (0 <= idx < num_causes):
                raise IndexError(f"cause_idx {idx} out of range [0, {num_causes})")

        for i, net in enumerate(self.causal_model.causes):
            for p in net.parameters():
                p.requires_grad = (i in cause_indices)

    def make_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        trainable_params = [
            p for p in self.causal_model.parameters() if p.requires_grad
        ]

        if not trainable_params:
            trainable_params = [
                p for p in self.world_model.parameters() if p.requires_grad
            ]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found")

        return torch.optim.Adam(trainable_params, lr=lr)

    def report_trainable(self):
        print("\n[TransferProtocol] Trainable parameters:")
        
        has_trainable = False
        for name, p in self.causal_model.named_parameters():
            if p.requires_grad:
                print(f"  ✓ {name}")
                has_trainable = True
        
        for name, p in self.world_model.named_parameters():
            if p.requires_grad:
                print(f"  ✓ {name}")
                has_trainable = True
        
        if not has_trainable:
            print("  (none)")
