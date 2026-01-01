import torch
from typing import List, Optional


class TransferProtocol:
    """
    TransferProtocol enforces a strict causal transfer procedure.

    Responsibilities:
    - Freeze learned knowledge
    - Diagnose which causal factors break under distribution shift
    - Selectively unfreeze only broken causes for minimal repair

    This module does NOT learn.
    It only controls what is allowed to learn.
    """

    def __init__(self, world_model, causal_model):
        self.world_model = world_model
        self.causal_model = causal_model

    # --------------------------------------------------
    # Freezing utilities
    # --------------------------------------------------
    def freeze_world_model(self):
        """
        Prevents any adaptation in the world model.
        Used to test whether learned dynamics transfer.
        """
        for p in self.world_model.parameters():
            p.requires_grad = False

    def freeze_causal_model(self):
        """
        Prevents causal slots from adapting.
        Used during diagnosis phase.
        """
        for p in self.causal_model.parameters():
            p.requires_grad = False

    def freeze_all(self):
        """
        Full freeze: no learning anywhere.
        """
        self.freeze_world_model()
        self.freeze_causal_model()

    # --------------------------------------------------
    # Diagnosis
    # --------------------------------------------------
    def diagnose_broken_causes(
        self,
        influence_scores: List[float],
        threshold: float = 1e-2
    ) -> List[int]:
        """
        Identifies which causal slots show abnormal influence.

        Args:
            influence_scores: output of causal_model.influence_profile
            threshold: minimum influence considered significant

        Returns:
            List of indices of broken causes
        """
        broken = [
            i for i, score in enumerate(influence_scores)
            if score > threshold
        ]
        return broken

    def dominant_cause(
        self,
        influence_scores: List[float]
    ) -> Optional[int]:
        """
        Returns the index of the most influential cause.
        Useful when exactly one cause should adapt.
        """
        if not influence_scores:
            return None
        return int(torch.tensor(influence_scores).argmax().item())

    # --------------------------------------------------
    # Selective unfreezing
    # --------------------------------------------------
    def unfreeze_single_cause(self, cause_idx: int):
        """
        Allows learning ONLY in the specified causal slot.
        All other causes remain frozen.
        """
        for i, net in enumerate(self.causal_model.causes):
            for p in net.parameters():
                p.requires_grad = (i == cause_idx)

    def unfreeze_multiple_causes(self, cause_indices: List[int]):
        """
        Allows learning in a selected subset of causal slots.
        """
        for i, net in enumerate(self.causal_model.causes):
            for p in net.parameters():
                p.requires_grad = (i in cause_indices)

    # --------------------------------------------------
    # Optimizer helper
    # --------------------------------------------------
    def make_optimizer(
        self,
        lr: float = 1e-3
    ) -> torch.optim.Optimizer:
        """
        Creates an optimizer that only updates unfrozen parameters.
        """
        trainable_params = filter(
            lambda p: p.requires_grad,
            self.causal_model.parameters()
        )

        return torch.optim.Adam(trainable_params, lr=lr)

    # --------------------------------------------------
    # Debug / sanity checks
    # --------------------------------------------------
    def report_trainable(self):
        """
        Prints which parameters are currently trainable.
        Useful for debugging transfer experiments.
        """
        print("\n[TransferProtocol] Trainable parameters:")
        for name, p in self.causal_model.named_parameters():
            if p.requires_grad:
                print(f"  ✓ {name}")
        for name, p in self.world_model.named_parameters():
            if p.requires_grad:
                print(f"  ✓ {name}")
