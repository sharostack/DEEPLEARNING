"""PGS correction schedule and validated experimental settings."""
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class PGSConfig:
    enabled: bool = False
    beta: float = 0.1
    forward_iteration_indices: Sequence[int] = (10, 15, 20, 25, 30, 35)
    correction_strength: float = 0.01
    strength_by_step: Mapping[int, float] = field(default_factory=dict)
    min_confidence: float = 0.2
    epsilon: float = 1e-8
    correction_mask: str = "editable_person"

    def __post_init__(self):
        if self.beta <= 0 or self.epsilon <= 0:
            raise ValueError("beta and epsilon must be positive")
        if self.correction_strength < 0:
            raise ValueError("correction_strength must be nonnegative")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0,1]")
        if any(i < 0 for i in self.forward_iteration_indices):
            raise ValueError("iteration indices must be nonnegative")
        if any(int(i) < 0 or float(s) < 0 for i, s in self.strength_by_step.items()):
            raise ValueError("step indices and strengths must be nonnegative")
        if self.correction_mask not in {"editable_person", "whole_person"}:
            raise ValueError("correction_mask must be editable_person or whole_person")

    def should_correct(self, forward_index: int, has_valid_ratios: bool = True) -> bool:
        return self.enabled and forward_index in self.forward_iteration_indices and has_valid_ratios

    def strength_at(self, forward_index: int) -> float:
        return float(self.strength_by_step.get(forward_index, self.correction_strength))
