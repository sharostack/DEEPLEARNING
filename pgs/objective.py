"""Differentiable generated-vs-original SIPS objective."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch

from .pgs_controller import PGSConfig
from .proportion_loss import confidence_weighted_proportion_loss
from .sips import calculate_sips
from .soft_argmax import soft_argmax_2d


class PGSObjective:
    """Compare configured ratios from a generated person to original-person ratios.

    ``definitions`` maps names to ``(a,b,c,d)`` joint indices and encodes
    distance(a,b)/distance(c,d). Final anatomical definitions must be supplied
    by Person 3; this class does not prescribe them.
    """
    def __init__(self, pose_model, definitions: Mapping[str, Sequence[int]], config: PGSConfig):
        if not definitions:
            raise ValueError("SIPS definitions are required; obtain the agreed ratios from Person 3")
        self.pose_model = pose_model
        self.definitions = dict(definitions)
        self.config = config
        self.reference_ratios: dict[str, torch.Tensor] | None = None
        self.reference_valid: dict[str, torch.Tensor] | None = None

    def set_reference(self, original_rgb: torch.Tensor) -> None:
        """Cache original person's ratios; reference inference is intentionally no-grad."""
        with torch.no_grad():
            heatmaps = self.pose_model(original_rgb)
            keypoints, confidence = soft_argmax_2d(heatmaps, self.config.beta)
            ratios, valid = calculate_sips(keypoints, confidence, self.definitions,
                                           self.config.min_confidence, self.config.epsilon)
        self.reference_ratios = {key: value.detach() for key, value in ratios.items()}
        self.reference_valid = {key: value.detach() for key, value in valid.items()}

    def __call__(self, generated_rgb: torch.Tensor) -> tuple[torch.Tensor | None, int]:
        if self.reference_ratios is None or self.reference_valid is None:
            raise RuntimeError("set_reference(original_person_rgb) must be called before PGS loss")
        heatmaps = self.pose_model(generated_rgb)
        keypoints, confidence = soft_argmax_2d(heatmaps, self.config.beta)
        ratios, valid = calculate_sips(keypoints, confidence, self.definitions,
                                       self.config.min_confidence, self.config.epsilon)
        combined_valid = {key: valid[key] & self.reference_valid[key].to(valid[key].device)
                          for key in ratios}
        ratio_confidence = {}
        for name, (a, b, c, d) in self.definitions.items():
            ratio_confidence[name] = confidence[..., [a, b, c, d]].amin(dim=-1)
        return confidence_weighted_proportion_loss(
            ratios, self.reference_ratios, combined_valid, ratio_confidence)
