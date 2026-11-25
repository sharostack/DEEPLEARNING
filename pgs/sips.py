"""SIPS ratio calculations; ratio definitions are supplied as joint-index pairs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch


def calculate_sips(keypoints: torch.Tensor, confidence: torch.Tensor,
                   definitions: Mapping[str, Sequence[int]], min_confidence: float = 0.2,
                   eps: float = 1e-8) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Calculate normalized distances for configured (numerator, denominator) joint pairs.

    This generic primitive does not assign anatomical meaning to any joint.
    Coordinates are [..., J, 2], confidence [..., J].
    """
    if keypoints.ndim < 2 or keypoints.shape[-1] != 2 or confidence.shape != keypoints.shape[:-1]:
        raise ValueError("expected keypoints [...,J,2] and matching confidence [...,J]")
    if not 0 <= min_confidence <= 1 or eps <= 0:
        raise ValueError("min_confidence must be in [0,1] and eps must be positive")
    values, valid = {}, {}
    for name, pair in definitions.items():
        if len(pair) != 4:
            raise ValueError(f"{name}: expected (a,b,c,d) for distance(a,b)/distance(c,d)")
        a, b, c, d = pair
        joints = keypoints.shape[-2]
        if min(pair) < 0 or max(pair) >= joints:
            raise IndexError(f"{name}: joint index outside [0,{joints})")
        numerator = torch.linalg.vector_norm(keypoints[..., a, :] - keypoints[..., b, :], dim=-1)
        denominator = torch.linalg.vector_norm(keypoints[..., c, :] - keypoints[..., d, :], dim=-1)
        joint_confidence = confidence[..., [a, b, c, d]]
        ok = ((torch.isfinite(joint_confidence) & (joint_confidence >= min_confidence)).all(dim=-1)
              & (denominator > eps) & torch.isfinite(numerator) & torch.isfinite(denominator))
        ratio = numerator / denominator.clamp_min(eps)
        ok = ok & torch.isfinite(ratio)
        values[name] = torch.where(ok, ratio, torch.zeros_like(ratio))
        valid[name] = ok
    return values, valid
