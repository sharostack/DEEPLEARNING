"""Confidence-weighted objective over caller-defined SIPS ratios."""
from __future__ import annotations

import torch


def confidence_weighted_proportion_loss(predicted: dict[str, torch.Tensor],
                                        target: dict[str, torch.Tensor],
                                        valid: dict[str, torch.Tensor],
                                        confidence: dict[str, torch.Tensor] | None = None
                                        ) -> tuple[torch.Tensor | None, int]:
    terms, weights = [], []
    for name, pred in predicted.items():
        if name not in target or name not in valid:
            continue
        target_value = target[name].to(pred)
        if pred.shape != target_value.shape or valid[name].shape != pred.shape:
            raise ValueError(f"{name}: predicted, target, and valid ratio shapes must match")
        mask = (valid[name].to(device=pred.device, dtype=torch.bool)
                & torch.isfinite(pred) & torch.isfinite(target_value))
        if confidence and name in confidence:
            confidence_value = confidence[name].to(pred)
            if confidence_value.shape != pred.shape:
                raise ValueError(f"{name}: confidence shape must match ratio shape")
            mask = mask & torch.isfinite(confidence_value) & (confidence_value > 0)
        if mask.any():
            weight = (confidence_value if confidence and name in confidence else torch.ones_like(pred))
            weight = torch.where(mask, weight.clamp_min(0), torch.zeros_like(weight))
            error = torch.where(mask, pred - target_value, torch.zeros_like(pred))
            terms.append(error.square() * weight)
            weights.append(weight)
    if not terms:
        return None, 0
    numerator = torch.stack([x.sum() for x in terms]).sum()
    denominator = torch.stack([x.sum() for x in weights]).sum().clamp_min(torch.finfo(numerator.dtype).eps)
    count = sum(int((w > 0).sum().item()) for w in weights)
    return numerator / denominator, count
