"""Differentiable spatial soft-argmax for pose heatmaps."""
from __future__ import annotations

import torch


def soft_argmax_2d(heatmaps: torch.Tensor, beta: float = 0.1,
                   normalize_coordinates: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable (x,y) coordinates and peak-probability confidence.

    Heatmaps are spatially normalized with ``softmax(H / beta)``. Coordinates
    default to [0,1] over the heatmap extent; set ``normalize_coordinates=False``
    for pixel coordinates. Any joint with a non-finite value is marked invalid
    with zero coordinates and zero confidence.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"expected [B,J,H,W], got {tuple(heatmaps.shape)}")
    if beta <= 0:
        raise ValueError("beta must be positive")
    b, j, h, w = heatmaps.shape
    if h < 1 or w < 1:
        raise ValueError("heatmap height and width must be nonzero")
    flat = heatmaps.reshape(b, j, h * w)
    valid = torch.isfinite(flat).all(dim=-1)
    safe = torch.nan_to_num(flat.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.softmax(safe / beta, dim=-1)
    x_scale = max(w - 1, 1) if normalize_coordinates else 1
    y_scale = max(h - 1, 1) if normalize_coordinates else 1
    ys, xs = torch.meshgrid(
        torch.arange(h, device=heatmaps.device, dtype=probs.dtype) / y_scale,
        torch.arange(w, device=heatmaps.device, dtype=probs.dtype) / x_scale, indexing="ij")
    coords = torch.stack(((probs * xs.reshape(1, 1, -1)).sum(-1),
                          (probs * ys.reshape(1, 1, -1)).sum(-1)), dim=-1)
    confidence = probs.amax(dim=-1)
    coords = torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    return coords, confidence
