"""Person-only spatial gradient correction for CatVTON's concatenated latent."""
from __future__ import annotations

import torch


def normalized_masked_correction(latent: torch.Tensor, loss: torch.Tensor,
                                 editable_mask: torch.Tensor, strength: float,
                                 epsilon: float = 1e-8, person_height: int | None = None,
                                 person_first: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct only editable person positions; the garment latent is untouched.

    CatVTON concatenates ``[person_latent, garment_latent]`` along height
    (dimension -2). ``editable_mask`` is the person-region mask at latent
    resolution, shaped [H,W], [B,H,W], or [B,1,H,W].
    """
    if latent.ndim != 4:
        raise ValueError("latent must be [B,C,H,W]")
    if not latent.requires_grad:
        raise ValueError("latent must require gradients")
    if strength < 0 or epsilon <= 0:
        raise ValueError("strength must be nonnegative and epsilon must be positive")
    if person_height is None:
        if latent.shape[-2] % 2:
            raise ValueError("concatenated latent height must be even")
        person_height = latent.shape[-2] // 2
    if person_height <= 0 or latent.shape[-2] != 2 * person_height:
        raise ValueError("latent height must contain equal person and garment regions")

    grad, = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False,
                                allow_unused=False)
    if grad.shape != latent.shape:
        raise ValueError("loss gradient shape differs from latent")
    if editable_mask.ndim == 2:
        editable_mask = editable_mask[None, None]
    elif editable_mask.ndim == 3:
        editable_mask = editable_mask.unsqueeze(1)
    if editable_mask.ndim != 4 or editable_mask.shape[1] != 1:
        raise ValueError("editable_mask must be [H,W], [B,H,W], or [B,1,H,W]")
    if editable_mask.shape[0] not in (1, latent.shape[0]) or editable_mask.shape[-2:] != (person_height, latent.shape[-1]):
        raise ValueError("editable mask must match the person latent region and batch")
    mask = editable_mask.to(device=latent.device, dtype=latent.dtype)
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("editable mask must contain finite values in [0,1]")

    zeros = torch.zeros_like(mask)
    full_mask = torch.cat((mask, zeros), dim=-2) if person_first else torch.cat((zeros, mask), dim=-2)
    grad = grad * full_mask
    editable_count = mask.sum().clamp_min(1)
    rms = torch.sqrt(grad.square().sum() / editable_count + epsilon)
    corrected = (latent - strength * grad / rms).detach()
    return corrected, grad.detach()
