"""Differentiable DDIM predicted-clean-sample reconstruction for PGS."""
from __future__ import annotations

import torch


def ddim_pred_original_sample(scheduler, model_output: torch.Tensor,
                              timestep: torch.Tensor | int,
                              sample: torch.Tensor) -> torch.Tensor:
    """Reconstruct DDIM x0 using the scheduler's configured prediction type.

    Matches DDIMScheduler.step's prediction and clipping branches (Diffusers
    0.26.3+). The input ``sample`` is the unscaled x_t, not scale_model_input.
    """
    prediction_type = scheduler.config.prediction_type
    t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    alphas = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas[t]
    beta_prod_t = 1 - alpha_prod_t
    if prediction_type == "epsilon":
        pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_original = model_output
    elif prediction_type == "v_prediction":
        pred_original = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"unsupported DDIM prediction_type {prediction_type!r}")

    if scheduler.config.thresholding:
        pred_original = scheduler._threshold_sample(pred_original)
    elif scheduler.config.clip_sample:
        clip_range = scheduler.config.clip_sample_range
        pred_original = pred_original.clamp(-clip_range, clip_range)
    return pred_original
