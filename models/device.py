"""Shared accelerator selection for PropVTON's inference and tensor utilities."""
from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve explicit devices or prefer MPS, then CUDA, then CPU.

    Explicit unavailable accelerators raise instead of silently moving work to
    another device. CUDA remains supported for deployment machines.
    """
    if device is not None and str(device).lower() != "auto":
        resolved = torch.device(device)
        if resolved.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but this PyTorch/macOS runtime has no available MPS backend")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return resolved

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_generator(seed: int, device: str | torch.device | None = None) -> torch.Generator:
    """Create a deterministic generator on the selected device."""
    return torch.Generator(device=resolve_device(device)).manual_seed(int(seed))


def clear_device_cache(device: str | torch.device | None = None) -> None:
    """Release cached accelerator memory where the active backend supports it."""
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
