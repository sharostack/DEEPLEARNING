"""Pair-aware loading and selection of CatVTON editable-region masks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from models.device import resolve_device


@dataclass
class TryOnInputs:
    """Optional tensor contract used by model-side callers and tests."""
    person_rgb: torch.Tensor
    skeleton_rgb: torch.Tensor
    garment_rgb: torch.Tensor
    garment_mask: torch.Tensor
    automasker_mask: torch.Tensor
    refined_bamnet_mask: torch.Tensor | None = None


def resolve_mask_path(derived_root: str | Path, split: str, source: str,
                      person_id: str, garment_id: str,
                      automasker_directory: str = "automasker_M0",
                      bamnet_directory: str = "bamnet_M") -> Path:
    """Resolve M0 by person ID or M* by the person/garment pair."""
    if split not in {"train", "test", "val"}:
        raise ValueError(f"unsupported split {split!r}; expected train, test, or val")
    for label, value in (("person_id", person_id), ("garment_id", garment_id)):
        if not value or Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be a non-empty filename stem, got {value!r}")
    root = Path(derived_root) / split
    if source == "automasker":
        return root / automasker_directory / f"{person_id}.png"
    if source == "bamnet":
        return root / bamnet_directory / f"{person_id}__{garment_id}.png"
    raise ValueError("source must be 'automasker' or 'bamnet'")


def load_binary_mask(path: str | Path, expected_hw: tuple[int, int] = (1024, 768),
                     device: str | torch.device = "cpu") -> torch.Tensor:
    """Load a binary PNG as float32 [1,1,H,W], where 1 means editable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mask file does not exist: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"mask must be a PNG file: {path} (format={image.format})")
            if image.mode not in {"1", "L"}:
                raise ValueError(f"mask must use binary or grayscale mode, got {image.mode!r}: {path}")
            pixels = np.asarray(image)
            size = image.size
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"could not read mask PNG {path}: {exc}") from exc

    height, width = expected_hw
    if size != (width, height):
        raise ValueError(f"mask dimensions are {size[1]}x{size[0]} (H,W), expected {height}x{width}: {path}")
    values = set(np.unique(pixels).tolist())
    # Mode 1 represents the same black/white convention as integer 0/255.
    if image.mode == "1":
        pixels = pixels.astype(np.uint8) * 255
        values = set(np.unique(pixels).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"mask must contain only 0 (protected) and 255 (editable), got {sorted(values)}: {path}")
    tensor = torch.from_numpy(np.array(pixels, dtype=np.float32, copy=True) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    device = resolve_device(device)
    try:
        return tensor.to(device=device, dtype=torch.float32)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not move mask tensor to {device}: {exc}") from exc


def validate_tryon_inputs(inputs: TryOnInputs) -> None:
    """Check image/mask shapes, finite values and [0,1] ranges."""
    for name in ("person_rgb", "skeleton_rgb", "garment_rgb"):
        image = getattr(inputs, name)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"{name} must be [B,3,H,W]")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"{name} must contain finite RGB values in [0,1]")
    batch = inputs.person_rgb.shape[0]
    for name in ("skeleton_rgb", "garment_rgb"):
        image = getattr(inputs, name)
        if image.shape[:2] != (batch, 3):
            raise ValueError(f"{name} batch size must match person_rgb")
        if image.shape[-2:] != inputs.person_rgb.shape[-2:]:
            raise ValueError(f"{name} spatial size must match person_rgb")
    for name in ("garment_mask", "automasker_mask", "refined_bamnet_mask"):
        mask = getattr(inputs, name)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[:2] != (batch, 1):
            raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W]")
        if mask.shape[-2:] != inputs.person_rgb.shape[-2:]:
            raise ValueError(f"{name} spatial size must match person_rgb")
        if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError(f"{name} must contain finite values in [0,1]")


def select_person_mask(inputs: TryOnInputs, mode: str) -> torch.Tensor:
    """Select M0, M*, or M* when supplied and otherwise M0."""
    validate_tryon_inputs(inputs)
    if mode == "automasker":
        return inputs.automasker_mask
    if mode == "bamnet":
        if inputs.refined_bamnet_mask is None:
            raise FileNotFoundError("mask_source='bamnet' requires a BAM-Net refined mask (M*)")
        return inputs.refined_bamnet_mask
    if mode == "auto":
        return inputs.refined_bamnet_mask if inputs.refined_bamnet_mask is not None else inputs.automasker_mask
    raise ValueError("mask_source must be 'automasker', 'bamnet', or 'auto'")


def load_selected_mask(derived_root: str | Path, split: str, person_id: str,
                       garment_id: str, mask_source: str,
                       expected_hw: tuple[int, int] = (1024, 768),
                       device: str | torch.device = "cpu",
                       automasker_directory: str = "automasker_M0",
                       bamnet_directory: str = "bamnet_M") -> tuple[torch.Tensor, Path, str]:
    """Load the configured mask; `auto` falls back only when the M* file is absent."""
    if mask_source not in {"automasker", "bamnet", "auto"}:
        raise ValueError("mask_source must be 'automasker', 'bamnet', or 'auto'")
    if mask_source == "automasker":
        path = resolve_mask_path(derived_root, split, "automasker", person_id, garment_id,
                                 automasker_directory, bamnet_directory)
        return load_binary_mask(path, expected_hw, device), path, "automasker"

    bamnet_path = resolve_mask_path(derived_root, split, "bamnet", person_id, garment_id,
                                    automasker_directory, bamnet_directory)
    if bamnet_path.is_file():
        return load_binary_mask(bamnet_path, expected_hw, device), bamnet_path, "bamnet"
    if mask_source == "bamnet":
        raise FileNotFoundError(f"BAM-Net mask required for pair {person_id}__{garment_id}: {bamnet_path}")

    automasker_path = resolve_mask_path(derived_root, split, "automasker", person_id, garment_id,
                                        automasker_directory, bamnet_directory)
    return load_binary_mask(automasker_path, expected_hw, device), automasker_path, "automasker"


def mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    """Convert validated [1,1,H,W] float mask to CatVTON's white-editable PIL mask."""
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError(f"CatVTON single-image mask must be [1,1,H,W], got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("CatVTON mask tensor must contain finite values in [0,1]")
    values = torch.unique(mask)
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("CatVTON mask tensor must be binary with 1=editable and 0=protected")
    pixels = mask[0, 0].detach().to(device="cpu", dtype=torch.uint8).numpy() * 255
    return Image.fromarray(pixels, mode="L")


def run_catvton_with_mask(pipe: Any, person_image: Image.Image, garment_image: Image.Image,
                          mask: torch.Tensor, **pipeline_kwargs: Any):
    """Pass a selected mask through CatVTON's existing pipeline mask argument."""
    if mask.shape[-2:] != (person_image.height, person_image.width):
        raise ValueError(
            f"mask dimensions {tuple(mask.shape[-2:])} (H,W) do not match person image "
            f"{(person_image.height, person_image.width)} (H,W)"
        )
    return pipe(image=person_image, condition_image=garment_image,
                mask=mask_tensor_to_pil(mask), **pipeline_kwargs)
