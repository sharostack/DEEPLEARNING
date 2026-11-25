"""PropVTON mask-selection entry point for CatVTON's existing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch

from models.mask_adapter import load_selected_mask, run_catvton_with_mask


def run_pair(pipe: Any, person_image: Image.Image, garment_image: Image.Image,
             derived_root: str | Path, split: str, person_id: str, garment_id: str,
             mask_source: str, expected_hw: tuple[int, int] = (1024, 768),
             device: str | torch.device = "cpu",
             automasker_directory: str = "automasker_M0",
             bamnet_directory: str = "bamnet_M", **pipeline_kwargs: Any):
    """Run one person/garment pair and return images plus mask provenance."""
    mask, mask_path, selected_source = load_selected_mask(
        derived_root=derived_root,
        split=split,
        person_id=person_id,
        garment_id=garment_id,
        mask_source=mask_source,
        expected_hw=expected_hw,
        device=device,
        automasker_directory=automasker_directory,
        bamnet_directory=bamnet_directory,
    )
    images = run_catvton_with_mask(pipe, person_image, garment_image, mask, **pipeline_kwargs)
    return images, {"mask_source": selected_source, "mask_path": str(mask_path)}
