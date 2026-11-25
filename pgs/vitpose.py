"""Frozen ViTPose adapter that returns differentiable raw heatmaps."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from models.device import resolve_device


class FrozenViTPose(nn.Module):
    """Run ViTPose's backbone/head without MMPose's NumPy-detaching decoder.

    The model weights are frozen, but ``forward`` is intentionally not wrapped
    in no-grad so gradients can flow from heatmaps to generated RGB tensors.
    Inputs are RGB float tensors in [0,1].
    """
    def __init__(self, model: nn.Module, input_size: tuple[int, int] = (192, 256),
                 mean: Sequence[float] = (0.485, 0.456, 0.406),
                 std: Sequence[float] = (0.229, 0.224, 0.225),
                 expected_image_hw: tuple[int, int] | None = (1024, 768)):
        super().__init__()
        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError("input_size must be (width,height) with positive values")
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError("mean and std must contain three channels and positive std values")
        self.model = model
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.expected_image_hw = expected_image_hw
        device = next(model.parameters(), torch.empty(0)).device
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1))
        self.model.eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, config_path: str | Path, checkpoint_path: str | Path,
                        device: str | torch.device | None = None,
                        expected_image_hw: tuple[int, int] = (1024, 768)) -> "FrozenViTPose":
        config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"ViTPose config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {checkpoint_path}")
        try:
            from mmpose.apis import init_pose_model
        except ImportError as exc:
            raise ImportError("ViTPose requires the compatible MMPose/MMCV runtime; see third_party/ViTPose") from exc
        device = resolve_device(device)
        model = init_pose_model(str(config_path), str(checkpoint_path), device=str(device))
        cfg = model.cfg
        input_size = tuple(int(v) for v in cfg.data_cfg.image_size)
        normalize = next((step for step in cfg.test_pipeline
                          if step.get("type") == "NormalizeTensor"), None)
        if normalize is None:
            raise ValueError("ViTPose config test_pipeline must define NormalizeTensor mean/std")
        return cls(model, input_size, normalize["mean"], normalize["std"], expected_image_hw)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("ViTPose input must be [B,3,H,W] RGB")
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("ViTPose RGB input must be finite and in [0,1]")
        if self.expected_image_hw is not None and tuple(rgb.shape[-2:]) != self.expected_image_hw:
            raise ValueError(f"ViTPose expected image HxW {self.expected_image_hw}, got {tuple(rgb.shape[-2:])}")
        width, height = self.input_size
        resized = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
        x = (resized - self.mean.to(resized)) / self.std.to(resized)
        features = self.model.backbone(x)
        if getattr(self.model, "with_neck", False):
            features = self.model.neck(features)
        heatmaps = self.model.keypoint_head(features)
        if not isinstance(heatmaps, torch.Tensor) or heatmaps.ndim != 4:
            raise ValueError("ViTPose keypoint head must return [B,J,H,W] raw heatmap tensor")
        return heatmaps
