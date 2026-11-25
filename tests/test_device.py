import pytest
import torch
import numpy as np
from PIL import Image

from models.device import make_generator, resolve_device
from models.mask_adapter import load_binary_mask
from pgs.latent_correction import normalized_masked_correction
from pgs.soft_argmax import soft_argmax_2d
from pgs.sips import calculate_sips


def test_automatic_device_is_available_backend():
    device = resolve_device()
    assert device.type in {"mps", "cuda", "cpu"}
    if device.type == "mps":
        assert torch.backends.mps.is_available()
    if device.type == "cuda":
        assert torch.cuda.is_available()


def test_explicit_cpu_and_generator():
    assert resolve_device("cpu").type == "cpu"
    generator = make_generator(7, "cpu")
    assert generator.device.type == "cpu"
    assert torch.equal(torch.rand(3, generator=generator),
                       torch.rand(3, generator=make_generator(7, "cpu")))


def test_unavailable_accelerator_fails_clearly():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        with pytest.raises(RuntimeError, match="MPS was requested"):
            resolve_device("mps")


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="MPS runtime is unavailable in this execution context")
def test_mps_mask_loading_and_pgs_tensor_gradient(tmp_path):
    device = resolve_device("mps")
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(mask_path)
    loaded = load_binary_mask(mask_path, expected_hw=(2, 2), device=device)
    assert loaded.device.type == "mps"
    assert set(torch.unique(loaded).cpu().tolist()) == {0.0, 1.0}
    generator = make_generator(11, device)
    random_values = torch.randn(2, 3, device=device, generator=generator)
    assert random_values.device.type == "mps"
    heatmaps = torch.randn(1, 4, 3, 3, device=device, requires_grad=True)
    points, confidence = soft_argmax_2d(heatmaps, beta=0.5)
    values, valid = calculate_sips(points, confidence, {"ratio": (0, 1, 2, 3)}, min_confidence=0)
    latent = torch.ones(1, 2, 4, 3, device=device, requires_grad=True)
    editable = torch.ones(1, 1, 2, 3, device=device)
    corrected, grad = normalized_masked_correction(
        latent, latent.square().sum(), editable, strength=0.01, person_height=2)
    (points.sum() + confidence.sum()).backward()
    assert heatmaps.grad is not None and torch.isfinite(heatmaps.grad).all()
    assert valid["ratio"].device.type == "mps"
    assert torch.count_nonzero(grad[:, :, 2:]) == 0
    assert torch.equal(corrected[:, :, 2:], latent.detach()[:, :, 2:])
