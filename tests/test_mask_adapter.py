"""Synthetic pair-mask loading/selection tests; no CatVTON model is loaded."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
import yaml

from inference.catvton_masked import run_pair
from models.mask_adapter import (
    TryOnInputs,
    load_binary_mask,
    load_selected_mask,
    resolve_mask_path,
    select_person_mask,
)


HW = (4, 3)  # height, width


def write_mask(path: Path, pixels: np.ndarray, mode: str = "L") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode=mode).save(path, format="PNG")
    return path


def inputs(mstar: torch.Tensor | None = None) -> TryOnInputs:
    rgb = torch.zeros(1, 3, *HW, dtype=torch.float32)
    mask = torch.zeros(1, 1, *HW, dtype=torch.float32)
    return TryOnInputs(rgb, rgb, rgb, mask, mask, mstar)


def test_pair_specific_filename_resolution():
    path = resolve_mask_path("data/derived", "train", "bamnet", "00006_00", "00001_00")
    assert path == Path("data/derived/train/bamnet_M/00006_00__00001_00.png")


def test_binary_mask_loads_as_float32_bchw_and_preserves_polarity(tmp_path):
    values = np.array([[0, 255, 0], [255, 0, 255], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    path = write_mask(tmp_path / "pair.png", values)
    tensor = load_binary_mask(path, expected_hw=HW, device="cpu")
    assert tensor.shape == (1, 1, *HW)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert torch.equal(tensor[0, 0], torch.from_numpy(values.astype(np.float32) / 255.0))
    assert tensor[0, 0, 0, 0].item() == 0.0  # protected
    assert tensor[0, 0, 0, 1].item() == 1.0  # editable


def test_loader_accepts_cuda_device_when_available(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment")
    path = write_mask(tmp_path / "cuda.png", np.zeros(HW, dtype=np.uint8))
    assert load_binary_mask(path, HW, "cuda").device.type == "cuda"


def test_missing_bamnet_pair_mask_does_not_fall_back_to_m0(tmp_path):
    write_mask(tmp_path / "train/automasker_M0/00006_00.png", np.zeros(HW, dtype=np.uint8))
    with pytest.raises(FileNotFoundError, match="BAM-Net mask required for pair"):
        load_selected_mask(tmp_path, "train", "00006_00", "00001_00", "bamnet", HW)


def test_missing_automasker_mask_reports_exact_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="00006_00.png"):
        load_selected_mask(tmp_path, "train", "00006_00", "00001_00", "automasker", HW)


def test_nonbinary_gray_values_are_rejected(tmp_path):
    path = write_mask(tmp_path / "gray.png", np.full(HW, 127, dtype=np.uint8))
    with pytest.raises(ValueError, match=r"only 0 \(protected\) and 255 \(editable\)"):
        load_binary_mask(path, HW)


def test_wrong_dimensions_are_rejected(tmp_path):
    path = write_mask(tmp_path / "wrong.png", np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions are 3x4 .* expected 4x3"):
        load_binary_mask(path, HW)


def test_auto_mode_uses_m0_when_pair_mask_is_absent(tmp_path):
    m0 = np.zeros(HW, dtype=np.uint8)
    m0[0, 0] = 255
    write_mask(tmp_path / "train/automasker_M0/00006_00.png", m0)
    tensor, path, source = load_selected_mask(tmp_path, "train", "00006_00", "00001_00", "auto", HW)
    assert source == "automasker"
    assert path.name == "00006_00.png"
    assert tensor[0, 0, 0, 0].item() == 1.0


def test_auto_mode_uses_mstar_when_pair_mask_exists(tmp_path):
    m0 = np.zeros(HW, dtype=np.uint8)
    mstar = np.full(HW, 255, dtype=np.uint8)
    write_mask(tmp_path / "test/automasker_M0/00006_00.png", m0)
    write_mask(tmp_path / "test/bamnet_M/00006_00__00001_00.png", mstar)
    tensor, path, source = load_selected_mask(tmp_path, "test", "00006_00", "00001_00", "auto", HW)
    assert source == "bamnet"
    assert path.name == "00006_00__00001_00.png"
    assert torch.all(tensor == 1)


def test_tensor_selector_explicit_bamnet_errors_and_auto_falls_back():
    missing = inputs()
    with pytest.raises(FileNotFoundError, match="requires a BAM-Net refined mask"):
        select_person_mask(missing, "bamnet")
    assert select_person_mask(missing, "auto") is missing.automasker_mask


def test_tensor_selector_auto_prefers_mstar():
    original = inputs()
    refined = torch.ones_like(original.automasker_mask)
    original.refined_bamnet_mask = refined
    assert select_person_mask(original, "auto") is refined
    assert select_person_mask(original, "bamnet") is refined


def test_inference_adapter_passes_selected_mask_to_existing_pipeline(tmp_path):
    pixels = np.zeros(HW, dtype=np.uint8)
    pixels[1, 2] = 255
    write_mask(tmp_path / "train/bamnet_M/00006_00__00001_00.png", pixels)

    class DummyPipe:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return ["synthetic-result"]

    pipe = DummyPipe()
    person = Image.new("RGB", (HW[1], HW[0]))
    garment = Image.new("RGB", (HW[1], HW[0]))
    result, metadata = run_pair(
        pipe, person, garment, tmp_path, "train", "00006_00", "00001_00", "bamnet",
        expected_hw=HW, device="cpu", num_inference_steps=1,
    )
    assert result == ["synthetic-result"]
    assert metadata["mask_source"] == "bamnet"
    assert metadata["mask_path"].endswith("00006_00__00001_00.png")
    assert pipe.kwargs["mask"].mode == "L"
    assert np.array_equal(np.asarray(pipe.kwargs["mask"]), pixels)
    assert pipe.kwargs["num_inference_steps"] == 1


def test_experiment_configs_select_a_b_and_c_mask_sources():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    expected = {"a": "automasker", "b": "bamnet", "c": "bamnet"}
    for experiment, source in expected.items():
        with (config_dir / f"experiment_{experiment}.yaml").open() as stream:
            config = yaml.safe_load(stream)
        assert config["mask_source"] == source
