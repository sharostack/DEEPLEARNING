"""Synthetic serialization checks; these tests never import or run CatVTON."""
import importlib.util
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from models.densepose_utils import first_densepose_detection


EXPORTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "catvtoncode"
    / "Comfyui-CatVTON"
    / "tools"
    / "automasker_batch.py"
)
_spec = importlib.util.spec_from_file_location("automasker_batch_under_test", EXPORTER_PATH)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def palette_image(indices: np.ndarray) -> Image.Image:
    image = Image.fromarray(indices.astype(np.uint8), mode="P")
    palette = [0] * 768
    # Deliberately make class IDs map to colors whose luminance differs from the ID.
    palette[3 * 5 : 3 * 5 + 3] = [255, 255, 255]
    palette[3 * 14 : 3 * 14 + 3] = [0, 0, 0]
    image.putpalette(palette)
    return image


def test_schp_palette_modes_preserve_stored_class_indices():
    expected = np.array([[0, 5, 14], [19, 2, 8]], dtype=np.uint8)
    for key, maximum in (("schp_lip", 19), ("schp_atr", 17)):
        labels = expected.copy()
        if maximum == 17:
            labels[labels > 17] = 17
        encoded = exporter.serialize_output(palette_image(labels), key, (3, 2))
        assert encoded.mode == "L"
        assert np.array_equal(np.asarray(encoded), labels)


def test_m0_is_binary_white_editable_black_protected():
    source = Image.fromarray(np.array([[0, 1, 255, 0]], dtype=np.uint8), mode="L")
    encoded = exporter.serialize_output(source, "mask", (4, 1))
    assert np.array_equal(np.asarray(encoded), np.array([[0, 255, 255, 0]], dtype=np.uint8))


def test_m0_rejects_ambiguous_grayscale_values():
    source = Image.fromarray(np.array([[0, 127]], dtype=np.uint8), mode="L")
    with pytest.raises(ValueError, match="expected binary values"):
        exporter.serialize_output(source, "mask", (2, 1))


def test_densepose_empty_detection_is_a_failure_not_a_zero_map():
    with pytest.raises(RuntimeError, match="produced no person detections"):
        first_densepose_detection([])


def test_densepose_preserves_raw_part_ids_through_24():
    labels = np.array([[0, 1, 15], [22, 23, 24]], dtype=np.uint8)
    encoded = exporter.serialize_output(Image.fromarray(labels, mode="L"), "densepose", (3, 2))
    assert np.array_equal(np.asarray(encoded), labels)


def test_all_outputs_keep_dimensions_relative_path_and_image_id(tmp_path):
    h, w = 2, 3
    result = {
        "mask": Image.fromarray(np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8), mode="L"),
        "schp_lip": palette_image(np.array([[0, 5, 19], [14, 2, 8]], dtype=np.uint8)),
        "schp_atr": palette_image(np.array([[0, 4, 17], [14, 2, 8]], dtype=np.uint8)),
        "densepose": Image.fromarray(np.array([[0, 24, 3], [4, 12, 20]], dtype=np.uint8), mode="L"),
    }
    relative = Path("subdir") / "00042_00.jpg"
    exporter.save_result_outputs(result, (w, h), tmp_path, relative)
    for folder, key in exporter.OUTPUTS:
        path = tmp_path / folder / "subdir" / "00042_00.png"
        assert path.is_file()
        with Image.open(path) as saved:
            assert saved.size == (w, h)
            assert np.array_equal(np.asarray(saved), np.asarray(exporter.serialize_output(result[key], key, (w, h))))
    valid, reason = exporter.outputs_are_valid(tmp_path, relative, (w, h))
    assert valid and reason == "all four outputs validated"


def test_resume_validation_rejects_partial_or_invalid_output_set(tmp_path):
    relative = Path("00006_00.jpg")
    size = (2, 1)
    mask = exporter.serialize_output(Image.fromarray(np.array([[0, 255]], dtype=np.uint8)), "mask", size)
    lip = exporter.serialize_output(palette_image(np.array([[0, 19]], dtype=np.uint8)), "schp_lip", size)
    atr = exporter.serialize_output(palette_image(np.array([[0, 17]], dtype=np.uint8)), "schp_atr", size)
    dense = exporter.serialize_output(Image.fromarray(np.array([[0, 24]], dtype=np.uint8)), "densepose", size)
    result = {"mask": mask, "schp_lip": lip, "schp_atr": atr, "densepose": dense}
    exporter.save_result_outputs(result, size, tmp_path, relative)
    assert exporter.outputs_are_valid(tmp_path, relative, size)[0]

    exporter.output_paths(tmp_path, relative)["densepose"].unlink()
    valid, error = exporter.outputs_are_valid(tmp_path, relative, size)
    assert not valid and "missing output file" in error


def test_dimension_mismatch_fails_clearly():
    wrong_size = Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L")
    with pytest.raises(ValueError, match="does not match input"):
        exporter.serialize_output(wrong_size, "densepose", (768, 1024))


def test_densepose_placeholder_and_out_of_range_labels_fail():
    placeholder = Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L")
    with pytest.raises(ValueError, match="does not match input"):
        exporter.serialize_output(placeholder, "densepose", (3, 2))
    invalid = Image.fromarray(np.array([[0, 25]], dtype=np.uint8), mode="L")
    with pytest.raises(ValueError, match=r"part IDs in \[0,24\]"):
        exporter.serialize_output(invalid, "densepose", (2, 1))


def test_missing_output_fails_instead_of_silently_skipping(tmp_path):
    one_pixel = Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L")
    result = {
        "mask": one_pixel,
        "schp_lip": one_pixel,
        "schp_atr": one_pixel,
    }
    with pytest.raises(ValueError, match="missing required output 'densepose'"):
        exporter.save_result_outputs(result, (1, 1), tmp_path, Path("sample.png"))


def test_person_id_and_relative_path_are_preserved_for_all_outputs(tmp_path):
    paths = exporter.output_paths(tmp_path, Path("category/00006_00.jpg"))
    assert paths["mask"] == tmp_path / "automasker_M0/category/00006_00.png"
    assert paths["schp_lip"] == tmp_path / "schp/lip/category/00006_00.png"
    assert paths["schp_atr"] == tmp_path / "schp/atr/category/00006_00.png"
    assert paths["densepose"] == tmp_path / "densepose/category/00006_00.png"


def test_resume_preserves_valid_outputs_including_existing_densepose(tmp_path):
    relative = Path("00006_00.jpg")
    size = (2, 1)
    first = {
        "mask": Image.fromarray(np.array([[0, 255]], dtype=np.uint8)),
        "schp_lip": palette_image(np.array([[0, 5]], dtype=np.uint8)),
        "schp_atr": palette_image(np.array([[0, 4]], dtype=np.uint8)),
        "densepose": Image.fromarray(np.array([[0, 24]], dtype=np.uint8)),
    }
    exporter.save_result_outputs(first, size, tmp_path, relative)
    before = {key: exporter.output_paths(tmp_path, relative)[key].read_bytes()
              for key in ("mask", "schp_lip", "schp_atr", "densepose")}
    second = dict(first)
    second["densepose"] = Image.fromarray(np.array([[0, 23]], dtype=np.uint8))
    exporter.save_result_outputs(second, size, tmp_path, relative, preserve_valid=True)
    after = {key: exporter.output_paths(tmp_path, relative)[key].read_bytes()
             for key in before}
    assert after == before


def test_corrupt_intermediate_is_reported_as_invalid(tmp_path):
    relative = Path("00006_00.png")
    size = (2, 1)
    outputs = {
        "mask": Image.fromarray(np.array([[0, 255]], dtype=np.uint8)),
        "schp_lip": palette_image(np.array([[0, 19]], dtype=np.uint8)),
        "schp_atr": palette_image(np.array([[0, 17]], dtype=np.uint8)),
        "densepose": Image.fromarray(np.array([[0, 24]], dtype=np.uint8)),
    }
    exporter.save_result_outputs(outputs, size, tmp_path, relative, preserve_valid=False)
    paths = exporter.output_paths(tmp_path, relative)
    paths["densepose"].write_bytes(b"not a png")
    valid, error = exporter.outputs_are_valid(tmp_path, relative, size)
    assert not valid
    assert "cannot identify image file" in error or "broken data stream" in error


def test_densepose_subprocess_failure_is_raised_clearly(tmp_path, monkeypatch):
    python = tmp_path / "python"
    worker = tmp_path / "worker.py"
    python.touch()
    worker.touch()

    class FailedProcess:
        stdout = io.StringIO("DENSEPOSE_MODEL_READY\ncheckpoint error\n")

        def wait(self):
            return 2

    monkeypatch.setattr(exporter.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(RuntimeError, match="DensePose subprocess failed.*checkpoint error"):
        exporter.run_densepose_worker(python, worker, tmp_path / "manifest.json",
                                      tmp_path / "models", tmp_path)


def test_original_automasker_logic_keeps_white_editable_and_black_protected():
    AutoMasker = exporter.load_automasker(EXPORTER_PATH.parents[1])
    background = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    protected = AutoMasker.cloth_agnostic_mask(background, background, background, part="upper")
    assert set(np.unique(np.asarray(protected))) == {0}

    lip_upper = Image.fromarray(np.full((64, 64), 5, dtype=np.uint8))
    atr_upper = Image.fromarray(np.full((64, 64), 4, dtype=np.uint8))
    editable = AutoMasker.cloth_agnostic_mask(background, lip_upper, atr_upper, part="upper")
    assert set(np.unique(np.asarray(editable))) == {255}

