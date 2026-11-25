# BAM-Net mask integration (Experiment B)

PropVTON loads pair-specific BAM-Net refined masks and passes the selected mask to CatVTON's existing `mask=` argument. CatVTON preprocessing and garment handling remain in `catvtoncode/Comfyui-CatVTON/model/pipeline.py`; the diffusion implementation is not modified.

## Input layout and conventions

```text
data/derived/{split}/
  automasker_M0/{person_id}.png
  bamnet_M/{person_id}__{garment_id}.png
```

Examples: `data/derived/train/bamnet_M/00006_00__00001_00.png`. The M0 path uses the person ID; the M* path uses both person and garment IDs. The split is retained as `train`, `test`, or `val` exactly as supplied.

Masks must be single-channel binary PNGs with pixel values only 0 and 255: 255/white/1 is editable and 0/black is protected. The loader rejects intermediate grayscale values, palette/RGB images, unreadable files, and dimensions different from the configured `(height, width)` (default 1024×768). It returns a float32 tensor shaped `[1,1,H,W]` with values 0/1. The CatVTON adapter converts that tensor back to a grayscale PIL mask before calling the pipeline, preserving the current PIL image preprocessing path.

## Selection behavior

- `mask_source: automasker` loads M0, preserving Experiment A.
- `mask_source: bamnet` requires the exact person/garment M* file. Missing or invalid M* raises an error; it never falls back to M0.
- `mask_source: auto` uses valid M* when that pair file exists; if the pair file is absent it loads M0. An existing but invalid M* raises an error rather than being hidden by fallback.

Selection is implemented in `models/mask_adapter.py`; `inference/catvton_masked.py:run_pair` resolves the IDs, loads the selected mask, and invokes the provided CatVTON pipeline. It returns pipeline images plus mask source/path metadata. This is a callable integration function; there is not yet a complete dataset inference CLI or pair-list driver.

## Experiment configuration

The shared `configs/experiment.yaml` sets `masks.derived_root`, M0/M* folder names, expected height/width, and `dataset.split`. Experiment configs override the mask source:

```yaml
# configs/experiment_a.yaml
mask_source: automasker

# configs/experiment_b.yaml
mask_source: bamnet

# configs/experiment_c.yaml
mask_source: bamnet
```

Example caller:

```python
from inference.catvton_masked import run_pair

images, provenance = run_pair(
    pipe=catvton_pipeline,
    person_image=person_pil,
    garment_image=garment_pil,
    derived_root="data/derived",
    split="train",
    person_id="00006_00",
    garment_id="00001_00",
    mask_source="bamnet",  # use experiment_b.yaml value
    expected_hw=(1024, 768),
    device="auto",
    num_inference_steps=50,
)
```

For experiment A, set `mask_source="automasker"`. For `auto`, the fallback behavior above applies. `run_pair` preserves the pipeline's existing person/garment preprocessing; it does not composite original pixels or alter the diffusion loop.

## Tests and current validation

Run synthetic tests with:

```bash
python -m pytest -q tests/test_mask_adapter.py tests/test_automasker_export_synthetic.py tests/test_pgs_synthetic.py
```

Tests use temporary synthetic PNGs and a dummy pipeline; they do not load model checkpoints or run GPU inference. Person 1 has not delivered real BAM-Net masks/checkpoint to this workspace, so the mask naming, binary convention, and output compatibility have not been verified against actual BAM-Net exports. Experiment B is structurally wired but awaits that validation and real CatVTON execution on a compatible runtime.
