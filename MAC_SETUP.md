# Native Apple Silicon development

The primary environment targets ARM64 Python 3.12, PyTorch 2.11, and torchvision 0.26. It uses FP32 by default, batch size 1, and zero DataLoader workers. The scientific/evaluation image resolution remains 1024x768. Smaller smoke-test dimensions are explicitly isolated under `development` in `configs/experiment.yaml`; they do not change experiment A/B/C settings or evaluation protocol.

## Create and install the clean environment

From the PropVTON root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
```

Alternatively, if you manage environments with Conda:

```bash
conda env create -f environment-macos.yml
conda activate propvton-macos
```

Confirm that the interpreter is native ARM64 and MPS is enabled:

```bash
python -c 'import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available()); assert platform.machine() == "arm64"'
```

For MPS operators not implemented by the installed PyTorch build, enable PyTorch's documented CPU fallback before starting Python:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Fallback may be slower and does not make CUDA-only extensions compatible. The shared device helper selects MPS first, CUDA on an NVIDIA host second, and CPU otherwise. Explicit unavailable devices raise an error. CUDA remains supported.

## Tests

Run the full project test suite:

```bash
python -m pytest -q
```

Or the targeted synthetic integration suite:

```bash
python -m pytest -q tests/test_mask_adapter.py tests/test_pgs_synthetic.py tests/test_automasker_export_synthetic.py tests/test_device.py
```

These tests do not download checkpoints, initialize Detectron2, or run diffusion inference.

## Component commands and current capability

### Mask adapter and PGS primitives

The above synthetic test command is the supported local validation. MPS tensor execution can be selected in a Python session with `from models.device import resolve_device; device = resolve_device()` and `tensor.to(device)`. A/B keep `pgs.enabled: false`; C remains enabled and requires Person 3's SIPS definitions plus a local ViTPose checkpoint before constructing its runtime.

### CatVTON pipeline

The code now resolves its device automatically and keeps the original model/checkpoint interfaces. It can be loaded from the existing ComfyUI node or `CatVTONPipeline` after supplying local model IDs/paths. Model loading may access Hugging Face and fetch model files; do not invoke it until you intend to download those weights. The checked-in project has no dedicated local-image inference CLI. A/B/C actual diffusion inference has not been verified on MPS.

For a local 5-step, 512x384 development run, select `development.enabled: true` and use the explicit `development.resolution` / `development.num_inference_steps` values in the caller. The development PGS index list has one correction step; the experiment config retains its full selected-step list. For official evaluation, leave development disabled and use `resolution: 1024x768` and the configured 50 inference steps. The caller must pass a local BAM-Net M* mask for B/C; there is no implicit mask substitution.

### AutoMasker (SCHP + DensePose)

The exporter remains:

```bash
python catvtoncode/Comfyui-CatVTON/tools/automasker_batch.py --split test --input-dir data/VTON-HD/test/image --output-dir data/derived/test --models checkpoints/CatVTON --limit 5 --resume --device auto
```

The dataset and required SCHP/DensePose checkpoint files are present, but this command is currently blocked by missing Detectron2/DensePose runtime dependencies and SCHP's CUDA extension. No model was substituted and no real AutoMasker inference has run.

### ViTPose

The wrapper's configuration path is in `configs/experiment.yaml`. A local file `checkpoints/ViTPose/vitpose_small.pth` exists, but compatibility with the configured Base WholeBody architecture is unverified. `FrozenViTPose.from_checkpoint(config, checkpoint, device=resolve_device())` also requires the absent compatible legacy MMPose/MMCV runtime; only dummy-model gradient tests are currently supported.

### VITON-HD baseline

The test script now chooses the shared device and loads checkpoints with `map_location`. It still needs its local VITON-HD dataset, three model checkpoints, and `torchgeometry`, which are not included in this core Mac environment. Training code in third-party SCHP is not a supported Mac workflow because of its custom CUDA extension.

## Memory settings

Use batch size 1, FP32, zero workers, and one gradient accumulation step as the default. This keeps loading conservative for a 16 GB unified-memory machine. Do not load AutoMasker, ViTPose, and CatVTON simultaneously. Release an unused component and call `models.device.clear_device_cache()` when switching between MPS/CUDA models. PGS adds decoder/pose activations at configured correction steps; begin development with one or two correction steps only after a compatible checkpoint is available.

## Data and checkpoints

Portable dataset root: `data/VTON-HD` (the local directory name; the upstream dataset is VITON-HD). Run the read-only integrity check with `python scripts/validate_vitonhd.py --root data/VTON-HD --split all`. SCHP LIP/ATR and DensePose checkpoints plus one unverified ViTPose checkpoint file are present under `checkpoints/`. The CatVTON diffusion base, VAE, attention adapter, and BAM-Net assets are not present. Do not relabel validation as test, change splits, or change evaluation resolution to accommodate local memory without an explicit experiment config change.
