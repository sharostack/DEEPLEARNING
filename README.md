# PropVTON

Proportion-Preserving Virtual Try-On with a frozen CatVTON diffusion backbone.

This repository contains a CatVTON mask adapter and synthetic-tested PGS components. The VITON-HD dataset, SCHP LIP/ATR checkpoints, DensePose checkpoint, and a local ViTPose checkpoint are present. BAM-Net is supplied by Person 1; its checkpoint and outputs are not included here. PGS has no real ViTPose runtime or end-to-end validation yet.

## Status

- `catvtoncode/Comfyui-CatVTON/` is a ComfyUI CatVTON checkout, not the canonical Zheng-Chong/CatVTON repository. The SD inpainting base, CatVTON attention adapter, and local VAE weights are absent.
- VITON-HD is at `data/VTON-HD` (11,647 train / 2,032 test pairs). Full dataset integrity validation passed. BAM-Net weights/masks are absent. SCHP LIP/ATR and DensePose checkpoint files are present; AutoMasker execution is blocked by its CUDA extension and missing Detectron2/DensePose runtime dependencies.
- `.venv` uses ARM64 Python 3.12.2, Torch 2.11.0 and torchvision 0.26.0. The sandbox-only suite is CPU-only, but a hardware-enabled full suite passed its MPS checks. CatVTON, AutoMasker, and ViTPose real model execution have not been run.
- A ViTPose file exists at `checkpoints/ViTPose/vitpose_small.pth`, but its provenance and compatibility with the configured Base WholeBody config have not been established. MMPose/MMCV and Person 3's seven ratio definitions are absent.
- Latest hardware-enabled suite: 38 passed, 1 skipped (CUDA unavailable). Sandbox-only suite: 37 passed, 2 skipped (CUDA and MPS unavailable in sandbox); see [MAC_SETUP.md](MAC_SETUP.md) and [docs/SETUP.md](docs/SETUP.md).
- See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for setup blockers and [docs/SETUP.md](docs/SETUP.md) for the inventory and intended setup.

## Repository map

```text
configs/       Experiment configuration
models/        Model adapters and frozen model interfaces
datasets/      Dataset metadata/readme only; data stays out of Git
inference/     Baseline and experiment inference entry points
pgs/           Differentiable PGS components and CatVTON integration
evaluation/    Metric and result aggregation tools
scripts/       Setup and experiment helpers
tests/         Unit and integration tests
outputs/       Local generated images/logs; ignored by Git
docs/          Setup, design, and experiment records
```

## Setup

See [MAC_SETUP.md](MAC_SETUP.md) for the native Mac environment and current platform boundaries.

## Experiments

The planned comparisons are original CatVTON AutoMasker, CatVTON with Person 1's BAM-Net refined mask, and CatVTON + BAM-Net + PGS. Use the same official VITON-HD test examples across configurations. Do not train or tune on test examples. Experiments are blocked until canonical CatVTON source/compatible assets are selected, model weights are available, AutoMasker dependencies work, Person 1 supplies BAM-Net weights/masks, and Person 3 supplies SIPS definitions for Experiment C. No PropVTON/BAM-Net training entry point is present in this repository.

## Data and weights

Never commit VITON-HD images, pretrained weights, generated datasets, credentials, or other restricted assets. See `.gitignore` and `datasets/README.md`.

## License

Project-specific licensing and attribution are pending confirmation from the project team. CatVTON's upstream code, checkpoints, and demo are marked CC BY-NC-SA 4.0; comply with upstream terms when obtaining or using them.
