# PropVTON setup and resource inventory

## Current Apple Silicon host audit

Audited on macOS 26.6.2, ARM64, Apple M4, with Python 3.12.2 from `/opt/anaconda3/bin/python`. The existing base environment contains Torch 2.7.1 and torchvision 0.22.1, but `torch.backends.mps.is_available()` is false. A clean project environment is described by `requirements-macos.txt`; no pre-existing environment was overwritten.

The dataset is present locally at `data/VTON-HD` (upstream name: VITON-HD) with official `train` and `test` folders and pair lists. The configs use this local root. SCHP LIP/ATR and DensePose AutoMasker checkpoints, plus a ViTPose checkpoint, are present under `checkpoints/`; CatVTON diffusion and attention adapter weights and BAM-Net assets are not present. A real-sample loader path mismatch was fixed in the VITON-HD loader and validated on train/test samples.

The checked-in VITON-HD and ViTPose source repositories are separate from dataset/checkpoint resources. BAM-Net checkpoints/masks are supplied by Person 1; the agreed seven PGS ratio definitions are supplied by Person 3.

## Known platform boundaries

- CatVTON's Diffusers pipeline, mask adapter, and PGS tensor primitives are written with device-agnostic tensor operations and can be developed on MPS/CPU. Actual MPS CatVTON inference remains unverified until dependencies and weights are present.
- The official AutoMasker combines SCHP and DensePose. The checked-in SCHP implementation imports a custom extension that compiles CUDA `.cu` sources (`model/SCHP/modules/functions.py`). DensePose uses Detectron2 v0.6 and CUDA-oriented compiled operators. Neither is included in the Mac environment; the original code is retained.
- The ViTPose checkout uses the legacy MMPose/MMCV stack, which is not installed. A 93 MiB file named `checkpoints/ViTPose/vitpose_small.pth` exists, but its provenance and compatibility with the configured Base WholeBody config have not been verified. The PropVTON differentiable wrapper is synthetic-testable; real checkpoint loading/MPS behavior is unverified.
- The original VITON-HD `test.py` uses `torchgeometry`; this package is deliberately outside the core environment. Device placement in its script is portable now, but the legacy baseline has not been run here.
- Some legacy SCHP training/extraction utilities outside AutoMasker's direct call path retain CUDA-only training logic and old machine-specific sample-path defaults. They are not used by the PropVTON exporter and are not part of the supported Mac workflow.
- `xformers`, Detectron2, and DensePose are intentionally omitted from the macOS requirements. The CatVTON attention processor uses PyTorch's native attention path; `xformers` is not needed for the project's standalone inference path.

See `MAC_SETUP.md` for environment creation, commands, and supported-workflow notes.
