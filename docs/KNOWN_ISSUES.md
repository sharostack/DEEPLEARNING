# Known issues and blockers

- `catvtoncode/Comfyui-CatVTON` points to `pzc163/Comfyui-CatVTON`, not the canonical `Zheng-Chong/CatVTON` repository. The inference loop is present, but the SD inpainting base, VAE, and CatVTON attention adapter weights are absent locally.
- The dataset is at `data/VTON-HD`; its pair lists are `train_pairs.txt` and `test_pairs.txt`. Configs and docs now point to these. Full read-only validation passed for 11,647 train and 2,032 test pairs, all referenced source assets, and 1024x768 image dimensions.
- `.venv` is ARM64 Python 3.12.2 with Torch 2.11.0 / torchvision 0.26.0. `huggingface-hub` was corrected from incompatible 2.4.0 to pinned 0.36.0 after `pip check` found a Transformers constraint violation. Sandbox-only runs cannot see MPS, but a hardware-enabled run of the full suite exercised MPS and passed.
- SCHP LIP/ATR and DensePose checkpoint files exist. SCHP sizes and SHA-256 hashes match the local CatVTON HF snapshot metadata. AutoMasker execution is blocked by missing Detectron2/MMCV and the SCHP CUDA extension.
- A ViTPose checkpoint file exists, but provenance/config compatibility are unknown; MMPose/MMCV are absent. Person 3's seven SIPS definitions and Person 1's BAM-Net checkpoint/mask export remain outstanding.
- No PropVTON/BAM-Net training entry point exists in this repository. No real CatVTON inference, AutoMasker execution, ViTPose run, metrics, or training has succeeded.
