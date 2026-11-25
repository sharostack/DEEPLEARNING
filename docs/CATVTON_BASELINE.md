# CatVTON baseline

## Status: implementation available, canonical-source and weights blocked

The repository at `catvtoncode/Comfyui-CatVTON/` is a ComfyUI implementation whose Git origin is `pzc163/Comfyui-CatVTON`, not the canonical Zheng-Chong/CatVTON repository. It contains `model/pipeline.py` with a full CatVTON denoising loop and `model/cloth_masker.py` with AutoMasker. This source was inspected, but it must not be described as the canonical upstream checkout. The local CatVTON snapshot includes SCHP and DensePose assets, but not the SD inpainting base, VAE, or CatVTON attention weights.

`inference/catvton_masked.py:run_pair` integrates mask selection with an initialized pipeline, but there is no checkpoint-loading dataset runner and no real inference has succeeded. Before a valid baseline is run, establish the allowed canonical CatVTON source/version, obtain compatible base/VAE/attention weights, satisfy AutoMasker dependencies, and validate one real pair. Record inputs, mask, output, config, seed, environment/revision, runtime, and peak accelerator memory.

Official CatVTON materials state CC BY-NC-SA 4.0. The official base identified by its README is Stable Diffusion v1.5 inpainting. Do not fetch model files until the exact dependency/revision and access/license conditions are checked against recovered source.
