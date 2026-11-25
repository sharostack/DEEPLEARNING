# Proportion-Guided Sampling (PGS)

## Data flow and status

`pgs/vitpose.py` wraps a legacy MMPose ViTPose model. It freezes all pose-model parameters but calls its backbone, optional neck and heatmap head directly, avoiding MMPose's ordinary CPU/NumPy decoder so gradients remain available to the image. The generated person image is decoded from the predicted clean diffusion latent, passed through frozen ViTPose, then through soft-argmax, SIPS and the loss. The original person image is processed once without gradients and supplies the reference ratios. The garment is never a proportion reference.

The wrapper currently resizes the full image to the configured ViTPose input size (192x256 for the included WholeBody config) and applies the config's normalization. This is a structural differentiable path; equivalence to a production top-down detector/crop preprocessing path has not been verified. ViTPose requires the compatible legacy MMPose/MMCV dependencies and a local checkpoint. No ViTPose checkpoint is bundled or claimed present.

`soft_argmax_2d` consumes `[B,J,H,W]`, spatially normalizes with `softmax(heatmap / temperature)`, returns normalized `(x,y)` coordinates in `[0,1]` and maximum spatial probability confidence. Non-finite joints become zero coordinates/confidence. `calculate_sips` accepts definitions as `(a,b,c,d)` and calculates `distance(a,b)/distance(c,d)`. It invalidates low-confidence joints and zero denominators. The seven final anatomical definitions, keypoint indices, confidence thresholds and coordinate convention must come from Person 3; the config deliberately leaves `ratio_definitions` empty until then.

The loss is confidence weighted:

`L = sum_r w_r * (ratio_generated_r - ratio_original_r)^2 / max(sum_r w_r, epsilon)`

where `w_r` is the minimum confidence among the four keypoints used by ratio `r`; invalid ratios contribute zero weight. If no valid ratios remain, the correction is skipped.

## Latent correction and scheduler

CatVTON concatenates `[person_latent, garment_latent]` along latent height (`dim=-2`). `normalized_masked_correction` splits by person height, applies the editable mask only to the person half, and concatenates an exactly zero garment correction. It uses `torch.autograd.grad(loss, x_t)` only, RMS normalizes the masked gradient, subtracts the configured strength, and detaches the corrected latent. The editable mask is nearest-neighbor resized from CatVTON's person mask; `whole_person` is an optional configured alternative.

At selected forward-loop indices, the pipeline enables gradients only for the PGS operation. It predicts noise at current `x_t`, reconstructs predicted clean `x0` using the configured DDIM prediction type (`epsilon`, `sample`, or `v_prediction`), decodes the person half, and evaluates the differentiable objective. It corrects `x_t`, reevaluates the denoiser at corrected `x_t`, then performs the existing DDIM scheduler step. This is an explicit latent-space correction before the scheduler update, not a replacement for the scheduler update. The reconstruction follows the installed Diffusers DDIM equations and applies the scheduler's configured thresholding/clipping behavior. Unrecognized prediction types raise an error. The scheduler remains the CatVTON DDIM scheduler; other scheduler classes are not supported by this helper.

The U-Net and VAE parameters are frozen; generated-image gradients can flow through their operations to the latent. Normal inference remains under the pipeline's `no_grad`; only the scheduled correction block enters `torch.enable_grad()`. A/B continue to use the existing pipeline path when no enabled `pgs_runtime` is passed. `inference/catvton_masked.py::run_pair` forwards `pgs_runtime` through its pipeline kwargs.

## Configuration and execution

Experiment C is configured with `mask_source: bamnet`, `pgs.enabled: true`, a ViTPose config path, a deliberately null checkpoint path, temperature, selected forward indices, correction strengths, mask policy and an empty ratio-definition map. Fill the checkpoint path and seven agreed definitions before constructing `PGSObjective`; it raises if definitions are absent. Experiments A and B keep PGS disabled.

Run CPU synthetic checks with:

```bash
python -m pytest -q tests/test_pgs_synthetic.py
```

These checks use synthetic tensors and dummy models only. No real ViTPose checkpoint, GPU execution, CatVTON inference, or PGS performance improvement has been verified. Person 3 must provide the ratio definitions. A CUDA machine plus compatible MMPose/MMCV and local ViTPose weights are required for real evaluation.
