# Experiments and evaluation

Three configurable scenarios are defined in `configs/experiment_a.yaml`, `experiment_b.yaml`, and `experiment_c.yaml`: A uses original AutoMasker, B uses the BAM-Net refined mask, and C adds PGS. Shared defaults preserve 1024×768, batch size 1, 50 denoising steps, PGS forward indices 10/15/20/25/30/35, correction strength 0.01, and soft-argmax temperature 0.1. Development-only resolution and step settings are explicitly separate. The seven SIPS ratio definitions remain pending Person 3.

Evaluation requirements: SSIM, LPIPS, FID, CLIP similarity, PNPE, inference time, and peak GPU memory. No evaluation scripts are provided yet because metric preprocessing/aggregation and PNPE definition are not specified, and generated predictions do not exist. Do not report fabricated measurements. Use fixed official test pairs and seeds across A/B/C after acquiring the dataset and runnable NVIDIA setup.
