# Dataset placement

Keep VITON-HD outside Git. The local copy is at `data/VTON-HD` (upstream project name VITON-HD). The official repository documents the paired train/test split:
https://github.com/shadow2496/VITON-HD

The authors report 11,647 training pairs and 2,032 test pairs at 1024×768. Preserve the provided split and annotations. For evaluation, use the official test split only; do not use test data for training or tuning. Set local data paths through an untracked config or environment variable.

The local `train_pairs.txt` and `test_pairs.txt` contain 11,647 and 2,032 pairs. Run `python scripts/validate_vitonhd.py --root data/VTON-HD --split all` to check pair references, annotations, JSON keypoints, image decodability, and official 1024x768 dimensions. Dataset assets are ignored by Git.
