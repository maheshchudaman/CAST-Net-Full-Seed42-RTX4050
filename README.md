# CAST-Net Full Model â€” Seed 42 on RTX 4050

This repository archives the code, configuration, and per-image evidence for
one full-capacity CAST-Net image-inpainting run on ADE20K.

## Scope and limitation

This is transparently a **single training run using seed 42**. It does not
establish variation across random initializations and must not be cited as a
three-seed result. The 2,021 test records below are per-image observations from
the one trained checkpoint, not independent training runs.

## Experiment

- Dataset: MIT ADE20K / `ADEChallengeData2016`
- Frozen split: 16,168 train / 2,021 validation / 2,021 test images
- Resolution: 256 Ã— 256
- Generator parameters: 5,276,359
- Training: 100 epochs, 101,100 optimizer steps
- Seed: 42
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB)
- Software: Python 3.12.10, PyTorch 2.13.0+cu130, CUDA 13.0
- Mixed precision: enabled
- Micro-batch: 4; gradient accumulation: 4; effective batch: 16
- Best checkpoint selection: maximum validation hole PSNR
- Best validation hole PSNR: 19.9118 dB at epoch 44

## Frozen-test results

The best checkpoint was evaluated on all 2,021 fixed test image/mask pairs.

| Metric | Result |
|---|---:|
| Hole PSNR | 19.8732 Â± 3.3306 dB |
| Full-image SSIM | 0.8999 Â± 0.0502 |
| Full-image LPIPS | 0.0696 Â± 0.0338 |
| FID | 10.4658 |
| Hole edge MAE | 0.3957 Â± 0.1643 |

Lower is better for LPIPS, FID, and edge MAE. The summary is in
`results/ade20k_rtx4050_seed42_test.summary.json`; all per-image observations
are in `results/ade20k_rtx4050_seed42_test.jsonl`.

## Integrity

- Best checkpoint SHA-256:
  `723b61c509365473407acb591e564491dd1979a4805a37e1c99824ef10f274f2`
- Test manifest SHA-256:
  `7959e8734ae69ba5b242fc7d7d004afd5d1f853ba438a9ff22b89b18fa3eae09`
- Test mask manifest SHA-256:
  `2e0b23da7ae14c2e4fcd5866dc6bda05e60abc7d6e15d0621ff0ae10f63ebd82`

The 84 MB checkpoint, ADE20K archive, fixed mask bank, and 408 MB generated
image set are intentionally excluded from ordinary Git history. The checkpoint
can be attached to a GitHub Release or archival record while retaining the
hash above. ADE20K must be obtained under its own terms.

## Reproduction

Install dependencies in a Python 3.12 environment with CUDA-enabled PyTorch,
prepare ADE20K and the frozen manifests/masks referenced by the configuration,
then run:

```powershell
python train.py --config configs/ade20k_rtx4050.yaml --seed 42 --device cuda
```

Evaluate the best checkpoint with:

```powershell
python evaluate.py `
  --config evidence/config.resolved.yaml `
  --checkpoint best.pt `
  --output results/ade20k_rtx4050_seed42_test.jsonl `
  --method CAST-Net-full --device cuda --save-images
```

Pretrained AlexNet and Inception weights are required by LPIPS and FID.

## Citation

See `CITATION.cff`. When reporting these numbers, explicitly describe them as
the seed-42 single-run result.

