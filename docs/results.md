# Results

Status: phase 1. **Nothing has been trained, filtered, simulated or
synthesised, so there are no results.** This file is the shape the report will
take; every cell is filled from a generated file under `results/`, never by
hand.

## Classifier

Source: `results/classifier/metrics.json` (test set only).

| Metric | Value |
|---|---|
| Accuracy | not run |
| Macro F1 | not run |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| clean | - | - | - | - |
| salt_pepper | - | - | - | - |
| gaussian | - | - | - | - |
| speckle | - | - | - | - |

Confusion matrix: `results/classifier/confusion_matrix.png` (not generated).
Training curves: `results/classifier/training_curves.png` (not generated).

## Denoising quality

Source: `results/denoising/quality_metrics.csv`.

| Noise class | Noisy MSE | Denoised MSE | Noisy PSNR | Denoised PSNR | Noisy SSIM | Denoised SSIM | PSNR gain |
|---|---|---|---|---|---|---|---|
| salt_pepper | - | - | - | - | - | - | - |
| gaussian | - | - | - | - | - | - | - |
| speckle | - | - | - | - | - | - | - |

## Baseline comparison

Source: `results/denoising/comparison.csv`.

| Baseline | MSE | PSNR | SSIM | Time per image |
|---|---|---|---|---|
| A: no filtering | - | - | - | - |
| B: median for everything | - | - | - | - |
| C: adaptive selection | - | - | - | - |

Whether C beats B is the question this project exists to answer. It gets
answered from measurements, whichever way they fall.

## RTL verification

See `docs/verification.md`. Not run.

## Hardware

See `docs/hardware.md`. No board selected, no synthesis run.

## Performance

| Mode | Avg | Min | Max | FPS | Latency |
|---|---|---|---|---|---|
| Single image (software) | - | - | - | - | - |
| Video (software) | - | - | - | - | - |
| FPGA simulation | - | - | - | - | - |
| FPGA hardware | - | - | - | - | - |

Real-time is not claimed anywhere until it is measured against a stated target
frame rate.
