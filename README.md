# Hardware-Realistic Phase-Only CGH Optimization: Robustness Ablation

This repository integrates **STE-based SLM phase quantization** and **Poisson photon shot noise** into a differentiable holographic propagation pipeline for phase-only CGH optimization.

## Overview

- Forward model: phase-only SLM, propagation, intensity formation, hardware degradation.
- Hardware degradations:
  - SLM phase quantization via `torch.autograd.Function` + Straight-Through Estimator (STE).
  - Poisson shot noise parameterized by peak photons per pixel.
- Optimization target: amplitude-domain reconstruction (`MSE` or `L1`) with Adam.
- Ablation protocol: ideal baseline, severe quantization, severe noise, and combined degradation.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1) Run ablation optimization

```bash
python main.py \
  --iters 500 \
  --height 256 \
  --width 256 \
  --device auto \
  --output-dir outputs/mps_full_500
```

`--device` supports:
- `auto` (prefers CUDA if available, then Apple MPS, then CPU),
- `cuda` (NVIDIA GPU),
- `cpu`.

### 2) Generate the presentation figure

```bash
python generate_figure.py \
  --output-dir outputs/mps_full_500 \
  --figure-name comparison_plot.png
```

The figure is a 1x5 grayscale comparison with PSNR in subplot titles:
1. Ground Truth Target
2. Ideal Baseline
3. 2-bit Quantization
4. 1e3 Photons
5. Combined (4-bit + 1e3)

## Ablation Results

The following metrics are from `outputs/mps_full_500/ablation_results.csv`:

| Case | Quantization Bits | Peak Photons | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| Ideal Baseline | FP32 | None | 13.73 | 0.0605 |
| Severe Quantization (2-bit) | 2 | None | 13.54 | 0.0591 |
| Severe Noise (1e3 photons) | FP32 | 1000 | 7.94 | 0.0251 |
| Combined Degradation (4-bit + 1e3 photons) | 4 | 1000 | 7.75 | 0.0219 |

## Notes on Perceptual Failure Modes

- **Quantization artifacts (2-bit):** collapsing continuous phase into only four phase levels removes critical phase degrees of freedom, which induces phase banding, stronger ghost replicas, and more visible symmetric diffraction orders from coarse phase steps.
- **Photon-noise speckle (1e3 photons):** at low photon budgets, Poisson variance is comparable to or larger than the local signal in many pixels, so amplitude estimates become shot-noise dominated and visually appear as strong high-frequency speckle with severe fidelity loss.
- **STE optimization behavior:** STE preserves gradient flow through non-differentiable quantization, enabling partial pre-compensation of discretization effects during optimization; however, STE cannot recover information destroyed by photon starvation, so noise-limited regimes remain fundamentally constrained by counting statistics.

## Key Files

- `forward_model.py`: hardware-realistic forward model and degradations.
- `main.py`: optimization and ablation runner.
- `generate_figure.py`: presentation figure generation.
- `outputs/mps_full_500/ablation_results.csv`: final metrics.
- `outputs/mps_full_500/comparison_plot.png`: final comparison figure.
