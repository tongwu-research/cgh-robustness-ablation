| Case | Quantization Bits | Peak Photons | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| Ideal Baseline | FP32 | None | 13.732 | 0.0605 |
| Severe Quantization (2-bit) | 2 | None | 13.535 | 0.0591 |
| Severe Noise (1e3 photons) | FP32 | 1000 | 7.943 | 0.0251 |
| Combined Degradation (4-bit + 1e3 photons) | 4 | 1000 | 7.754 | 0.0219 |
