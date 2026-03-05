"""
Phase-only CGH optimization with hardware-realistic forward degradations.

This script runs an ablation study for:
- Ideal baseline (continuous phase, no noise)
- Severe quantization (2-bit SLM phase)
- Severe noise (1e3 peak photons)
- Combined degradation (4-bit SLM phase + 1e3 peak photons)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from forward_model import ASMConfig, HardwareHolographicDisplay, wrap_phase


@dataclass
class AblationCase:
    """Single ablation configuration."""

    name: str
    quantization_bits: Optional[int]
    peak_photons: Optional[float]


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic behavior as much as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str) -> logging.Logger:
    """Configure a logger that writes both console and file logs."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("hardware_cgh")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device from CLI argument."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _download_cameraman_image(download_path: str) -> bool:
    """Try downloading the standard cameraman image."""
    url = "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/camera.png"
    try:
        urllib.request.urlretrieve(url, download_path)
        return True
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _generate_logo_target(path: str, resolution: Tuple[int, int]) -> None:
    """Generate a clear grayscale logo target if download is unavailable."""
    height, width = resolution
    image = Image.new("L", (width, height), color=20)
    draw = ImageDraw.Draw(image)

    draw.rectangle([(12, 12), (width - 12, height - 12)], outline=220, width=3)
    draw.ellipse([(width * 0.08, height * 0.18), (width * 0.35, height * 0.45)], outline=180, width=2)
    draw.rectangle([(width * 0.62, height * 0.16), (width * 0.9, height * 0.44)], outline=180, width=2)
    draw.line([(width * 0.1, height * 0.74), (width * 0.9, height * 0.74)], fill=180, width=2)

    font = ImageFont.load_default()
    text = "UCL CGH"
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    text_x = (width - text_w) // 2
    text_y = int(height * 0.53) - text_h // 2
    draw.text((text_x, text_y), text, fill=235, font=font)

    image.save(path)


def prepare_target_image(
    target_path: Optional[str],
    resolution: Tuple[int, int],
    output_dir: str,
    logger: logging.Logger,
) -> str:
    """Resolve target image path by using user input or auto-preparing a standard target."""
    if target_path is not None:
        return target_path

    target_assets_dir = os.path.join(output_dir, "target_assets")
    os.makedirs(target_assets_dir, exist_ok=True)
    downloaded_path = os.path.join(target_assets_dir, "camera_downloaded.png")
    generated_path = os.path.join(target_assets_dir, "logo_generated.png")

    if _download_cameraman_image(downloaded_path):
        logger.info("Using downloaded standard target image: %s", downloaded_path)
        return downloaded_path

    logger.info("Download failed, generating fallback logo target image.")
    _generate_logo_target(generated_path, resolution)
    return generated_path


def load_target_amplitude(
    target_path: Optional[str],
    resolution: Tuple[int, int],
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, str]:
    """
    Load target amplitude from image and return tensor plus the source image path.

    Returns:
        (target tensor [1,1,H,W] in [0,1], source image path)
    """
    height, width = resolution
    resolved_target_path = prepare_target_image(
        target_path=target_path,
        resolution=resolution,
        output_dir=output_dir,
        logger=logger,
    )

    img = Image.open(resolved_target_path).convert("L").resize((width, height), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    target = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return target.unsqueeze(0).unsqueeze(0), resolved_target_path


def normalize_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] per sample for stable metric computation."""
    min_val = x.amin(dim=(-2, -1), keepdim=True)
    max_val = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val**2) / mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window_2d = torch.outer(gauss_1d, gauss_1d)
    window_2d = window_2d / window_2d.sum()
    return window_2d.unsqueeze(0).unsqueeze(0)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Compute SSIM for grayscale images."""
    window = _gaussian_window(window_size=window_size, sigma=sigma, device=pred.device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding)
    mu_y = F.conv2d(target, window, padding=padding)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def save_tensor_image(path: str, tensor: torch.Tensor) -> None:
    """Save a [1,1,H,W] tensor in [0,1] to PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def safe_name(name: str) -> str:
    """Convert case name to a filesystem-friendly string."""
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "+"}:
            keep.append("_")
    return "".join(keep).strip("_")


def optimize_phase(
    case: AblationCase,
    target_amp: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Run phase-only optimization for one ablation case."""
    height, width = target_amp.shape[-2], target_amp.shape[-1]
    asm_cfg = ASMConfig(
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        distance=args.distance,
    )
    display = HardwareHolographicDisplay(
        resolution=(height, width),
        propagation_mode=args.propagation_mode,
        asm_cfg=asm_cfg,
        quantization_bits=case.quantization_bits,
        peak_photons=case.peak_photons,
        device=device,
    ).to(device)

    phase = torch.empty_like(target_amp).uniform_(-math.pi, math.pi)
    phase = torch.nn.Parameter(phase)
    optimizer = torch.optim.Adam([phase], lr=args.lr)

    logger.info(
        "Start case='%s' | bits=%s | peak_photons=%s",
        case.name,
        str(case.quantization_bits),
        str(case.peak_photons),
    )

    for step in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        forward_dict = display(phase)
        recon_amp = forward_dict["amplitude"].to(dtype=torch.float32)
        target_for_loss = target_amp.to(dtype=torch.float32)

        if args.loss == "mse":
            loss = F.mse_loss(recon_amp, target_for_loss)
        else:
            loss = F.l1_loss(recon_amp, target_for_loss)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            phase.data = wrap_phase(phase.data)

        if step == 1 or step % args.log_every == 0 or step == args.iters:
            with torch.no_grad():
                recon_eval = normalize_to_unit(recon_amp.detach())
                target_eval = normalize_to_unit(target_amp)
                psnr = compute_psnr(recon_eval, target_eval)
                ssim = compute_ssim(recon_eval, target_eval)
            logger.info(
                "Case='%s' | iter=%d/%d | loss=%.6f | PSNR=%.3f | SSIM=%.4f",
                case.name,
                step,
                args.iters,
                loss.item(),
                psnr,
                ssim,
            )

    with torch.no_grad():
        final_dict = display(phase)
        final_recon = normalize_to_unit(final_dict["amplitude"])
        final_target = normalize_to_unit(target_amp)
        final_psnr = compute_psnr(final_recon, final_target)
        final_ssim = compute_ssim(final_recon, final_target)

        case_dir = os.path.join(output_dir, safe_name(case.name))
        os.makedirs(case_dir, exist_ok=True)
        save_tensor_image(os.path.join(case_dir, "target.png"), final_target)
        save_tensor_image(os.path.join(case_dir, "reconstruction.png"), final_recon)
        np.save(os.path.join(case_dir, "target.npy"), final_target.squeeze().detach().cpu().numpy())
        np.save(os.path.join(case_dir, "reconstruction.npy"), final_recon.squeeze().detach().cpu().numpy())

        phase_vis = (wrap_phase(final_dict["quantized_phase"]) + math.pi) / (2.0 * math.pi)
        save_tensor_image(os.path.join(case_dir, "quantized_phase.png"), phase_vis)
        np.save(os.path.join(case_dir, "quantized_phase.npy"), phase_vis.squeeze().detach().cpu().numpy())

    logger.info(
        "Final case='%s' | PSNR=%.3f dB | SSIM=%.4f",
        case.name,
        final_psnr,
        final_ssim,
    )
    return {
        "case": case.name,
        "quantization_bits": -1 if case.quantization_bits is None else case.quantization_bits,
        "peak_photons": -1.0 if case.peak_photons is None else float(case.peak_photons),
        "psnr_db": final_psnr,
        "ssim": final_ssim,
    }


def build_ablation_cases() -> List[AblationCase]:
    """Create the requested ablation conditions."""
    return [
        AblationCase(name="Ideal Baseline", quantization_bits=None, peak_photons=None),
        AblationCase(name="Severe Quantization (2-bit)", quantization_bits=2, peak_photons=None),
        AblationCase(name="Severe Noise (1e3 photons)", quantization_bits=None, peak_photons=1e3),
        AblationCase(name="Combined Degradation (4-bit + 1e3 photons)", quantization_bits=4, peak_photons=1e3),
    ]


def write_results(results: List[Dict[str, float]], output_dir: str, logger: logging.Logger) -> None:
    """Write CSV and Markdown results files."""
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "quantization_bits", "peak_photons", "psnr_db", "ssim"],
        )
        writer.writeheader()
        writer.writerows(results)

    md_lines = [
        "| Case | Quantization Bits | Peak Photons | PSNR (dB) | SSIM |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        q_bits = "FP32" if row["quantization_bits"] < 0 else str(int(row["quantization_bits"]))
        photons = "None" if row["peak_photons"] < 0 else f"{row['peak_photons']:.0f}"
        md_lines.append(
            f"| {row['case']} | {q_bits} | {photons} | {row['psnr_db']:.3f} | {row['ssim']:.4f} |"
        )
    md_table = "\n".join(md_lines)

    md_path = os.path.join(output_dir, "ablation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    logger.info("Saved results CSV to: %s", csv_path)
    logger.info("Saved results Markdown to: %s", md_path)
    logger.info("\n%s", md_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware-aware CGH phase optimization ablation.")
    parser.add_argument("--target", type=str, default=None, help="Path to target grayscale image.")
    parser.add_argument("--height", type=int, default=256, help="Target image height.")
    parser.add_argument("--width", type=int, default=256, help="Target image width.")
    parser.add_argument("--iters", type=int, default=500, help="Optimization steps per ablation case.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate for phase optimization.")
    parser.add_argument("--loss", type=str, choices=["mse", "l1"], default="mse", help="Optimization loss.")
    parser.add_argument("--seed", type=int, default=2026, help="Global random seed.")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu.")
    parser.add_argument("--log-every", type=int, default=50, help="Log interval in optimization steps.")
    parser.add_argument(
        "--propagation-mode",
        type=str,
        choices=["asm", "dummy"],
        default="asm",
        help="Propagation backend.",
    )

    # ASM parameters.
    parser.add_argument("--wavelength", type=float, default=532e-9, help="Wavelength in meters.")
    parser.add_argument("--pixel-pitch", type=float, default=8e-6, help="SLM pixel pitch in meters.")
    parser.add_argument("--distance", type=float, default=0.2, help="Propagation distance in meters.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", f"trial_{stamp}")
    logger = setup_logger(output_dir)

    logger.info("Runtime device: %s", str(device))
    logger.info("Output directory: %s", output_dir)
    logger.info("Ablation iterations per case: %d", args.iters)

    target_amp, target_source_path = load_target_amplitude(
        target_path=args.target,
        resolution=(args.height, args.width),
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    normalized_target = normalize_to_unit(target_amp)
    save_tensor_image(os.path.join(output_dir, "target_input.png"), normalized_target)
    np.save(os.path.join(output_dir, "target.npy"), normalized_target.squeeze().detach().cpu().numpy())
    logger.info("Target loaded from: %s", target_source_path)
    logger.info("Target loaded with shape=%s", tuple(target_amp.shape))

    cases = build_ablation_cases()
    results: List[Dict[str, float]] = []
    for case in cases:
        row = optimize_phase(
            case=case,
            target_amp=target_amp,
            args=args,
            device=device,
            output_dir=output_dir,
            logger=logger,
        )
        results.append(row)

    write_results(results=results, output_dir=output_dir, logger=logger)
    logger.info("Ablation finished successfully.")


if __name__ == "__main__":
    main()
