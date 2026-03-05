"""
Generate a 1x5 comparison figure for holographic ablation outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def safe_name(name: str) -> str:
    """Convert case name to a filesystem-friendly string."""
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "+"}:
            keep.append("_")
    return "".join(keep).strip("_")


def load_array_or_png(base_path_without_ext: str) -> np.ndarray:
    """Load image data from NPY first, then PNG as fallback."""
    npy_path = f"{base_path_without_ext}.npy"
    png_path = f"{base_path_without_ext}.png"

    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    if os.path.exists(png_path):
        return np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
    raise FileNotFoundError(f"Missing both NPY and PNG for: {base_path_without_ext}")


def load_psnr_map(csv_path: str) -> Dict[str, float]:
    """Load PSNR metrics by case from ablation CSV."""
    psnr_map: Dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr_map[row["case"]] = float(row["psnr_db"])
    return psnr_map


def resolve_panels(output_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Resolve target and reconstruction arrays for all required panels."""
    target = load_array_or_png(os.path.join(output_dir, "target"))

    case_names = {
        "Ideal Baseline": "Ideal Baseline",
        "2-bit Quantization": "Severe Quantization (2-bit)",
        "1e3 Photons": "Severe Noise (1e3 photons)",
        "Combined (4-bit + 1e3)": "Combined Degradation (4-bit + 1e3 photons)",
    }

    reconstructions: Dict[str, np.ndarray] = {}
    for panel_name, case_name in case_names.items():
        case_dir = os.path.join(output_dir, safe_name(case_name))
        reconstructions[panel_name] = load_array_or_png(os.path.join(case_dir, "reconstruction"))
    return target, reconstructions


def generate_figure(output_dir: str, figure_name: str) -> str:
    """Generate and save a presentation-ready 1x5 comparison plot."""
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    psnr_map = load_psnr_map(csv_path)
    target, reconstructions = resolve_panels(output_dir)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    panels = [
        ("Ground Truth Target", target, None),
        ("Ideal Baseline", reconstructions["Ideal Baseline"], "Ideal Baseline"),
        ("2-bit Quantization", reconstructions["2-bit Quantization"], "Severe Quantization (2-bit)"),
        ("1e3 Photons", reconstructions["1e3 Photons"], "Severe Noise (1e3 photons)"),
        (
            "Combined (4-bit + 1e3)",
            reconstructions["Combined (4-bit + 1e3)"],
            "Combined Degradation (4-bit + 1e3 photons)",
        ),
    ]

    for ax, (title, image, case_key) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        if case_key is None:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"{title}\nPSNR: {psnr_map[case_key]:.2f} dB", fontsize=12)

    fig.tight_layout(pad=0.6, w_pad=0.5)
    save_path = os.path.join(output_dir, figure_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1x5 ablation comparison figure.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory with ablation outputs.")
    parser.add_argument("--figure-name", type=str, default="comparison_plot.png", help="Output figure filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = generate_figure(args.output_dir, args.figure_name)
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()
