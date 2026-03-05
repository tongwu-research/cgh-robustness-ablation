
"Create a presentation-ready comparison figure from ablation outputs."


from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def safe_name(name: str) -> str:
    """Convert case name to a filesystem-friendly string."""
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "+"}:
            keep.append("_")
    return "".join(keep).strip("_")


def load_psnr_map(csv_path: str) -> Dict[str, float]:
    """Load PSNR values from ablation CSV by case name."""
    psnr_map: Dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr_map[row["case"]] = float(row["psnr_db"])
    return psnr_map


def load_images(output_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Load target and reconstructed amplitudes for plotting."""
    target = np.load(os.path.join(output_dir, "target.npy"))

    cases = [
        ("Ideal Baseline", "Ideal Baseline"),
        ("Severe Quantization (2-bit)", "2-bit Quantization"),
        ("Severe Noise (1e3 photons)", "1e3 Photons"),
        ("Combined Degradation (4-bit + 1e3 photons)", "Combined (4-bit + 1e3)"),
    ]

    panels: List[Tuple[str, np.ndarray]] = [("Ground Truth Target", target)]
    for case_name, panel_title in cases:
        case_dir = os.path.join(output_dir, safe_name(case_name))
        recon = np.load(os.path.join(case_dir, "reconstruction.npy"))
        panels.append((panel_title, recon))
    return panels


def make_plot(output_dir: str, figure_name: str) -> str:
    """Generate and save the comparison figure."""
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    psnr_map = load_psnr_map(csv_path)
    panels = load_images(output_dir)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    case_name_map = {
        "Ideal Baseline": "Ideal Baseline",
        "2-bit Quantization": "Severe Quantization (2-bit)",
        "1e3 Photons": "Severe Noise (1e3 photons)",
        "Combined (4-bit + 1e3)": "Combined Degradation (4-bit + 1e3 photons)",
    }

    for ax, (panel_title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        if panel_title == "Ground Truth Target":
            ax.set_title("Ground Truth Target", fontsize=12)
        else:
            case_name = case_name_map[panel_title]
            psnr = psnr_map[case_name]
            ax.set_title(f"{panel_title}\nPSNR: {psnr:.2f} dB", fontsize=12)

    fig.tight_layout(pad=0.6, w_pad=0.5)
    save_path = os.path.join(output_dir, figure_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hologram ablation study outputs.")
    parser.add_argument("--output-dir", type=str, required=True, help="Ablation output directory.")
    parser.add_argument(
        "--figure-name",
        type=str,
        default="comparison_plot.png",
        help="Output figure filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = make_plot(output_dir=args.output_dir, figure_name=args.figure_name)
    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()
