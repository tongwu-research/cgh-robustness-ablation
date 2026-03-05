"""
Hardware-realistic forward model for phase-only CGH optimization.

This module provides:
1) A straight-through-estimator (STE) phase quantizer.
2) A modular holographic display model with optional propagators.
3) Poisson shot noise simulation in the sensor/intensity domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Wrap a phase tensor into [-pi, pi)."""
    return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi


class STEPhaseQuantizer(torch.autograd.Function):
    """Phase quantizer with a straight-through-estimator in backward pass."""

    @staticmethod
    def forward(ctx, phase: torch.Tensor, num_bits: int) -> torch.Tensor:
        if num_bits <= 0:
            raise ValueError("num_bits must be a positive integer when quantization is enabled.")

        levels = 2**num_bits
        step = 2.0 * math.pi / float(levels)
        phase_wrapped = wrap_phase(phase)
        shifted = phase_wrapped + math.pi

        # Use floor-binning on [0, 2*pi) to get exactly 2^N valid phase bins.
        phase_indices = torch.floor(shifted / step).clamp(0, levels - 1)
        quantized = phase_indices * step - math.pi

        # Keep strict physical support in [-pi, pi).
        quantized = torch.where(quantized >= math.pi, quantized - 2.0 * math.pi, quantized)
        quantized = torch.where(quantized < -math.pi, quantized + 2.0 * math.pi, quantized)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through estimator: pass gradients as identity.
        return grad_output, None


@dataclass
class ASMConfig:
    """Configuration for the angular spectrum propagator."""

    wavelength: float = 532e-9
    pixel_pitch: float = 8e-6
    distance: float = 0.2


class AngularSpectrumPropagator(nn.Module):
    """Free-space propagation using the Angular Spectrum Method (ASM)."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        cfg: ASMConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.height, self.width = resolution
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        transfer_fn, bandlimit_mask = self._build_transfer_function()
        self.register_buffer("transfer_fn", transfer_fn, persistent=True)
        self.register_buffer("bandlimit_mask", bandlimit_mask, persistent=True)

    def _build_transfer_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = self.height, self.width
        fx = torch.fft.fftfreq(w, d=self.cfg.pixel_pitch, device=self.device, dtype=self.dtype)
        fy = torch.fft.fftfreq(h, d=self.cfg.pixel_pitch, device=self.device, dtype=self.dtype)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

        # Physical propagating-wave mask (remove evanescent components).
        inv_lambda_sq = (1.0 / self.cfg.wavelength) ** 2
        radial_sq = fx_grid**2 + fy_grid**2
        propagating_mask = radial_sq <= inv_lambda_sq
        kz_sq = torch.where(propagating_mask, inv_lambda_sq - radial_sq, torch.zeros_like(radial_sq))
        kz = 2.0 * math.pi * torch.sqrt(kz_sq)

        # Band-limited ASM mask to avoid high-frequency aliasing at larger distances.
        z = abs(float(self.cfg.distance))
        dfx = 1.0 / (w * self.cfg.pixel_pitch)
        dfy = 1.0 / (h * self.cfg.pixel_pitch)
        if z > 0.0:
            fx_limit = 1.0 / (self.cfg.wavelength * math.sqrt(1.0 + (2.0 * z * dfx) ** 2))
            fy_limit = 1.0 / (self.cfg.wavelength * math.sqrt(1.0 + (2.0 * z * dfy) ** 2))
        else:
            fx_limit = float("inf")
            fy_limit = float("inf")
        nyquist = 1.0 / (2.0 * self.cfg.pixel_pitch)
        fx_limit = min(fx_limit, nyquist)
        fy_limit = min(fy_limit, nyquist)
        sampling_mask = (fx_grid.abs() <= fx_limit) & (fy_grid.abs() <= fy_limit)
        bandlimit_mask = propagating_mask & sampling_mask

        phase = kz * self.cfg.distance
        real = torch.cos(phase)
        imag = torch.sin(phase)
        transfer_fn = torch.complex(real, imag).to(torch.complex64)
        transfer_fn = torch.where(bandlimit_mask, transfer_fn, torch.zeros_like(transfer_fn))
        return transfer_fn, bandlimit_mask.to(torch.bool)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        # field: [B, 1, H, W], complex
        field_ft = torch.fft.fft2(field, dim=(-2, -1))
        propagated_ft = field_ft * self.transfer_fn
        propagated = torch.fft.ifft2(propagated_ft, dim=(-2, -1))
        return propagated


class DummyLearnedPropagator(nn.Module):
    """
    Lightweight learned propagator placeholder.

    It mimics a learned propagation block by operating on real/imag channels.
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        real_imag = torch.cat([field.real, field.imag], dim=1)
        out = self.net(real_imag)
        return torch.complex(out[:, :1], out[:, 1:2])


class HardwareHolographicDisplay(nn.Module):
    """Hardware-aware display forward model with quantization and Poisson noise."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        propagation_mode: str = "asm",
        asm_cfg: Optional[ASMConfig] = None,
        quantization_bits: Optional[int] = None,
        peak_photons: Optional[float] = None,
        poisson_ste: bool = True,
        intensity_norm_factor: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.height, self.width = resolution
        self.quantization_bits = quantization_bits
        self.peak_photons = peak_photons
        self.poisson_ste = poisson_ste
        self.device = device if device is not None else torch.device("cpu")

        if propagation_mode == "asm":
            self.propagator = AngularSpectrumPropagator(
                resolution=resolution,
                cfg=asm_cfg if asm_cfg is not None else ASMConfig(),
                device=self.device,
            )
        elif propagation_mode == "dummy":
            self.propagator = DummyLearnedPropagator()
        else:
            raise ValueError(f"Unsupported propagation_mode: {propagation_mode}")

        if intensity_norm_factor is not None:
            if intensity_norm_factor <= 0:
                raise ValueError("intensity_norm_factor must be positive.")
            fixed_norm = torch.tensor(float(intensity_norm_factor), device=self.device, dtype=torch.float32)
        else:
            fixed_norm = self._estimate_fixed_intensity_norm()
        self.register_buffer("intensity_norm_factor", fixed_norm, persistent=True)

    def quantize_phase(self, phase: torch.Tensor) -> torch.Tensor:
        if self.quantization_bits is None:
            # Continuous-phase baseline: pass phase through without discretization.
            return phase
        return STEPhaseQuantizer.apply(phase, int(self.quantization_bits))

    def _estimate_fixed_intensity_norm(self) -> torch.Tensor:
        """
        Estimate a fixed normalization factor once using a reference plane wave.

        This factor stays constant across optimization iterations to preserve
        consistent energy scaling.
        """
        with torch.no_grad():
            reference_phase = torch.zeros((1, 1, self.height, self.width), device=self.device, dtype=torch.float32)
            reference_field = torch.complex(torch.cos(reference_phase), torch.sin(reference_phase))
            reference_propagated = self.propagator(reference_field)
            reference_intensity = reference_propagated.real.square() + reference_propagated.imag.square()
            norm = reference_intensity.amax().clamp_min(1e-8).to(torch.float32)
        return norm

    def _normalize_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        """Normalize raw optical intensity with a fixed global scaling factor."""
        return (intensity / self.intensity_norm_factor.clamp_min(1e-8)).clamp(min=0.0, max=1.0)

    def _sample_poisson_counts(self, expected_counts: torch.Tensor) -> torch.Tensor:
        """Sample Poisson counts with an MPS-safe fallback."""
        expected_counts = expected_counts.clamp_min(0.0)
        try:
            return torch.poisson(expected_counts)
        except RuntimeError as err:
            if expected_counts.device.type != "mps":
                raise err
            sampled_cpu = torch.poisson(expected_counts.detach().cpu())
            return sampled_cpu.to(expected_counts.device)

    def _apply_poisson_noise(self, normalized_intensity: torch.Tensor) -> torch.Tensor:
        """
        Inject Poisson shot noise in count-space and return normalized intensity in [0, 1].

        Uses STE by default to preserve gradients through the stochastic node.
        """
        if self.peak_photons is None:
            return normalized_intensity
        if self.peak_photons <= 0:
            raise ValueError("peak_photons must be positive when Poisson noise is enabled.")

        peak_photons = float(self.peak_photons)
        poisson_input = (normalized_intensity.to(torch.float32) * peak_photons).clamp(min=0.0)
        poisson_input = torch.nan_to_num(poisson_input, nan=0.0, posinf=peak_photons, neginf=0.0)
        sampled_counts = self._sample_poisson_counts(poisson_input)

        if self.poisson_ste:
            # STE: forward uses stochastic counts, backward follows the deterministic path.
            noisy_counts = poisson_input + (sampled_counts - poisson_input).detach()
        else:
            noisy_counts = sampled_counts

        noisy_intensity = (noisy_counts / peak_photons).clamp(0.0, 1.0)
        return noisy_intensity

    def forward(self, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            phase: Tensor with shape [B, 1, H, W] and range approximately [-pi, pi].

        Returns:
            A dictionary containing quantized phase, intensity, noisy intensity, and amplitude.
        """
        # 1) Apply hardware phase quantization (or keep continuous for ideal baseline).
        quantized_phase = self.quantize_phase(phase)

        # 2) Convert phase-only SLM pattern to a unit-amplitude complex wavefront.
        slm_field = torch.complex(torch.cos(quantized_phase), torch.sin(quantized_phase))

        # 3) Propagate the complex field to the target plane.
        propagated_field = self.propagator(slm_field)

        # 4) Compute raw optical intensity from propagated field magnitude.
        intensity = (propagated_field.real.square() + propagated_field.imag.square()).to(torch.float32)
        normalized_intensity = self._normalize_intensity(intensity)

        # 5) Inject Poisson shot noise in photon-counting domain with STE-compatible path.
        noisy_intensity = self._apply_poisson_noise(normalized_intensity)

        # 6) Convert normalized intensity to normalized amplitude in [0, 1].
        amplitude = torch.sqrt(noisy_intensity.clamp(0.0, 1.0) + 1e-12).to(torch.float32)

        return {
            "quantized_phase": quantized_phase,
            "propagated_field": propagated_field,
            "intensity": intensity,
            "normalized_intensity": normalized_intensity,
            "noisy_intensity": noisy_intensity,
            "amplitude": amplitude,
        }
