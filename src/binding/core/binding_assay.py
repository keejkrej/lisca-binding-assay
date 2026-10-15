"""Picasso particle detection and per-ROI summary statistics for `lnp-binding`."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from binding.core.picasso import detect_spots_picasso

DEFAULT_MIN_GRADIENT = 250.0
DEFAULT_BOX_SIZE = 7
NOISE_GRADIENT_SIGMAS = 25.0

PARTICLE_CSV_HEADERS = [
    "roi",
    "t",
    "minutes",
    "particle",
    "x",
    "y",
    "intensity",
    "radius",
    "photons",
    "sx",
    "sy",
    "fwhm",
    "background",
    "snb",
    "net_gradient",
]

BINDING_CSV_HEADERS = [
    "roi",
    "t",
    "minutes",
    "particles",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
]

BINDING_SUMMARY_CSV_HEADERS = [
    "roi",
    "frames",
    "particles_mean",
    "intensity_mean",
    "intensity_median",
    "intensity_iqr",
    "radius_mean",
    "radius_median",
    "radius_iqr",
]


@dataclass(frozen=True)
class ParticleSpot:
    x: float
    y: float
    intensity: float
    radius: float
    photons: float
    sx: float
    sy: float
    fwhm: float
    background: float
    snb: float
    net_gradient: float


@dataclass(frozen=True)
class BindingFrameStats:
    particles: int = 0
    intensity_mean: float = 0.0
    intensity_median: float = 0.0
    intensity_iqr: float = 0.0
    radius_mean: float = 0.0
    radius_median: float = 0.0
    radius_iqr: float = 0.0


class FluorescenceHistogram:
    """Histogram of fluorescence values in [0, 65535] to estimate shared position noise."""

    def __init__(self) -> None:
        self._bins = np.zeros(65536, dtype=np.uint32)
        self._count = 0

    def sample(self, pixels: np.ndarray) -> None:
        flat = np.asarray(pixels, dtype=np.float64).ravel()[::4]
        valid = flat[np.isfinite(flat) & (flat >= 0.0)]
        rounded = np.rint(valid).astype(np.int64)
        in_range = rounded[(rounded >= 0) & (rounded < self._bins.size)]
        if in_range.size == 0:
            return
        counts = np.bincount(in_range, minlength=self._bins.size)
        self._bins += counts[: self._bins.size].astype(np.uint32)
        self._count += int(in_range.size)

    def min_gradient(self) -> float:
        if self._count == 0:
            return DEFAULT_MIN_GRADIENT
        low = _percentile_bin(self._bins, self._count, 0.5)
        deviations = np.zeros_like(self._bins)
        nonzero = np.nonzero(self._bins)[0]
        for bin_idx in nonzero.tolist():
            count = int(self._bins[bin_idx])
            slot = min(int(round(abs(float(bin_idx) - low))), deviations.size - 1)
            deviations[slot] += count
        mad = _percentile_bin(deviations, self._count, 0.5)
        sigma = 1.4826 * mad
        return max(NOISE_GRADIENT_SIGMAS * sigma, DEFAULT_MIN_GRADIENT)


def _percentile_bin(bins: np.ndarray, count: int, quantile: float) -> float:
    if count == 0:
        return 0.0
    target = int(round((count - 1) * quantile))
    seen = 0
    for bin_idx, bin_count in enumerate(bins.tolist()):
        seen += int(bin_count)
        if seen > target:
            return float(bin_idx)
    return 0.0


def _round4(v: float) -> float:
    return round(float(v), 4)


def detect_particles_frame(
    frame: np.ndarray,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
) -> list[ParticleSpot]:
    df = detect_spots_picasso(
        np.asarray(frame, dtype=np.float64),
        min_gradient=min_gradient,
        box_size=DEFAULT_BOX_SIZE,
        fit_method="lq",
        model="elliptic",
        baseline=0.0,
        sensitivity=1.0,
        gain=1.0,
        gaussian_sigma=None,
    )
    spots: list[ParticleSpot] = []
    for row in df.to_dict(orient="records"):
        sx = max(float(row["sx"]), 0.0)
        sy = max(float(row["sy"]), 0.0)
        radius = _round4(math.sqrt(sx * sy))
        spots.append(
            ParticleSpot(
                x=float(row["x"]),
                y=float(row["y"]),
                intensity=float(row["intens_A"]),
                radius=radius,
                photons=float(row["photons"]),
                sx=sx,
                sy=sy,
                fwhm=float(row["fwhm"]),
                background=float(row["intens_B"]),
                snb=float(row["snb"]),
                net_gradient=float(row["net_gradient"]),
            )
        )
    return spots


def summarize_particles(particles: list[ParticleSpot]) -> BindingFrameStats:
    if not particles:
        return BindingFrameStats()
    intensities = np.asarray([p.intensity for p in particles], dtype=np.float64)
    radii = np.asarray([p.radius for p in particles], dtype=np.float64)

    intensity_mean = _round4(float(np.mean(intensities)))
    intensity_median = _round4(float(np.percentile(intensities, 50.0)))
    intensity_iqr = _round4(
        float(np.percentile(intensities, 75.0) - np.percentile(intensities, 25.0))
    )

    radius_mean = _round4(float(np.mean(radii)))
    radius_median = _round4(float(np.percentile(radii, 50.0)))
    radius_iqr = _round4(
        float(np.percentile(radii, 75.0) - np.percentile(radii, 25.0))
    )

    return BindingFrameStats(
        particles=len(particles),
        intensity_mean=intensity_mean,
        intensity_median=intensity_median,
        intensity_iqr=intensity_iqr,
        radius_mean=radius_mean,
        radius_median=radius_median,
        radius_iqr=radius_iqr,
    )
