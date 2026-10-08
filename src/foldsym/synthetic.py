"""Synthetic nanodiffraction patterns with known fold symmetry.

The real CuZr metallic glass data is private, so these patterns let anyone run
the pipeline, reproduce the demo figures and test the classifier end to end.
They mimic the main features of an amorphous 4D-STEM pattern: a bright central
beam, a diffuse amorphous ring, shot-like noise, and (optionally) ``n`` bright
speckles on the ring that encode local n-fold order.
"""

from __future__ import annotations

import numpy as np


def make_pattern(
    n_fold: int = 0,
    size: int = 120,
    ring_radius: float = 34.0,
    speckle_sigma: float = 3.5,
    speckle_amplitude: float = 8.0,
    rotation_deg: float = 0.0,
    angle_jitter_deg: float = 0.0,
    noise: float = 0.15,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return a ``(size, size)`` float32 pattern with ``n_fold`` speckles (0 for none)."""
    rng = np.random.default_rng(rng)
    c = size // 2
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    r = np.hypot(yy - c, xx - c)

    img = 4.0 * np.exp(-(r**2) / (2 * 3.0**2))  # central beam
    img += 1.0 * np.exp(-((r - ring_radius) ** 2) / (2 * 6.0**2))  # amorphous ring
    img += 0.25 * np.exp(-r / 40.0)  # diffuse background

    for k in range(n_fold):
        theta = np.radians(rotation_deg + 360.0 * k / n_fold + rng.normal(0, angle_jitter_deg))
        sy, sx = c + ring_radius * np.sin(theta), c + ring_radius * np.cos(theta)
        img += speckle_amplitude * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * speckle_sigma**2))

    img *= rng.lognormal(0.0, noise, size=img.shape)
    return img.astype(np.float32)


def make_stack(folds, rng: np.random.Generator | int | None = 0, **kwargs) -> np.ndarray:
    """Stack of patterns, one per entry of ``folds`` (0 means no speckles)."""
    rng = np.random.default_rng(rng)
    return np.stack([make_pattern(n, rng=rng, **kwargs) for n in folds])
