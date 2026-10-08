"""Bright speckle detection in a single nanodiffraction pattern."""

from __future__ import annotations

import numpy as np
from skimage import exposure, measure

from .config import DetectorConfig


def center_of(shape) -> tuple[int, int]:
    """Pixel used as the pattern center (the unscattered beam position)."""
    return shape[0] // 2, shape[1] // 2


def detect_circular_speckles(image: np.ndarray, config: DetectorConfig | None = None) -> np.ndarray:
    """Find bright, roughly circular speckles outside the central beam.

    Steps:
      1. Zero out a disk of ``center_mask_radius`` around the center beam.
      2. Stretch contrast between the 2nd and 98th percentile of nonzero pixels.
      3. Threshold at ``threshold_percentile`` of nonzero pixels.
      4. Label connected components and keep those whose equivalent radius
         lies in ``[min_radius, max_radius]`` and whose centroid is outside the mask.
      5. Drop speckles outside 0.5x to 1.5x of the median speckle radius.

    Returns:
        ``(k, 3)`` array of ``(y, x, radius)`` rows, or an empty ``(0, 3)`` array.
    """
    cfg = config or DetectorConfig()
    image = np.asarray(image)
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64)
    cy, cx = center_of(image.shape)
    rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
    beam = (rr - cy) ** 2 + (cc - cx) ** 2 <= cfg.center_mask_radius**2

    masked = image.copy()
    masked[beam] = 0
    positive = masked[masked > 0]
    if positive.size == 0:
        return np.empty((0, 3))

    lo, hi = np.percentile(positive, cfg.contrast_percentiles)
    if hi <= lo:
        return np.empty((0, 3))
    masked = exposure.rescale_intensity(masked, in_range=(lo, hi))

    positive = masked[masked > 0]
    if positive.size == 0:
        return np.empty((0, 3))
    binary = masked > np.percentile(positive, cfg.threshold_percentile)

    speckles = []
    for prop in measure.regionprops(measure.label(binary)):
        if cfg.min_radius**2 <= prop.area / np.pi <= cfg.max_radius**2:
            y, x = prop.centroid
            if (y - cy) ** 2 + (x - cx) ** 2 > cfg.center_mask_radius**2:
                speckles.append((y, x, np.sqrt(prop.area / np.pi)))

    if speckles:
        median_r = np.median([s[2] for s in speckles])
        lo_f, hi_f = cfg.size_consistency
        speckles = [s for s in speckles if lo_f * median_r <= s[2] <= hi_f * median_r]

    return np.array(speckles, dtype=float).reshape(-1, 3)
