"""Rule-based fold symmetry classification from detected speckles."""

from __future__ import annotations

import numpy as np

from .config import ClassifierConfig
from .detect import center_of


def speckle_angles(speckles: np.ndarray, image_shape) -> np.ndarray:
    """Azimuthal angle (degrees, 0 to 360, sorted) of each speckle about the center."""
    cy, cx = center_of(image_shape)
    angles = np.degrees(np.arctan2(speckles[:, 0] - cy, speckles[:, 1] - cx)) % 360
    return np.sort(angles)


def classify_pattern(speckles: np.ndarray, image_shape, config: ClassifierConfig | None = None) -> str:
    """Assign a fold label by comparing speckle angles with ideal n-fold geometry.

    For each candidate fold ``n`` (in ``config.fold_order``), the sorted speckle
    angles are compared with the ideal angles ``0, 360/n, 2*360/n, ...``. A fold
    is accepted when every angle is within ``tolerance[n] * 360/n`` of an ideal
    angle and, for even ``n``, speckles form opposite (180 degree) pairs.

    Returns one of ``"None"`` (fewer than 2 speckles), ``"n-fold"``,
    ``"n-fold (Odd)"`` or ``"Other"``.

    Note: the ideal angles start at 0 degrees, so the rule assumes speckles are
    aligned with the detector axes. See the README section on limitations.
    """
    cfg = config or ClassifierConfig()
    speckles = np.asarray(speckles, dtype=float).reshape(-1, 3)
    if len(speckles) < 2:
        return "None"

    angles = speckle_angles(speckles, image_shape)

    for n in cfg.fold_order:
        if cfg.exact_count and len(speckles) != n:
            continue
        if not cfg.exact_count and len(speckles) < n:
            continue

        ideal = np.linspace(0, 360, n, endpoint=False)
        diff = np.min(np.abs((angles[:, None] - ideal + 180) % 360 - 180), axis=1)
        tol = cfg.tolerance[n]

        if np.all(diff[:n] <= tol * (360 / n)):
            if n % 2 == 0:
                half = n // 2
                opposite = np.abs(angles[:half] - angles[half:n] + 180) % 360 <= tol * 180
                if np.all(opposite):
                    return f"{n}-fold"
            else:
                return f"{n}-fold (Odd)"

    return "Other"
