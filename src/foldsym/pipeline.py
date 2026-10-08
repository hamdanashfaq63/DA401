"""End-to-end processing of a 4D-STEM nanodiffraction stack."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from skimage import draw

from .classify import classify_pattern
from .config import LABELS, V2, PipelineConfig
from .detect import detect_circular_speckles


@dataclass
class PatternResult:
    label: str
    speckles: np.ndarray  # (k, 3) rows of (y, x, radius)
    speckle_intensity: float  # mean intensity inside detected speckles, NaN if none


def speckle_mean_intensity(image: np.ndarray, speckles: np.ndarray) -> float:
    if len(speckles) == 0:
        return float("nan")
    mask = np.zeros(image.shape, dtype=bool)
    for y, x, r in speckles:
        rr, cc = draw.disk((int(y), int(x)), int(r), shape=image.shape)
        mask[rr, cc] = True
    return float(np.mean(image[mask])) if mask.any() else float("nan")


def process_image(image: np.ndarray, config: PipelineConfig = V2) -> PatternResult:
    """Detect speckles in one pattern and classify its fold symmetry."""
    speckles = detect_circular_speckles(image, config.detector)
    if config.classifier.max_speckles is not None:
        speckles = speckles[: config.classifier.max_speckles]
    label = classify_pattern(speckles, image.shape, config.classifier)
    return PatternResult(label, speckles, speckle_mean_intensity(image, speckles))


def process_stack(stack: Iterable[np.ndarray], config: PipelineConfig = V2, progress: bool = True) -> List[PatternResult]:
    """Run :func:`process_image` over every pattern in a ``(N, H, W)`` stack."""
    iterator = stack
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(stack, desc=f"foldsym[{config.name}]", unit="pattern")
    return [process_image(img, config) for img in iterator]


def summarize(labels: Iterable[str]) -> dict:
    """Count and percentage per label, over all patterns (none are dropped)."""
    labels = list(labels)
    counts = Counter(labels)
    total = len(labels)
    return {
        lab: {"count": counts.get(lab, 0), "percent": 100.0 * counts.get(lab, 0) / total if total else 0.0}
        for lab in LABELS
    } | {"total": total}
