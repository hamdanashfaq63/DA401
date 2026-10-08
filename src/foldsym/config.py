"""Pipeline configurations.

Two configurations are provided, each a faithful port of a version of the
pipeline in ``notebooks/01_capstone_exploration.ipynb``:

* ``V1`` is the configuration behind the results reported in the capstone
  paper (notebook cell 6): 95th percentile threshold, speckle radius 4 to 15 px,
  exact speckle count per fold, 15% angular tolerance for every fold.
* ``V2`` is the later, tuned configuration (notebook cells 7 to 9): 92nd
  percentile threshold, speckle radius 6 to 10 px, at least ``n`` speckles per
  fold, 25% tolerance for 4- and 6-fold, at most 8 speckles considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Ideal angular spacing (degrees) for each fold order.
FOLD_SPACING_DEG: Dict[int, float] = {n: 360.0 / n for n in (2, 3, 4, 5, 6)}

#: Every label the classifier can emit, in display order.
LABELS: Tuple[str, ...] = (
    "2-fold",
    "3-fold (Odd)",
    "4-fold",
    "5-fold (Odd)",
    "6-fold",
    "None",
    "Other",
)


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters for :func:`foldsym.detect.detect_circular_speckles`."""

    min_radius: float = 6.0
    max_radius: float = 10.0
    center_mask_radius: int = 8
    threshold_percentile: float = 92.0
    contrast_percentiles: Tuple[float, float] = (2.0, 98.0)
    size_consistency: Tuple[float, float] = (0.5, 1.5)


@dataclass(frozen=True)
class ClassifierConfig:
    """Parameters for :func:`foldsym.classify.classify_pattern`."""

    fold_order: Tuple[int, ...] = (4, 6, 2, 3, 5)
    exact_count: bool = False
    tolerance: Dict[int, float] = field(
        default_factory=lambda: {2: 0.15, 3: 0.15, 4: 0.25, 5: 0.15, 6: 0.25}
    )
    max_speckles: Optional[int] = 8


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    detector: DetectorConfig
    classifier: ClassifierConfig


V1 = PipelineConfig(
    name="v1",
    detector=DetectorConfig(min_radius=4.0, max_radius=15.0, threshold_percentile=95.0),
    classifier=ClassifierConfig(
        fold_order=(2, 4, 6, 3, 5),
        exact_count=True,
        tolerance={n: 0.15 for n in (2, 3, 4, 5, 6)},
        max_speckles=None,
    ),
)

V2 = PipelineConfig(
    name="v2",
    detector=DetectorConfig(),
    classifier=ClassifierConfig(),
)

CONFIGS: Dict[str, PipelineConfig] = {"v1": V1, "v2": V2}


def get_config(name: str) -> PipelineConfig:
    try:
        return CONFIGS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown config {name!r}; choose from {sorted(CONFIGS)}") from exc
