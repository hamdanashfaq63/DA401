import numpy as np
import pytest

from foldsym import V1, V2, classify_pattern

SHAPE = (120, 120)


def ring(angles_deg, radius=34.0, r=7.0):
    c = SHAPE[0] // 2
    a = np.radians(angles_deg)
    return np.column_stack([c + radius * np.sin(a), c + radius * np.cos(a), np.full(len(a), r)])


@pytest.mark.parametrize(
    "angles, expected",
    [
        ([0, 180], "2-fold"),
        ([0, 120, 240], "3-fold (Odd)"),
        ([0, 90, 180, 270], "4-fold"),
        ([0, 72, 144, 216, 288], "5-fold (Odd)"),
        ([0, 60, 120, 180, 240, 300], "6-fold"),
    ],
)
@pytest.mark.parametrize("config", [V1, V2], ids=["v1", "v2"])
def test_ideal_geometry(angles, expected, config):
    assert classify_pattern(ring(angles), SHAPE, config.classifier) == expected


@pytest.mark.parametrize("config", [V1, V2], ids=["v1", "v2"])
def test_fewer_than_two_speckles_is_none(config):
    assert classify_pattern(np.empty((0, 3)), SHAPE, config.classifier) == "None"
    assert classify_pattern(ring([45]), SHAPE, config.classifier) == "None"


def test_irregular_geometry_is_other():
    assert classify_pattern(ring([10, 37, 200]), SHAPE, V1.classifier) == "Other"


def test_small_angular_noise_is_tolerated():
    assert classify_pattern(ring([4, 93, 178, 268]), SHAPE, V1.classifier) == "4-fold"


def test_v1_requires_exact_count_v2_does_not():
    extra = ring([0, 90, 180, 270, 300])
    assert classify_pattern(extra, SHAPE, V1.classifier) == "Other"
    assert classify_pattern(extra, SHAPE, V2.classifier) == "4-fold"


def test_known_limitation_fixed_reference_orientation():
    """Ideal angles start at 0 degrees, so a 4-fold pattern rotated 45 degrees is missed."""
    assert classify_pattern(ring([45, 135, 225, 315]), SHAPE, V1.classifier) != "4-fold"
