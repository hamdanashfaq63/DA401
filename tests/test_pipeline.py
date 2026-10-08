import numpy as np
import pytest

from foldsym import LABELS, V1, V2, detect_circular_speckles, process_image, summarize
from foldsym.cli import main
from foldsym.synthetic import make_pattern, make_stack

EXPECTED = {2: "2-fold", 3: "3-fold (Odd)", 4: "4-fold", 5: "5-fold (Odd)", 6: "6-fold"}


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_v1_recovers_synthetic_folds(n):
    labels = [process_image(make_pattern(n, rng=s), V1).label for s in range(10)]
    assert labels.count(EXPECTED[n]) >= 9


@pytest.mark.parametrize("n", [4, 5, 6])
def test_v2_recovers_synthetic_folds(n):
    labels = [process_image(make_pattern(n, rng=s), V2).label for s in range(10)]
    assert labels.count(EXPECTED[n]) >= 9


def test_detector_finds_expected_number_of_speckles():
    speckles = detect_circular_speckles(make_pattern(4, rng=1), V1.detector)
    assert speckles.shape == (4, 3)


def test_blank_and_constant_images_do_not_crash():
    for img in (np.zeros((120, 120), np.float32), np.ones((120, 120), np.float32)):
        r = process_image(img, V2)
        assert r.label == "None"
        assert np.isnan(r.speckle_intensity)


def test_integer_images_are_supported():
    img = (make_pattern(6, rng=3) * 1000).astype(np.uint16)
    assert process_image(img, V1).label == "6-fold"


def test_summarize_counts_every_pattern():
    s = summarize(["2-fold", "None", "None", "Other"])
    assert s["total"] == 4
    assert s["None"]["count"] == 2
    assert set(LABELS) <= set(s)
    assert sum(s[lab]["count"] for lab in LABELS) == 4


def test_cli_run_and_demo(tmp_path):
    stack = make_stack([0, 4, 6], rng=0)
    np.save(tmp_path / "stack.npy", stack)
    assert main(["run", str(tmp_path / "stack.npy"), "--config", "v1", "--out", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "labels.csv").exists()
    assert main(["demo", "--per-fold", "2", "--out", str(tmp_path / "demo")]) == 0
    assert (tmp_path / "demo" / "distribution.png").exists()
