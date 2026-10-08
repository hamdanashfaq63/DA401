"""Regenerate the README figures that do not need the private dataset.

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldsym import V1, process_image
from foldsym.synthetic import make_pattern
from foldsym.viz import plot_distribution, show_pattern

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 100})

# Counts reported in the capstone paper (v1 config, 65,536 patterns).
# Patterns where no speckle was detected are counted as "None".
V1_REPORTED = {
    "2-fold": 3221,
    "3-fold (Odd)": 59,
    "4-fold": 91,
    "5-fold (Odd)": 0,
    "6-fold": 1,
    "None": 27662,
    "Other": 34502,
}


def synthetic_demo():
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.9))
    for ax, (n, seed) in zip(axes, [(0, 11), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]):
        img = make_pattern(n, rng=seed, rotation_deg=0, angle_jitter_deg=3)
        r = process_image(img, V1)
        show_pattern(ax, img, r.speckles if n else None, r.label if n else "no speckles")
        ax.set_xlabel(f"ground truth: {n}-fold" if n else "ground truth: none", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "synthetic_demo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def rotation_sensitivity(trials=40):
    angles = np.arange(0, 46, 2.5)
    expected = {2: "2-fold", 4: "4-fold", 6: "6-fold"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for n, color in zip(expected, ["#e4572e", "#2e86ab", "#29a36a"]):
        acc = []
        for a in angles:
            hits = sum(process_image(make_pattern(n, rotation_deg=a, rng=s), V1).label == expected[n] for s in range(trials))
            acc.append(hits / trials)
        ax.plot(angles, acc, "o-", color=color, label=f"{n}-fold", ms=4)
    ax.set_xlabel("rotation of speckle set relative to detector axes (degrees)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Rule-based classifier vs. pattern orientation (synthetic, v1)", loc="left", fontweight="bold", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "rotation_sensitivity.png", dpi=160)
    plt.close(fig)


def reported_distribution():
    total = sum(V1_REPORTED.values())
    summary = {k: {"count": v, "percent": 100 * v / total} for k, v in V1_REPORTED.items()}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    plot_distribution(summary, ax, title=f"Fold labels across {total:,} CuZr nanodiffraction patterns (v1)")
    fig.tight_layout()
    fig.savefig(ASSETS / "real_distribution.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    synthetic_demo()
    rotation_sensitivity()
    reported_distribution()
    print(f"Figures written to {ASSETS}")
