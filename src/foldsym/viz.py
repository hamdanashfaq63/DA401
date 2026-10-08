"""Plotting helpers for patterns, detections and label distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .config import LABELS
from .detect import center_of

LABEL_COLORS = {
    "2-fold": "#e4572e",
    "3-fold (Odd)": "#f3a712",
    "4-fold": "#2e86ab",
    "5-fold (Odd)": "#8e44ad",
    "6-fold": "#29a36a",
    "None": "#9aa0a6",
    "Other": "#5f6368",
}


def show_pattern(ax, image, speckles=None, label=None, center_mask_radius=8, log=True):
    """Display one pattern with detected speckles circled."""
    img = np.asarray(image, dtype=float).copy()
    cy, cx = center_of(img.shape)
    rr, cc = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(rr - cy) ** 2 + (cc - cx) ** 2 <= center_mask_radius**2] = np.nan
    if log:
        img = np.log1p(np.clip(img, 0, None))
    ax.imshow(img, cmap="magma", interpolation="nearest")
    if speckles is not None:
        for y, x, r in np.asarray(speckles).reshape(-1, 3):
            ax.add_patch(Circle((x, y), r + 1.5, fill=False, lw=1.6, ec="#7fffd4"))
    if label is not None:
        ax.set_title(label, fontsize=11, color=LABEL_COLORS.get(label, "black"), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_distribution(summary: dict, ax=None, title=None, log=True):
    """Horizontal bar chart of label counts from :func:`foldsym.pipeline.summarize`."""
    ax = ax or plt.gca()
    labels = [lab for lab in LABELS if lab in summary]
    counts = [summary[lab]["count"] for lab in labels]
    y = np.arange(len(labels))
    shown = [c if c > 0 else np.nan for c in counts] if log else counts
    ax.barh(y, shown, color=[LABEL_COLORS[lab] for lab in labels])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        ax.set_xlim(left=0.7)
    for yi, lab, c in zip(y, labels, counts):
        ax.text(max(c, 1) * (1.15 if log else 1.01), yi, f"{c:,}  ({summary[lab]['percent']:.2f}%)", va="center", fontsize=9)
    ax.set_xlabel("patterns" + (" (log scale)" if log else ""))
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return ax
