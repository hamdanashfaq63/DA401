"""Command line interface.

Examples:
    foldsym demo --out results/demo
    foldsym run data/STEMcroppedRAW.tif --config v1 --out results/v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import CONFIGS, get_config
from .pipeline import process_stack, summarize


def _write_outputs(results, out: Path, config_name: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "label", "n_speckles", "speckle_intensity"])
        for i, r in enumerate(results):
            w.writerow([i, r.label, len(r.speckles), r.speckle_intensity])
    summary = summarize(r.label for r in results)
    (out / "summary.json").write_text(json.dumps({"config": config_name, **summary}, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import plot_distribution

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_distribution(summary, ax, title=f"Fold symmetry distribution ({config_name})")
    fig.tight_layout()
    fig.savefig(out / "distribution.png", dpi=200)
    plt.close(fig)
    return summary


def _print_summary(summary: dict) -> None:
    print(f"\n{'label':<14}{'count':>10}{'percent':>10}")
    for lab, v in summary.items():
        if lab != "total":
            print(f"{lab:<14}{v['count']:>10,}{v['percent']:>9.2f}%")
    print(f"{'total':<14}{summary['total']:>10,}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="foldsym", description="Fold symmetry detection in 4D-STEM nanodiffraction patterns.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="classify every pattern in a (N, H, W) TIFF or .npy stack")
    run.add_argument("stack", type=Path)
    run.add_argument("--config", default="v1", choices=sorted(CONFIGS))
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--limit", type=int, default=None, help="only process the first N patterns")

    demo = sub.add_parser("demo", help="run on synthetic patterns with known symmetry")
    demo.add_argument("--config", default="v1", choices=sorted(CONFIGS))
    demo.add_argument("--per-fold", type=int, default=50)
    demo.add_argument("--out", type=Path, default=Path("results/demo"))

    args = p.parse_args(argv)
    cfg = get_config(args.config)

    if args.cmd == "run":
        if args.stack.suffix == ".npy":
            stack = np.load(args.stack, mmap_mode="r")
        else:
            import tifffile

            stack = tifffile.imread(args.stack)
        if args.limit:
            stack = stack[: args.limit]
        print(f"Loaded stack {stack.shape} from {args.stack}")
        results = process_stack(stack, cfg)
        summary = _write_outputs(results, args.out, cfg.name)
    else:
        from .synthetic import make_stack

        truth = np.repeat([0, 2, 3, 4, 5, 6], args.per_fold)
        results = process_stack(make_stack(truth, rng=0), cfg)
        summary = _write_outputs(results, args.out, cfg.name)
        expected = {0: "None", 2: "2-fold", 3: "3-fold (Odd)", 4: "4-fold", 5: "5-fold (Odd)", 6: "6-fold"}
        acc = np.mean([r.label == expected[t] for r, t in zip(results, truth)])
        print(f"\nSynthetic accuracy ({cfg.name}): {acc:.1%}")

    _print_summary(summary)
    print(f"\nWrote labels.csv, summary.json and distribution.png to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
