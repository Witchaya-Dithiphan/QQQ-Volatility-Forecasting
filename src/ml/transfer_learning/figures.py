"""Report figures from saved artifacts only (nothing is refit).

    python -m src.ml.transfer_learning.figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.contracts import FEATURE_COLUMNS, VARIANTS
from ..core.data import load
from ..core.metrics import classification_metrics
from ..core.report import PERSISTENCE_FEATURE
from .experiment import RUNS, mlp_dir
from .sources import ROOT

FIGURES = ROOT / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRANSFORMER, GRU, BASELINE, ACCENT = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def color(name: str) -> str:
    return TRANSFORMER if name.startswith("T-") else GRU if name.startswith("G-") else BASELINE


def ap_dots(summary: dict, title: str, path) -> None:
    rows = sorted(summary, key=lambda r: r["average_precision"])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    style(ax)
    for i, row in enumerate(rows):
        ci = row["ap_ci"]
        ax.plot([ci["low"], ci["high"]], [i, i], color=color(row["model"]), linewidth=2, alpha=0.45,
                solid_capstyle="round")
        ax.plot(row["average_precision"], i, "o", markersize=8, color=color(row["model"]),
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(ci["high"] + 0.01, i, f"{row['average_precision']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows], color=INK)
    ax.set_xlabel("Average precision (95% moving-block bootstrap)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pretrain_curves(variant: str, path) -> None:
    """Shared SSL curve on the ETFs, then this variant's DAPT, both as masked MSE on QQQ Train windows."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE, sharey=True)
    for ax, prefix, hue, label in ((axes[0], "T", TRANSFORMER, "Transformer"), (axes[1], "G", GRU, "GRU")):
        style(ax)
        for seed_dir in sorted((RUNS / f"{prefix}-pretrain").glob("seed_*")):
            ssl = json.loads((seed_dir / "history.json").read_text(encoding="utf-8"))["ssl"]["monitor_loss"]
            dapt_path = RUNS / variant / f"{prefix}-dapt" / seed_dir.name / "history.json"
            dapt = json.loads(dapt_path.read_text(encoding="utf-8"))["dapt"]["monitor_loss"]
            first = seed_dir.name == "seed_0"
            ax.plot(range(1, len(ssl) + 1), ssl, color=hue, linewidth=2, alpha=0.8,
                    label="SSL on 10 ETFs" if first else None)
            ax.plot(range(len(ssl), len(ssl) + len(dapt) + 1), [ssl[-1], *dapt], color=ACCENT, linewidth=2,
                    alpha=0.8, label=f"DAPT on QQQ Train ({variant})" if first else None)
        ax.axvline(len(ssl) + 0.5, color=GRID, linewidth=1)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        ax.set_title(label, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    axes[0].set_ylabel("masked MSE on QQQ Train windows", color=MUTED, fontsize=9)
    # The shared SSL run logged its monitor on with_spike Train windows, whichever variant is plotted.
    fig.text(0.01, 0.01, f"Monitor: fixed 512 QQQ Train windows (SSL: with_spike; DAPT: {variant}). 3 seeds per backbone.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pr_curves(curves: dict, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4), facecolor=SURFACE)
    style(ax)
    hues = {"T-full": TRANSFORMER, "T-scratch": ACCENT, "mlp": GRU, "baseline_persistence": BASELINE}
    for name, (y, score) in curves.items():
        pr = classification_metrics(y, score)["pr_curve"]["value"]
        ap = classification_metrics(y, score)["average_precision"]["value"]
        ax.step(pr["recall"], pr["precision"], where="post", color=hues[name], linewidth=2, label=f"{name} (AP {ap:.3f})")
    ax.axhline(np.mean(y), color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("recall", color=MUTED, fontsize=9)
    ax.set_ylabel("precision", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mean_scores(variant: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    frames = [pd.read_csv(d / "validation_predictions.csv") for d in sorted((RUNS / variant / arm).glob("seed_*"))]
    return frames[0]["target"].to_numpy(), np.mean([f["raw_score"] for f in frames], axis=0)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        summary = json.loads((ROOT / variant / "validation_summary.json").read_text(encoding="utf-8"))
        ap_dots(summary["leaderboard"], f"Validation AP by arm, {variant} (seed-averaged)",
                FIGURES / f"{variant}_validation_ap.png")
        pretrain_curves(variant, FIGURES / f"{variant}_pretraining_loss.png")
        y, full = mean_scores(variant, "T-full")
        _, scratch = mean_scores(variant, "T-scratch")
        mlp = pd.read_csv(mlp_dir(variant) / "validation_predictions.csv")
        persistence = load(variant).validation.X[:, FEATURE_COLUMNS.index(PERSISTENCE_FEATURE)]
        curves = {"T-full": (y, full), "T-scratch": (y, scratch), "mlp": (y, mlp["raw_score"].to_numpy()),
                  "baseline_persistence": (y, persistence)}
        pr_curves(curves, f"Validation precision-recall, {variant}", FIGURES / f"{variant}_validation_pr.png")
        test_dir = ROOT / "test" / variant
        if (test_dir / "test_metrics.json").is_file():
            metrics = json.loads((test_dir / "test_metrics.json").read_text(encoding="utf-8"))
            rows = [{"model": m, "average_precision": e["point"]["average_precision"], "ap_ci": e["ap_ci"]}
                    for m, e in metrics["models"].items()]
            ap_dots(rows, f"Test AP, {variant} (post-hoc; Test was opened before)", FIGURES / f"{variant}_test_ap.png")
            curves = {}
            for name in ("T-full", "T-scratch", "mlp", "baseline_persistence"):
                frame = pd.read_csv(test_dir / f"{name}_test_predictions.csv")
                curves[name] = (frame["target"].to_numpy(), frame["raw_score"].to_numpy())
            pr_curves(curves, f"Test precision-recall, {variant} (post-hoc)", FIGURES / f"{variant}_test_pr.png")


if __name__ == "__main__":
    main()
