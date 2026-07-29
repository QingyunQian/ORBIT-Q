#!/usr/bin/env python3
"""Render the two five-pair Task 10 removal ablations."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "factor-ablation.svg"


def load(name):
    return json.loads((ROOT / "profiles" / name).read_text())


def paired_panel(ax, report, title, ablation_label):
    promoted = report["optimized_runtime_sec"]
    ablation = report["ablation_runtime_sec"]
    x = np.array([0, 1])
    for i, (control, removed) in enumerate(zip(promoted, ablation)):
        ax.plot(x, [control, removed], marker="o", linewidth=1.2, alpha=0.72)
        ax.text(1.03, removed, str(i + 1), va="center", fontsize=7)
    ax.set_xticks(x, ["Promoted", ablation_label])
    ax.set_ylabel("Canonical runtime (s)")
    control_mean = np.mean(promoted)
    ablation_mean = np.mean(ablation)
    delta = (ablation_mean / control_mean - 1) * 100
    ax.set_title(
        f"{title}\nmeans {control_mean:.3f}s → {ablation_mean:.3f}s ({delta:+.1f}%)",
        fontsize=10,
    )


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-10-factor-ablation"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    paired_panel(
        axes[0],
        load("ablation-no-scan-five-pair.json"),
        "Remove whole-training scan",
        "Python loop",
    )
    paired_panel(
        axes[1],
        load("ablation-unfused-rotations-five-pair.json"),
        "Remove local rotation fusion",
        "Unfused rotations",
    )
    axes[1].set_ylabel("")
    fig.suptitle(
        "Task 10 five-pair removal ablations — scan is neutral; fusion is secondary",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
