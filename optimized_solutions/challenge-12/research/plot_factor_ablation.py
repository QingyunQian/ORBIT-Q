#!/usr/bin/env python3
"""Render Task 12 factor-ablation panels from tracked measurements."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "factor-ablation.svg"


def panel(ax, title, ratio, control, ablation):
    bars = ax.bar(
        [f"Promoted\n{control}", f"Removal\n{ablation}"],
        [1, ratio],
        color=["#4472C4", "#C44E52"],
    )
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_ylabel("Runtime normalized to promoted")
    ax.set_title(title)
    for bar, value in zip(bars, [1, ratio]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    scan = json.loads((ROOT / "profiles" / "factor-ablation.json").read_text())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-12-factor-ablation"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    panel(
        axes[0],
        "Fixed Pade → adaptive expm",
        0.3808 / 0.1225,
        "0.1225 ms",
        "0.3808 ms",
    )
    cold = scan["cold_profile_total_sec"]
    panel(
        axes[1],
        "Scan → Python dispatch",
        cold["python_loop"] / cold["scan"],
        f'{cold["scan"]:.3f} s cold',
        f'{cold["python_loop"]:.3f} s cold',
    )
    panel(
        axes[2],
        "Pair-fused → unfused network",
        2.3206 / 2.1237,
        "2.124 s",
        "2.321 s",
    )
    fig.suptitle(
        "Task 12 factor ablations — fixed Pade dominates; scan and pair fusion are secondary",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
