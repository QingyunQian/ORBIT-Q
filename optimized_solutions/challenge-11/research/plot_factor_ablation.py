#!/usr/bin/env python3
"""Render Task 11 component-ablation panels from the tracked profile."""

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
    data = json.loads((ROOT / "profiles" / "factor-ablation.json").read_text())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-11-factor-ablation"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    panel(
        axes[0],
        "Fixed Pade → adaptive expm",
        data["entangler"]["adaptive_over_fixed_execution"],
        "44.78 µs",
        "75.91 µs",
    )
    panel(
        axes[1],
        "Diagonal onsite → expectations",
        data["onsite"]["expectations_over_diagonal_execution"],
        "0.228 ms",
        "18.513 ms",
    )
    axes[1].set_yscale("log")
    panel(
        axes[2],
        "Scan → Python dispatch",
        data["training_control_flow"]["loop_over_scan_execution"],
        "1.839 s / 10 steps",
        "2.130 s / 10 steps",
    )
    fig.suptitle(
        "Task 11 component ablations — isolated ratios are explanatory, not multiplicative",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
