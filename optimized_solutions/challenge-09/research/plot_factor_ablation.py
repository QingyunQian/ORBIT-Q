#!/usr/bin/env python3
"""Render Task 09 structural and follow-up factor screens."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("figures") / "factor-ablation.svg"


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-09-factor-ablation"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    blue, orange, red, gray = "#4472C4", "#ED7D31", "#C44E52", "#A5A5A5"

    gates = [3897, 74, 80]
    bars = axes[0].bar(
        ["Expert graph", "Cone group A", "Cone group B"],
        gates,
        color=[red, blue, blue],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Circuit gates (log scale)")
    axes[0].set_title("Explicit causal-cone reduction")
    axes[0].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, gates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    labels = ["Pre-resolved\nmethods", "Combined\nloss", "Threaded\ngroups", "Manual gate\nfusion"]
    relative = [0.9385, 7.139375 / 7.721969, 7.447021 / 6.900286, 1.019713]
    x = np.arange(len(labels))
    bars = axes[1].bar(x, relative, color=[red, red, orange, gray])
    axes[1].axhline(1, color="black", linewidth=0.9)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.85, 1.12)
    axes[1].set_ylabel("Performance relative to parent")
    axes[1].set_title("Follow-up screens (>1 faster)")
    for bar, value in zip(bars, relative):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    timeout = axes[2].bar(
        ["Compact candidate", "Disable inner\nlight-cone pass"],
        [8.767, 300],
        color=[blue, red],
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Runtime (s, log scale)")
    axes[2].set_title("Framework light-cone ablation")
    axes[2].text(
        timeout[0].get_x() + timeout[0].get_width() / 2,
        9.5,
        "8.77s",
        ha="center",
        fontsize=8,
    )
    axes[2].text(
        timeout[1].get_x() + timeout[1].get_width() / 2,
        245,
        ">300s timeout",
        ha="center",
        va="top",
        fontsize=8,
        color="white",
    )

    fig.suptitle(
        "Task 09 attribution — compact cones dominate; packing and scan remain coupled",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
