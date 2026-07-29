#!/usr/bin/env python3
"""Render the Task 06 factor-ablation summary from frozen campaign data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).with_name("figures") / "factor-ablation.svg"


def label_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-06-factor-ablation"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    blue, orange, red = "#4472C4", "#ED7D31", "#C44E52"

    labels = ["Expert", "+ Euler fusion", "+ dt0=None", "+ jaxode"]
    runtimes = [45.037164, 42.412637, 42.361953, 27.747994]
    bars = axes[0].bar(labels, runtimes, color=[blue, orange, "#A5A5A5", blue])
    label_bars(axes[0], bars, "{:.1f}s")
    axes[0].set_ylabel("Canonical runtime (s)")
    axes[0].set_title("Cumulative screens (jaxode: −34.6% vs fused)")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].set_ylim(0, 51)

    cold = [2.5819, 2.2421]
    bars = axes[1].bar(["Unfused", "Euler fused"], cold, color=[red, blue])
    label_bars(axes[1], bars, "{:.3f}s")
    axes[1].set_ylabel("Compile + first execution (s)")
    axes[1].set_title("Euler fusion (cold −13.2%)")
    axes[1].set_ylim(0, 3.05)
    axes[1].text(
        0.5,
        0.82,
        "steady execution: 1.0042×",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
    )

    slowdowns = [1 / 0.293381, 1 / 0.284729]
    bars = axes[2].bar(["Analog action", "Target action"], slowdowns, color=red)
    label_bars(axes[2], bars, "{:.2f}×")
    axes[2].axhline(1, color="black", linewidth=0.8)
    axes[2].set_ylabel("BCOO / termwise runtime")
    axes[2].set_title("Sparse BCOO screen (rejected)")
    axes[2].set_ylim(0, 4.05)
    axes[2].text(
        0.5,
        0.82,
        ">1 is slower; rejected",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
    )

    fig.suptitle(
        "Task 06 factor attribution — jaxode dominates; other factors are small or negative",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
