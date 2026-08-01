"""Generate Luna/high outcome and agent-resource figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
SUMMARY = json.loads((ROOT / "summary.json").read_text())
COMPARISON = json.loads((ROOT / "model-comparison.json").read_text())
TASKS = np.arange(1, 13)

GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
LIGHT_BLUE = "#56B4E9"
GRAY = "#555555"
PALE_PASS = "#D9ECE7"
TEXT = "#222222"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 240,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGS / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def outcome_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.0), constrained_layout=True)
    runs = COMPARISON["runs"]
    labels = [run["label"] for run in runs]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    matrix = np.array(
        [[1 if task in run["passes"] else 0 for task in TASKS] for run in runs]
    )
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=ListedColormap(["#FFFFFF", PALE_PASS]),
        vmin=0,
        vmax=1,
    )
    for row, color in enumerate(colors):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                "P" if matrix[row, col] else "F",
                color=color if matrix[row, col] else TEXT,
                ha="center",
                va="center",
                weight="bold",
                fontsize=10,
            )
    ax.set_xticks(np.arange(12), [f"{task:02d}" for task in TASKS])
    ax.set_yticks(np.arange(4), labels)
    ax.tick_params(length=0)
    ax.set_xlabel("Challenge")
    ax.set_title(
        "GPT-5.6 TensorCircuit solver comparison: final task outcomes",
        fontsize=14,
        weight="bold",
        pad=14,
    )
    ax.text(
        0,
        -0.20,
        "P = valid solution; F = failed task. One valid model outcome per challenge.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    save(fig, "gpt56luna-high-outcomes")


def resource_figure() -> None:
    rows = SUMMARY["tasks"]
    wall_min = np.array([(row["agent_wall_time_sec"] or 0) / 60 for row in rows])
    input_tokens = np.array([row["n_input_tokens"] or 0 for row in rows])
    cache_tokens = np.array([row["n_cache_tokens"] or 0 for row in rows])
    noncache_tokens = np.maximum(input_tokens - cache_tokens, 0)
    output_tokens = np.array([row["n_output_tokens"] or 0 for row in rows])
    costs = np.array([row["cost_usd"] or 0 for row in rows])
    passed = np.array([row["reward"] == 1 for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    ax0, ax1, ax2 = axes

    ax0.bar(
        TASKS[passed],
        wall_min[passed],
        color=GREEN,
        edgecolor="black",
        linewidth=0.8,
        width=0.72,
    )
    ax0.bar(
        TASKS[~passed],
        wall_min[~passed],
        color="white",
        edgecolor="black",
        linewidth=0.8,
        width=0.72,
    )
    ax0.set_xticks(TASKS, [f"{task:02d}" for task in TASKS])
    ax0.set_xlabel("Challenge")
    ax0.set_ylabel("Agent solve time (min)")
    ax0.set_title("Solver wall time", loc="left")
    ax0.text(-0.14, 1.08, "(a)", transform=ax0.transAxes, fontsize=15, weight="bold")
    ax0.text(0.02, 0.96, f"Total: {wall_min.sum():.1f} min", transform=ax0.transAxes, va="top")
    ax0.legend(
        handles=[
            Patch(facecolor=GREEN, edgecolor="black", label="Passed"),
            Patch(facecolor="white", edgecolor="black", label="Failed"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    ax0.grid(axis="y", alpha=0.35)

    scale = 1e6
    ax1.bar(
        TASKS,
        noncache_tokens / scale,
        color=GRAY,
        edgecolor="black",
        linewidth=0.5,
        width=0.72,
        label="Non-cache-read prompt",
    )
    ax1.bar(
        TASKS,
        cache_tokens / scale,
        bottom=noncache_tokens / scale,
        color=LIGHT_BLUE,
        edgecolor="black",
        linewidth=0.5,
        width=0.72,
        label="Cache-read prompt",
    )
    ax1.bar(
        TASKS,
        output_tokens / scale,
        bottom=input_tokens / scale,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.5,
        width=0.72,
        label="Output",
    )
    ax1.set_xticks(TASKS, [f"{task:02d}" for task in TASKS])
    ax1.set_xlabel("Challenge")
    ax1.set_ylabel("Solving-side tokens (million)")
    ax1.set_title("Solver token use", loc="left")
    ax1.text(-0.14, 1.08, "(b)", transform=ax1.transAxes, fontsize=15, weight="bold")
    ax1.text(
        0.02,
        0.96,
        f"Total: {(input_tokens.sum() + output_tokens.sum()) / scale:.3f}M",
        transform=ax1.transAxes,
        va="top",
    )
    ax1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.35, 1.0))
    ax1.grid(axis="y", alpha=0.35)

    names = ["Sol high", "Sol ultra", "Terra high", "Luna high"]
    valid = np.array([10, 11, 9, 10])
    solve_per_valid = np.array([19.77, 16.62, 12385.907 / 60 / 9, wall_min.sum() / 10])
    cost_per_valid = np.array([2.55, 2.73, 12.036862 / 9, costs.sum() / 10])
    total_cost = np.array([25.527418, 30.02, 12.036862, costs.sum()])
    bubble_colors = [BLUE, ORANGE, GREEN, PURPLE]
    ax2.scatter(
        solve_per_valid,
        cost_per_valid,
        s=75 * total_cost,
        color=bubble_colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )
    offsets = [(28, -26), (28, 10), (-92, 10), (-72, 12)]
    for name, count, xval, yval, color, offset in zip(
        names, valid, solve_per_valid, cost_per_valid, bubble_colors, offsets
    ):
        ax2.annotate(
            f"{name}\n({count}/12)",
            (xval, yval),
            xytext=offset,
            textcoords="offset points",
            color=color,
            weight="bold",
            fontsize=9,
        )
    ax2.set_xlim(15.5, 25.8)
    ax2.set_ylim(0.0, 3.05)
    ax2.set_xlabel("Solve time per valid solution (min)")
    ax2.set_ylabel("Recorded cost per valid solution (USD)")
    ax2.set_title("Configuration-level resource efficiency", loc="left")
    ax2.text(-0.14, 1.08, "(c)", transform=ax2.transAxes, fontsize=15, weight="bold")
    ax2.text(0.98, 0.04, "Marker area proportional to total recorded cost", transform=ax2.transAxes, ha="right", fontsize=8)
    ax2.grid(alpha=0.35)

    fig.suptitle("GPT-5.6 Luna high agent-side resource use", fontsize=14, weight="bold")
    save(fig, "gpt56luna-high-agent-resource-use")


if __name__ == "__main__":
    outcome_figure()
    resource_figure()
