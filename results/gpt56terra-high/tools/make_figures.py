"""Generate the Terra/high outcome and agent-resource figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
SUMMARY = json.loads((ROOT / "summary.json").read_text())
COMPARISON = json.loads((ROOT / "model-comparison.json").read_text())
TASKS = np.arange(1, 13)

PASS = "#2A9D8F"
FAIL = "#D1495B"
BLUE = "#3B6FB6"
ORANGE = "#E6863B"
PURPLE = "#7B61A8"
GRAY = "#D9D9D9"
TEXT = "#222222"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
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
    fig = plt.figure(figsize=(13.2, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.65, 1.7])

    runs = COMPARISON["runs"]
    labels = [run["label"] for run in runs]
    colors = [BLUE, ORANGE, PURPLE]

    ax0 = fig.add_subplot(grid[0, 0])
    counts = [len(run["passes"]) for run in runs]
    bars = ax0.bar(np.arange(3), counts, color=colors, width=0.68)
    for bar, count in zip(bars, counts):
        ax0.text(bar.get_x() + bar.get_width() / 2, count + 0.2, f"{count}/12", ha="center", va="bottom", weight="bold")
    ax0.set_ylim(0, 12.8)
    ax0.set_yticks([0, 3, 6, 9, 12])
    ax0.set_xticks(np.arange(3), labels, rotation=25, ha="right")
    ax0.set_ylabel("Valid solutions")
    ax0.set_title("a  Overall validity", loc="left", weight="bold")
    ax0.grid(axis="y", alpha=0.22)

    ax1 = fig.add_subplot(grid[0, 1])
    matrix = np.array([[1 if task in run["passes"] else 0 for task in TASKS] for run in runs])
    ax1.imshow(matrix, aspect="auto", cmap=ListedColormap([FAIL, PASS]), vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax1.text(col, row, "P" if matrix[row, col] else "F", color="white", ha="center", va="center", weight="bold", fontsize=8)
    ax1.set_xticks(np.arange(12), [f"{task:02d}" for task in TASKS])
    ax1.set_yticks(np.arange(3), labels)
    ax1.tick_params(length=0)
    ax1.set_xlabel("Challenge task")
    ax1.set_title("b  Task-level outcome", loc="left", weight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2 = fig.add_subplot(grid[0, 2])
    rows = SUMMARY["tasks"]
    runtime = np.array([row["runtime_sec"] if row["runtime_sec"] >= 0 else np.nan for row in rows])
    passed = np.array([row["reward"] == 1 for row in rows])
    bars = ax2.bar(TASKS, np.nan_to_num(runtime), color=np.where(passed, PASS, FAIL), width=0.72)
    for task, value in zip(TASKS, runtime):
        if np.isnan(value):
            ax2.text(task, 3.0, "n/a", ha="center", va="bottom", rotation=90, fontsize=7, color="#666666")
    for task, value, ok in zip(TASKS, runtime, passed):
        if not ok and not np.isnan(value):
            ax2.text(task, value + 3.0, "F", ha="center", va="bottom", color=FAIL, weight="bold")
    ax2.set_xticks(TASKS, [f"{task:02d}" for task in TASKS])
    ax2.set_ylim(0, max(np.nanmax(runtime) * 1.18, 125))
    ax2.set_xlabel("Challenge task")
    ax2.set_ylabel("End-to-end solution runtime (s)")
    ax2.set_title("c  Terra/high artifact runtime", loc="left", weight="bold")
    ax2.grid(axis="y", alpha=0.22)
    ax2.text(0.02, 0.98, "Green: pass   Red: fail", transform=ax2.transAxes, va="top", fontsize=8, color=TEXT)

    save(fig, "gpt56terra-high-outcomes")


def resource_figure() -> None:
    rows = SUMMARY["tasks"]
    wall_min = np.array([(row["agent_wall_time_sec"] or 0) / 60 for row in rows])
    input_tokens = np.array([row["n_input_tokens"] or 0 for row in rows])
    cache_tokens = np.array([row["n_cache_tokens"] or 0 for row in rows])
    noncache_tokens = np.maximum(input_tokens - cache_tokens, 0)
    output_tokens = np.array([row["n_output_tokens"] or 0 for row in rows])
    costs = np.array([row["cost_usd"] or 0 for row in rows])
    passed = np.array([row["reward"] == 1 for row in rows])
    task_colors = np.where(passed, PASS, FAIL)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    ax0, ax1, ax2 = axes

    ax0.bar(TASKS, wall_min, color=task_colors, width=0.72)
    ax0.axhline(30, color="#777777", linestyle="--", linewidth=1)
    ax0.set_xticks(TASKS, [f"{task:02d}" for task in TASKS])
    ax0.set_xlabel("Challenge task")
    ax0.set_ylabel("Agent solve time (min)")
    ax0.set_title("a  Solver wall time", loc="left", weight="bold")
    ax0.grid(axis="y", alpha=0.22)

    scale = 1e6
    ax1.bar(TASKS, cache_tokens / scale, color="#9EC1E6", width=0.72, label="Cache-read input")
    ax1.bar(TASKS, noncache_tokens / scale, bottom=cache_tokens / scale, color=BLUE, width=0.72, label="Non-cache input")
    ax1.bar(TASKS, output_tokens / scale, bottom=input_tokens / scale, color=ORANGE, width=0.72, label="Output")
    ax1.set_xticks(TASKS, [f"{task:02d}" for task in TASKS])
    ax1.set_xlabel("Challenge task")
    ax1.set_ylabel("Tokens (millions)")
    ax1.set_title("b  Solver token use", loc="left", weight="bold")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    ax1.grid(axis="y", alpha=0.22)

    ax2.scatter(wall_min[passed], costs[passed], s=70, color=PASS, edgecolor="white", linewidth=0.7, label="Pass", zorder=3)
    ax2.scatter(wall_min[~passed], costs[~passed], s=70, color=FAIL, marker="X", edgecolor="white", linewidth=0.7, label="Fail", zorder=3)
    for task, x, y in zip(TASKS, wall_min, costs):
        ax2.annotate(f"{task:02d}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax2.set_xlabel("Agent solve time (min)")
    ax2.set_ylabel("Recorded solver cost (USD)")
    ax2.set_title("c  Time–cost profile", loc="left", weight="bold")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(alpha=0.22)

    save(fig, "gpt56terra-high-agent-resource-use")


if __name__ == "__main__":
    outcome_figure()
    resource_figure()
