#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "agent-resource-use.json"
OUTPUT_PATH = ROOT / "figs" / "gpt56sol-high-agent-resource-use.png"

PASS_COLOR = "#009E73"
FAIL_COLOR = "#FFFFFF"
NON_CACHE_COLOR = "#595959"
CACHE_COLOR = "#56B4E9"
OUTPUT_COLOR = "#D55E00"
BASELINE_COLOR = "#0072B2"
CHECKLIST_COLOR = "#56B4E9"
GPT56_COLOR = "#009E73"


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.15,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
    )


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    run = data["gpt56_sol_high"]
    tasks = run["tasks"]
    labels = [f"{row['task']:02d}" for row in tasks]
    x = np.arange(len(tasks))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    wall_minutes = np.array([row["agent_wall_sec"] / 60 for row in tasks])
    wall_colors = [PASS_COLOR if row["passed"] else FAIL_COLOR for row in tasks]
    axes[0].bar(x, wall_minutes, color=wall_colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Agent solve wall time (min)")
    axes[0].set_xlabel("Challenge")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(wall_minutes) * 1.13)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(
        handles=[
            Patch(facecolor=PASS_COLOR, edgecolor="black", label="Passed"),
            Patch(facecolor=FAIL_COLOR, edgecolor="black", label="Failed"),
        ],
        loc="upper right",
        frameon=False,
    )
    axes[0].text(
        0.02,
        0.97,
        (
            f"Total: {int(run['agent_wall_sec'] // 3600)}h "
            f"{int(run['agent_wall_sec'] % 3600 // 60)}m "
            f"{run['agent_wall_sec'] % 60:.1f}s"
        ),
        transform=axes[0].transAxes,
        va="top",
    )
    panel_label(axes[0], "(a)")

    non_cache = np.array([row["non_cache_input_tokens"] for row in tasks]) / 1e6
    cache = np.array([row["cache_tokens"] for row in tasks]) / 1e6
    output = np.array([row["output_tokens"] for row in tasks]) / 1e6
    axes[1].bar(
        x,
        non_cache,
        color=NON_CACHE_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Non-cache-read prompt",
    )
    axes[1].bar(
        x,
        cache,
        bottom=non_cache,
        color=CACHE_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Cache-read prompt",
    )
    axes[1].bar(
        x,
        output,
        bottom=non_cache + cache,
        color=OUTPUT_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Output",
    )
    axes[1].set_ylabel("Solving-side tokens (million)")
    axes[1].set_xlabel("Challenge")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(non_cache + cache + output) * 1.13)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].text(
        0.02,
        0.97,
        f"Total: {run['total_tokens'] / 1e6:.3f}M",
        transform=axes[1].transAxes,
        va="top",
    )
    panel_label(axes[1], "(b)")

    configs = data["paper_baselines"] + [
        {
            "label": run["label"],
            "passes": run["passes"],
            "solve_time_per_valid_solution_min": run[
                "solve_time_per_valid_solution_min"
            ],
            "cost_per_valid_solution_usd": run["cost_per_valid_solution_usd"],
            "cost_usd": run["cost_usd"],
        }
    ]
    colors = [BASELINE_COLOR, CHECKLIST_COLOR, GPT56_COLOR]
    offsets = [(0.35, -0.16), (0.35, 0.10), (-7.4, 0.08)]
    for item, color, (dx, dy) in zip(configs, colors, offsets, strict=True):
        solve_min = item["solve_time_per_valid_solution_min"]
        cost_per_valid = item["cost_per_valid_solution_usd"]
        axes[2].scatter(
            solve_min,
            cost_per_valid,
            s=item["cost_usd"] * 48,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        axes[2].annotate(
            f"{item['label']}\n({item['passes']}/12)",
            (solve_min, cost_per_valid),
            xytext=(solve_min + dx, cost_per_valid + dy),
            fontsize=10,
            fontweight="bold",
            color=color,
        )
    axes[2].set_xlabel("Solve time per valid solution (min)")
    axes[2].set_ylabel("Cost per valid solution (USD)")
    axes[2].set_xlim(6.5, 25.0)
    axes[2].set_ylim(1.05, 3.15)
    axes[2].grid(color="#DDDDDD", linewidth=0.8)
    axes[2].set_axisbelow(True)
    axes[2].text(
        0.98,
        0.03,
        "Marker area ∝ total cost",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    panel_label(axes[2], "(c)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)

    fig.suptitle(
        "GPT-5.6 Sol high agent-side resource use (adjudicated)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(w_pad=2.5)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
