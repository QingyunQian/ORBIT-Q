#!/usr/bin/env python3
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
RESOURCE = json.loads((ROOT / "agent-resource-use.json").read_text())
COMPARISON = json.loads((ROOT / "effort-comparison.json").read_text())

HIGH_COLOR = "#0072B2"
ULTRA_COLOR = "#D55E00"
PASS_COLOR = "#009E73"
FAIL_COLOR = "#FFFFFF"
NON_CACHE_COLOR = "#595959"
CACHE_COLOR = "#56B4E9"
OUTPUT_COLOR = "#D55E00"
GRID_COLOR = "#DDDDDD"


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.14,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(exist_ok=True)
    target = FIGS / name
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(FIGS / name)


def ultra_resource_figure() -> None:
    run = RESOURCE["gpt56_sol_ultra"]
    tasks = run["tasks"]
    x = np.arange(12)
    labels = [f"{row['task']:02d}" for row in tasks]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    wall = np.array([row["agent_wall_sec"] / 60 for row in tasks])
    colors = [PASS_COLOR if row["passed"] else FAIL_COLOR for row in tasks]
    axes[0].bar(x, wall, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Agent solve wall time (min)")
    axes[0].set_xlabel("Challenge")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, wall.max() * 1.14)
    axes[0].legend(
        handles=[
            Patch(facecolor=PASS_COLOR, edgecolor="black", label="Passed"),
            Patch(facecolor=FAIL_COLOR, edgecolor="black", label="Failed"),
        ],
        frameon=False,
        loc="upper right",
    )
    axes[0].text(
        0.02,
        0.97,
        f"Total: {run['agent_wall_sec'] / 60:.1f} min",
        transform=axes[0].transAxes,
        va="top",
    )
    finish_axes(axes[0])
    panel_label(axes[0], "(a)")

    non_cache = np.array([row["non_cache_input_tokens"] for row in tasks]) / 1e6
    cache = np.array([row["cache_tokens"] for row in tasks]) / 1e6
    output = np.array([row["output_tokens"] for row in tasks]) / 1e6
    axes[1].bar(
        x,
        non_cache,
        color=NON_CACHE_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Non-cache-read prompt",
    )
    axes[1].bar(
        x,
        cache,
        bottom=non_cache,
        color=CACHE_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Cache-read prompt",
    )
    axes[1].bar(
        x,
        output,
        bottom=non_cache + cache,
        color=OUTPUT_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Output",
    )
    axes[1].set_ylabel("Solving-side tokens (million)")
    axes[1].set_xlabel("Challenge")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, (non_cache + cache + output).max() * 1.14)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].text(
        0.02,
        0.97,
        f"Total: {run['total_tokens'] / 1e6:.3f}M",
        transform=axes[1].transAxes,
        va="top",
    )
    finish_axes(axes[1])
    panel_label(axes[1], "(b)")

    resource = COMPARISON["agent_resource_use"]
    points = [
        (resource["high"], HIGH_COLOR),
        (resource["ultra"], ULTRA_COLOR),
    ]
    for item, color in points:
        x_value = item["solve_time_per_valid_solution_min"]
        y_value = item["cost_per_valid_solution_usd"]
        axes[2].scatter(
            x_value,
            y_value,
            s=item["cost_usd"] * 65,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
    high = resource["high"]
    ultra = resource["ultra"]
    axes[2].annotate(
        "High\n(10/12)",
        (
            high["solve_time_per_valid_solution_min"],
            high["cost_per_valid_solution_usd"],
        ),
        xytext=(-65, -18),
        textcoords="offset points",
        color=HIGH_COLOR,
        fontweight="bold",
    )
    axes[2].annotate(
        "Ultra\n(10/12)",
        (
            ultra["solve_time_per_valid_solution_min"],
            ultra["cost_per_valid_solution_usd"],
        ),
        xytext=(12, 6),
        textcoords="offset points",
        color=ULTRA_COLOR,
        fontweight="bold",
    )
    axes[2].set_xlabel("Solve time per valid solution (min)")
    axes[2].set_ylabel("Cost per valid solution (USD)")
    axes[2].set_xlim(16.0, 21.2)
    axes[2].set_ylim(2.30, 3.20)
    axes[2].text(
        0.98,
        0.03,
        "Marker area proportional to total cost",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    finish_axes(axes[2])
    axes[2].grid(color=GRID_COLOR, linewidth=0.8)
    panel_label(axes[2], "(c)")

    fig.suptitle(
        "GPT-5.6 Sol ultra agent-side resource use",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=2.4)
    save(fig, "gpt56sol-ultra-agent-resource-use.png")


def outcomes_figure() -> None:
    validity = COMPARISON["validity"]
    runtime = COMPARISON["artifact_runtime"]
    high_tasks = COMPARISON["agent_resource_use"]["high"]["tasks"]
    ultra_tasks = COMPARISON["agent_resource_use"]["ultra"]["tasks"]
    task_numbers = np.arange(1, 13)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.0),
        gridspec_kw={"width_ratios": [0.85, 1.55, 1.55]},
    )

    matrix = np.array(
        [
            [int(row["passed"]) for row in high_tasks],
            [int(row["passed"]) for row in ultra_tasks],
        ]
    )
    cmap = ListedColormap(["#FFFFFF", "#D9EDE7"])
    axes[0].imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(12), [f"{x:02d}" for x in task_numbers])
    axes[0].set_yticks([0, 1], ["High", "Ultra"])
    axes[0].set_xlabel("Challenge")
    for row in range(2):
        for col in range(12):
            passed = bool(matrix[row, col])
            color = HIGH_COLOR if row == 0 else ULTRA_COLOR
            axes[0].text(
                col,
                row,
                "P" if passed else "F",
                ha="center",
                va="center",
                color=color if passed else "#222222",
                fontweight="bold",
            )
    axes[0].set_xticks(np.arange(-0.5, 12, 1), minor=True)
    axes[0].set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axes[0].grid(which="minor", color="#BBBBBB", linewidth=0.8)
    axes[0].tick_params(which="minor", bottom=False, left=False)
    axes[0].set_title("Final task outcomes", loc="left")
    axes[0].text(
        0.0,
        -0.22,
        "Both: 02–07, 09–12\nUltra only: none  |  Neither: 01, 08",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
    )
    panel_label(axes[0], "(a)")

    rows = runtime["tasks"]
    high_runtime = np.array([row["high_runtime_sec"] for row in rows])
    ultra_runtime = np.array([row["ultra_runtime_sec"] for row in rows])
    width = 0.37
    for values, offset, color, key, label in [
        (high_runtime, -width / 2, HIGH_COLOR, "high_passed", "High"),
        (ultra_runtime, width / 2, ULTRA_COLOR, "ultra_passed", "Ultra"),
    ]:
        for index, value in enumerate(values):
            passed = rows[index][key]
            axes[1].bar(
                task_numbers[index] + offset,
                value,
                width,
                color=color if passed else "white",
                edgecolor=color,
                linewidth=1.0,
                hatch=None if passed else "//",
                label=None,
            )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Candidate runtime (s, log scale)")
    axes[1].set_xlabel("Challenge")
    axes[1].set_xticks(task_numbers, [f"{x:02d}" for x in task_numbers])
    axes[1].legend(
        handles=[
            Patch(facecolor=HIGH_COLOR, edgecolor="black", label="High"),
            Patch(facecolor=ULTRA_COLOR, edgecolor="black", label="Ultra"),
            Patch(
                facecolor="white",
                edgecolor="#555555",
                hatch="//",
                label="Failed reward",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    axes[1].set_title("Task-level artifact runtime", loc="left")
    finish_axes(axes[1])
    panel_label(axes[1], "(b)")

    common = runtime["common_passed_tasks"]["tasks"]
    common_rows = [rows[task - 1] for task in common]
    common_x = np.arange(len(common))
    high_ratio = np.array([row["high_over_expert"] for row in common_rows])
    ultra_ratio = np.array([row["ultra_over_expert"] for row in common_rows])
    axes[2].bar(
        common_x - width / 2,
        high_ratio,
        width,
        color=HIGH_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="High",
    )
    axes[2].bar(
        common_x + width / 2,
        ultra_ratio,
        width,
        color=ULTRA_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Ultra",
    )
    axes[2].axhline(1, color="#888888", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Candidate runtime / expert reference")
    axes[2].set_xlabel("Common passed challenge")
    axes[2].set_xticks(common_x, [f"{x:02d}" for x in common])
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].set_title("Common-task artifact efficiency", loc="left")
    axes[2].text(
        0.98,
        0.97,
        (
            "Geometric mean\n"
            f"High: {runtime['common_passed_tasks']['high_geometric_mean_over_expert']:.2f}×\n"
            f"Ultra: {runtime['common_passed_tasks']['ultra_geometric_mean_over_expert']:.2f}×"
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    finish_axes(axes[2])
    panel_label(axes[2], "(c)")

    fig.suptitle(
        "GPT-5.6 Sol high vs ultra: validity and artifact efficiency",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=2.2)
    save(fig, "gpt56sol-high-vs-ultra-outcomes.png")


def resource_comparison_figure() -> None:
    resource = COMPARISON["agent_resource_use"]
    high = resource["high"]
    ultra = resource["ultra"]
    high_tasks = high["tasks"]
    ultra_tasks = ultra["tasks"]
    task_numbers = np.arange(1, 13)
    width = 0.37

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))

    metrics = [
        (
            "agent_wall_sec",
            60,
            "Agent solve wall time (min)",
            f"Totals: H {high['agent_wall_sec'] / 60:.1f} | U {ultra['agent_wall_sec'] / 60:.1f} min",
        ),
        (
            "total_tokens",
            1e6,
            "Solving-side tokens (million)",
            f"Totals: H {high['total_tokens'] / 1e6:.3f}M | U {ultra['total_tokens'] / 1e6:.3f}M",
        ),
        (
            "cost_usd",
            1,
            "Recorded solver cost (USD)",
            f"Totals: H ${high['cost_usd']:.2f} | U ${ultra['cost_usd']:.2f}",
        ),
    ]
    for panel, (key, divisor, ylabel, total_text) in enumerate(metrics):
        high_values = np.array([row[key] for row in high_tasks]) / divisor
        ultra_values = np.array([row[key] for row in ultra_tasks]) / divisor
        axes[panel].bar(
            task_numbers - width / 2,
            high_values,
            width,
            color=HIGH_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label="High",
        )
        axes[panel].bar(
            task_numbers + width / 2,
            ultra_values,
            width,
            color=ULTRA_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label="Ultra",
        )
        axes[panel].set_ylabel(ylabel)
        axes[panel].set_xlabel("Challenge")
        axes[panel].set_xticks(
            task_numbers, [f"{x:02d}" for x in task_numbers]
        )
        axes[panel].legend(frameon=False, loc="upper right")
        axes[panel].text(
            0.02,
            0.97,
            total_text,
            transform=axes[panel].transAxes,
            va="top",
            fontsize=9,
        )
        finish_axes(axes[panel])
        panel_label(axes[panel], f"({chr(ord('a') + panel)})")

    delta = resource["ultra_minus_high_percent"]
    fig.suptitle(
        (
            "GPT-5.6 Sol high vs ultra: per-task solver resources\n"
            f"Ultra total tokens {delta['total_tokens']:+.1f}%, "
            f"cost {delta['cost_usd']:+.1f}%, "
            f"agent wall time {delta['agent_wall_sec']:+.1f}%"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout(w_pad=2.2)
    save(fig, "gpt56sol-high-vs-ultra-resources.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
        }
    )
    ultra_resource_figure()
    outcomes_figure()
    resource_comparison_figure()


if __name__ == "__main__":
    main()
