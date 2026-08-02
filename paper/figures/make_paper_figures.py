"""Generate paper-ready ORBIT-Q result figures from committed source data.

The script deliberately keeps the benchmark outcome layer separate from the
artifact-speed and agent-resource layers.  Missing reference ratios and
non-comparable Fable resource data are shown as missing rather than inferred.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "source_data"
OUT = ROOT / "paper" / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Keep the publication exports explicit so the static figure preflight and a
# reader can audit the deliverables without executing the script first.
VECTOR_EXTENSIONS = (".svg", ".pdf")
RASTER_EXTENSIONS = (".png", ".tiff")
# dpi=600 is used for both raster deliverables.
RASTER_DPI = 600

MODEL_ORDER = [
    "gpt56_sol_high",
    "gpt56_sol_ultra",
    "gpt56_terra_high",
    "gpt56_luna_high",
    "deepseek_v4_flash_high",
    "deepseek_v4_flash_max",
    "fable5",
]
COLORS = {
    "gpt56_sol_high": "#0072B2",
    "gpt56_sol_ultra": "#D55E00",
    "gpt56_terra_high": "#009E73",
    "gpt56_luna_high": "#CC79A7",
    "deepseek_v4_flash_high": "#4D4D4D",
    "deepseek_v4_flash_max": "#B2182B",
    "fable5": "#756BB1",
}
SHORT = {
    "gpt56_sol_high": "Sol high",
    "gpt56_sol_ultra": "Sol ultra",
    "gpt56_terra_high": "Terra high",
    "gpt56_luna_high": "Luna high",
    "deepseek_v4_flash_high": "DeepSeek high",
    "deepseek_v4_flash_max": "DeepSeek max",
    "fable5": "Fable 5",
}
SCATTER_SHORT = {
    "gpt56_sol_high": "Sol high",
    "gpt56_sol_ultra": "Sol ultra",
    "gpt56_terra_high": "Terra",
    "gpt56_luna_high": "Luna",
    "deepseek_v4_flash_high": "DS high",
    "deepseek_v4_flash_max": "DS max",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for ext in VECTOR_EXTENSIONS:
        fig.savefig(OUT / f"{stem}{ext}")
    for ext in RASTER_EXTENSIONS:
        fig.savefig(OUT / f"{stem}{ext}", dpi=RASTER_DPI)
    plt.close(fig)


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_1(models: list[dict[str, str]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)
    rows = [r for r in models if r["model"] in MODEL_ORDER]
    labels = [SHORT[r["model"]] for r in rows]
    passes = np.array([float(r["passes"]) for r in rows])
    n_failed = 12 - passes
    y = np.arange(len(rows))
    colors = [COLORS[r["model"]] for r in rows]
    ax0.barh(y, passes, color=colors, edgecolor="white", linewidth=0.5, label="Final valid")
    ax0.barh(y, n_failed, left=passes, color="#E6E6E6", edgecolor="white", linewidth=0.5, label="Failed")
    for yy, p in zip(y, passes):
        ax0.text(p + 0.16, yy, f"{int(p)}/12", va="center", ha="left", fontsize=6.6)
    ax0.set_yticks(y, labels)
    ax0.invert_yaxis()
    ax0.set_xlim(0, 12.9)
    ax0.set_xticks([0, 3, 6, 9, 12])
    ax0.set_xlabel("Final valid solutions")
    ax0.set_title("Final validity across 12 tasks", loc="left", fontweight="bold")
    style_axes(ax0)
    ax0.legend(frameon=False, ncol=2, loc="lower right", handlelength=1.0, columnspacing=0.8)

    # Paper-style failure-rate / relative-runtime plane for measurements with a
    # committed same-machine reference ratio. Missing new-run ratios stay out.
    measured = [r for r in rows if r["gm_slowdown"]]
    for r in measured:
        x = float(r["gm_slowdown"])
        fail_rate = 100.0 * float(r["failures"]) / 12.0
        ax1.scatter(x, fail_rate, s=52, color=COLORS[r["model"]], edgecolor="black", linewidth=0.45, zorder=3)
        if r["model"] == "gpt56_sol_ultra":
            tx, ty, ha = x * 1.03, fail_rate + 1.25, "left"
        elif r["model"] == "gpt56_sol_high":
            tx, ty, ha = x * 1.10, fail_rate + 0.25, "left"
        elif r["model"] == "fable5":
            tx, ty, ha = x * 1.04, fail_rate + 1.25, "left"
        else:
            tx, ty, ha = x * 1.04, fail_rate + 1.6, "left"
        ax1.text(tx, ty, SHORT[r["model"]], color=COLORS[r["model"]], ha=ha, va="bottom", fontsize=6.1)
    ax1.scatter(1.0, 0.0, marker="D", s=36, color="#666666", edgecolor="black", linewidth=0.4, zorder=3)
    ax1.text(1.0, 2.2, "Expert\nreference", ha="center", va="bottom", fontsize=6.1, color="#555555")
    ax1.set_xscale("log")
    ax1.set_xlim(0.75, 23)
    ax1.set_ylim(-2, 23)
    ax1.set_xlabel("Geometric-mean artifact slowdown / expert")
    ax1.set_ylabel("Failure rate (%)")
    ax1.set_title("Dual-axis view (ratios available)", loc="left", fontweight="bold")
    ax1.text(0.02, 0.02, "Terra/Luna/DeepSeek ratios not committed", transform=ax1.transAxes, fontsize=5.8, color="#666666")
    ax1.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig.suptitle("ORBIT-Q agent-axis results", fontsize=9.5, fontweight="bold")
    save_figure(fig, "fig1_dual_axis_updated")


def figure_2(outcomes: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.45), constrained_layout=True)
    order = [m for m in MODEL_ORDER]
    arr = np.full((len(order), 12), np.nan)
    for r in outcomes:
        if r["model"] in order:
            arr[order.index(r["model"]), int(r["task"]) - 1] = float(r["final_pass"])
    cmap = mpl.colors.ListedColormap(["#F2F2F2", "#009E73"])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, "P" if val == 1 else "F", ha="center", va="center", fontsize=6.4, color="#111111")
    ax.set_xticks(np.arange(12), [f"{i:02d}" for i in range(1, 13)])
    ax.set_yticks(np.arange(len(order)), [SHORT[m] for m in order])
    ax.set_xlabel("Challenge")
    ax.set_title("Task-level validity matrix (final adjudicated outcome)", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.text(0.0, -0.17, "P = final valid artifact; F = failed functional, static, audit, or manual-review criterion. Task 08 is F for all six GPT-5.6/DeepSeek rows after final expert review.", transform=ax.transAxes, fontsize=5.8, color="#444444")
    save_figure(fig, "fig2_task_level_matrix")


def figure_3(models: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.9))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.30, top=0.78, wspace=0.52)
    rows = [r for r in models if r["resource_comparable"] == "yes"]
    labels = [SHORT[r["model"]] for r in rows]
    x = np.arange(len(rows))
    colors = [COLORS[r["model"]] for r in rows]
    wall = np.array([float(r["agent_wall_min"]) for r in rows])
    toks = np.array([float(r["total_tokens_m"]) for r in rows])
    c = np.array([float(r["cost_per_valid_usd"]) for r in rows])
    axes[0].bar(x, wall, color=colors, edgecolor="white", linewidth=0.4)
    axes[0].set_ylabel("Agent wall time (min)")
    axes[0].set_title("Solve wall time", loc="left", fontweight="bold")
    axes[1].bar(x, toks, color=colors, edgecolor="white", linewidth=0.4)
    axes[1].set_ylabel("Solving-side tokens (M)")
    axes[1].set_title("Token use", loc="left", fontweight="bold")
    axes[2].scatter(wall / np.array([float(r["passes"]) for r in rows]), c, s=np.maximum(c * 120, 24), c=colors, edgecolor="black", linewidth=0.4)
    label_offsets = {
        "gpt56_sol_high": (0.8, -0.10),
        "gpt56_sol_ultra": (-0.9, 0.10),
        "gpt56_terra_high": (-0.2, 0.10),
        "gpt56_luna_high": (0.0, 0.10),
        "deepseek_v4_flash_high": (-3.4, 0.34),
        "deepseek_v4_flash_max": (1.3, 0.06),
    }
    for idx, (xx, yy) in enumerate(zip(wall / np.array([float(r["passes"]) for r in rows]), c)):
        dx, dy = label_offsets[rows[idx]["model"]]
        axes[2].text(xx + dx, yy + dy, SCATTER_SHORT[rows[idx]["model"]], fontsize=5.7, ha="center", va="bottom")
    axes[2].set_xlabel("Solve time / valid solution (min)")
    axes[2].set_ylabel("Cost / valid solution (USD)")
    axes[2].set_title("Cost / valid", loc="left", fontweight="bold")
    axes[2].grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axes[2].set_axisbelow(True)
    for ax in axes[:2]:
        ax.set_xticks(x, labels, rotation=45, ha="right")
        style_axes(ax)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Agent-side resource use", y=0.98, fontsize=9.5, fontweight="bold")
    fig.text(0.5, -0.02, "Fable 5 is omitted: the Cursor/Fable harness did not record comparable solving-side token, wall-time, or cost fields.", ha="center", fontsize=5.8, color="#444444")
    save_figure(fig, "fig3_agent_resources")


def figure_4(expert: list[dict[str, str]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 3.35), constrained_layout=True, gridspec_kw={"width_ratios": [1.08, 1.0]})
    tasks = np.array([int(r["task"]) for r in expert])
    speed = np.array([float(r["speedup"]) for r in expert])
    dom = np.array([float(r["dominant_factor_speedup"]) for r in expert])
    base = np.array([float(r["baseline_sec"]) for r in expert])
    opt = np.array([float(r["optimized_sec"]) for r in expert])
    y = np.arange(len(tasks))
    bar_colors = ["#B2182B" if s < 1.1 else "#009E73" for s in speed]
    ax0.barh(y, speed, color=bar_colors, edgecolor="white", linewidth=0.5)
    for yy, s in zip(y, speed):
        ax0.text(s * 1.04, yy, f"{s:.2f}×", va="center", fontsize=6.0)
    ax0.axvline(1.0, color="#333333", linestyle="--", linewidth=0.7)
    ax0.set_xscale("log")
    ax0.set_yticks(y, [f"Task {t:02d}" for t in tasks])
    ax0.invert_yaxis()
    ax0.set_xlabel("Expert baseline / optimized runtime")
    ax0.set_title("Human-expert optimization", loc="left", fontweight="bold")
    ax0.text(0.02, -0.18, "Task 08 interval crosses 1×; bounded batching is a memory-completion result, not a confirmed speedup.", transform=ax0.transAxes, fontsize=5.7, color="#444444")
    ax0.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax0.set_axisbelow(True)
    ax1.barh(y + 0.18, base, height=0.34, color="#A6CEE3", label="Public expert")
    ax1.barh(y - 0.18, opt, height=0.34, color="#1F78B4", label="Optimized expert")
    ax1.set_xscale("log")
    ax1.set_yticks(y, [f"{t:02d}" for t in tasks])
    ax1.invert_yaxis()
    ax1.set_xlabel("Measured runtime (s; log scale)")
    ax1.set_title("Before / after runtime", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower right", handlelength=1.0)
    ax1.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax1.set_axisbelow(True)
    fig.suptitle("TensorCircuit-NG human-expert reference optimization", fontsize=9.5, fontweight="bold")
    save_figure(fig, "fig4_expert_speedups")


def figure_5(insights: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.23, top=0.82)
    rows = insights
    y = np.arange(len(rows))
    vals = []
    for r in rows:
        s = r["measured_effect"].split("x")[0]
        vals.append(float(s))
    colors = ["#D55E00" if r["task"] in {"03", "07"} else "#0072B2" for r in rows]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"Task {r['task']} — {r['primary_factor']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Reported end-to-end speedup (×; log scale)")
    ax.set_title("Take-home optimization insights", loc="left", fontweight="bold")
    for yy, v in zip(y, vals):
        ax.text(v * 1.04, yy, f"{v:.2f}×", va="center", fontsize=6.0)
    ax.axvline(1, color="#333333", linestyle="--", linewidth=0.7)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.22, "Orange: exact challenge-design reductions (Tasks 03 and 07); blue: framework-native implementation improvements. Factor rows are direct ablations and are not additive.", transform=ax.transAxes, fontsize=5.8, color="#444444")
    save_figure(fig, "fig5_insight_speedups")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1(read_csv("benchmark_models.csv"))
    figure_2(read_csv("task_outcomes.csv"))
    figure_3(read_csv("benchmark_models.csv"))
    figure_4(read_csv("expert_optimization.csv"))
    figure_5(read_csv("insights.csv"))


if __name__ == "__main__":
    main()
