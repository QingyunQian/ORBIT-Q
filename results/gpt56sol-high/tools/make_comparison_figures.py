"""Plot the GPT-5.5 paper run against the GPT-5.6 Sol high rerun."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "runtime-comparison-mac.json").read_text())
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

old = DATA["gpt55_paper_run"]
new = DATA["gpt56_run"]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.8, 5.2))

# (a) Same validity/efficiency coordinates as the paper's agent axis.
ax0.scatter(
    [0],
    [1],
    marker="D",
    s=170,
    facecolor="#eeeeee",
    edgecolor="#777777",
    linewidth=1.2,
    zorder=3,
)
ax0.scatter(
    [old["failure_rate_pct"]],
    [old["passed_task_geometric_mean_slowdown"]],
    s=210,
    color="#0072B2",
    edgecolor="black",
    linewidth=1,
    zorder=3,
)
ax0.scatter(
    [new["failure_rate_pct"]],
    [new["passed_task_geometric_mean_slowdown"]],
    s=240,
    color="#D55E00",
    edgecolor="black",
    linewidth=1,
    zorder=4,
)
ax0.axhline(1, color="#888888", linestyle="--", linewidth=1)
ax0.axvline(0, color="#888888", linestyle="--", linewidth=1)
ax0.annotate("Expert TC\n(12/12)", (0, 1), xytext=(7, -2), textcoords="offset points")
ax0.annotate(
    f"GPT-5.5 high\n({old['passes']}/12, {old['passed_task_geometric_mean_slowdown']:.2f}×)",
    (old["failure_rate_pct"], old["passed_task_geometric_mean_slowdown"]),
    xytext=(10, 3),
    textcoords="offset points",
    color="#0072B2",
    weight="bold",
)
ax0.annotate(
    f"GPT-5.6 Sol high\n({new['passes']}/12, {new['passed_task_geometric_mean_slowdown']:.2f}×)",
    (new["failure_rate_pct"], new["passed_task_geometric_mean_slowdown"]),
    xytext=(10, -34),
    textcoords="offset points",
    color="#D55E00",
    weight="bold",
)
ax0.set_xlim(-2, 35)
ax0.set_ylim(0.8, 2.55)
ax0.set_xticks([0, 10, 20, 30])
ax0.set_xlabel("Failure rate (%)")
ax0.set_ylabel("Geometric mean runtime / expert TC\n(passed tasks; lower is better)")
ax0.set_title("(a) Agent-axis validity and artifact efficiency", loc="left")
ax0.grid(alpha=0.25)

# (b) Task-level ratios. F denotes a failed reward and is excluded from gmean.
tasks = np.arange(1, 13)
width = 0.36
old_map = {x["task"]: x for x in old["tasks"]}
new_map = {x["task"]: x for x in new["tasks"]}
old_values = [
    old_map[i]["runtime_ratio"] if old_map[i]["passed"] else np.nan for i in tasks
]
new_values = [
    new_map[i]["runtime_ratio"] if new_map[i]["reward"] == 1 else np.nan for i in tasks
]
ax1.bar(
    tasks - width / 2,
    old_values,
    width,
    label="GPT-5.5 high",
    color="#0072B2",
    edgecolor="black",
    linewidth=0.5,
)
ax1.bar(
    tasks + width / 2,
    new_values,
    width,
    label="GPT-5.6 Sol high",
    color="#D55E00",
    edgecolor="black",
    linewidth=0.5,
)
for task in tasks:
    if not old_map[task]["passed"]:
        ax1.text(
            task - width / 2,
            0.43,
            "F",
            ha="center",
            va="center",
            color="#0072B2",
            weight="bold",
        )
    if new_map[task]["reward"] != 1:
        ax1.text(
            task + width / 2,
            0.43,
            "F",
            ha="center",
            va="center",
            color="#D55E00",
            weight="bold",
        )
ax1.axhline(1, color="#888888", linestyle="--", linewidth=1)
ax1.set_yscale("log")
ax1.set_ylim(0.35, 6.2)
ax1.set_xticks(tasks)
ax1.set_xticklabels([f"{x:02d}" for x in tasks])
ax1.set_xlabel("Challenge task")
ax1.set_ylabel("Candidate runtime / same-run expert reference")
ax1.set_title("(b) Task-level artifact runtime; F = failed reward", loc="left")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.25, axis="y", which="both")

fig.suptitle(
    "ORBIT-Q TensorCircuit: GPT-5.5 high vs GPT-5.6 Sol high (adjudicated)",
    y=1.01,
    fontsize=12,
)
fig.text(
    0.5,
    0.005,
    "Challenge 05 corrected after source- and runtime-API-grounded adjudication.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.035, 1, 1))
out = FIGS / "gpt55-vs-gpt56-comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
