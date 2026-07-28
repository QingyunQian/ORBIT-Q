"""Generate paper-style summary figures from the recorded per-task results.

Reads challenge-NN/reward.json (official stamps) and
challenge-NN/runtime-comparison.json, writes two figures into figs/.
Run from the repository root:  python3 results/fable5/tools/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

tasks, ratios, rewards, cand_t, ref_t, audits = [], [], [], [], [], []
for nn in range(1, 13):
    d = ROOT / f"challenge-{nn:02d}"
    rc = json.loads((d / "runtime-comparison.json").read_text())
    rw = json.loads((d / "reward.json").read_text())
    tasks.append(f"{nn:02d}")
    key = "runtime_ratio_candidate_over_reference"
    ratios.append(rc.get(key) or rc["primary"][key])
    cand = rc.get("candidate_runtime_sec") or rc["primary"]["candidate_runtime_sec"]
    ref = (rc.get("reference_official") or rc["primary"]["reference_official"])["runtime_sec"]
    cand_t.append(cand)
    ref_t.append(ref)
    rewards.append(rw["reward"])
    audits.append(rw["llm_audit_score"])

ratios = np.array(ratios)
rewards = np.array(rewards)

# --- Figure 1: dual-axis scatter (validity vs artifact efficiency) ---
fig, ax = plt.subplots(figsize=(7.2, 4.6))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tasks)))
for x, y, t, c in zip(ratios, rewards, tasks, colors):
    ax.scatter(x, y, s=110, color=c, edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9)
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
ax.text(1.0, 0.62, " expert reference\n (T/T_ref = 1)", fontsize=8, color="gray")
ax.set_xscale("log")
ax.set_xlim(0.5, 30)
ax.set_ylim(0.55, 1.08)
ax.set_xlabel("artifact runtime ratio  T / T$_{ref}$  (log scale, lower is better)")
ax.set_ylabel("official pass reward\n(functional x static x LLM audit)")
ax.set_title("ORBIT-Q agent axis: Fable 5 (Cursor agent) x TensorCircuit-NG\n"
             "all 12 tasks pass with reward = 1.0; efficiency spread 0.74x - 18.1x",
             fontsize=10)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIGS / "fable5_dual_axis_scatter.png", dpi=180)

# --- Figure 2: per-task T/T_ref bars with runtimes ---
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bar_colors = ["#2a9d8f" if r <= 1.0 else "#e9c46a" if r <= 4 else "#e76f51" for r in ratios]
bars = ax.bar(tasks, ratios, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
for b, r, ct, rt in zip(bars, ratios, cand_t, ref_t):
    ax.annotate(f"{r:.2f}\n{ct:.0f}s/{rt:.0f}s", (b.get_x() + b.get_width() / 2, r),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)
# matched-precision (complex64) control overlay where the official candidate ran complex128
for i, nn in enumerate(range(1, 13)):
    rc = json.loads((ROOT / f"challenge-{nn:02d}" / "runtime-comparison.json").read_text())
    mc = rc.get("matched_precision_control")
    if mc and mc.get("matched_precision_ratio"):
        ax.scatter([i], [mc["matched_precision_ratio"]], marker="D", s=42,
                   color="white", edgecolor="#264653", linewidth=1.4, zorder=4)
    elif mc:
        ax.annotate("c64\nfails", (i, 0.55), ha="center", fontsize=7, color="#e76f51")
ax.scatter([], [], marker="D", s=42, color="white", edgecolor="#264653",
           linewidth=1.4, label="complex64 control (matched precision)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_yscale("log")
ax.set_ylim(0.4, 40)
ax.set_xlabel("challenge task")
ax.set_ylabel("T / T$_{ref}$  (log scale)")
ax.set_title("Per-task artifact efficiency vs reference, same machine & image "
             "(annotated: candidate/reference seconds)", fontsize=9.5)
ax.grid(alpha=0.25, axis="y", which="both")
fig.tight_layout()
fig.savefig(FIGS / "fable5_ratio_bars.png", dpi=180)

print("figures written:", sorted(p.name for p in FIGS.iterdir()))
print("ratios:", dict(zip(tasks, np.round(ratios, 2))))
print("median ratio:", round(float(np.median(ratios)), 2))
