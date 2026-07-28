"""Generate paper-style summary figures from the recorded per-task results.

Reads challenge-NN/reward.json (official stamps) and
challenge-NN/runtime-comparison.json, writes two figures into figs/.
The dual-axis scatter uses unified matched-precision (complex64) ratios:
the publication references all run complex64, so tasks whose official
candidate ran complex128 are plotted with their complex64-control ratio;
task 04's control fails functionally, so it is plotted at its official
(complex128) ratio with a distinct marker.
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

tasks, official, matched, needs_c128, rewards, cand_t, ref_t = [], [], [], [], [], [], []
for nn in range(1, 13):
    d = ROOT / f"challenge-{nn:02d}"
    rc = json.loads((d / "runtime-comparison.json").read_text())
    rw = json.loads((d / "reward.json").read_text())
    key = "runtime_ratio_candidate_over_reference"
    off = rc.get(key) or rc["primary"][key]
    cand = rc.get("candidate_runtime_sec") or rc["primary"]["candidate_runtime_sec"]
    ref = (rc.get("reference_official") or rc["primary"]["reference_official"])["runtime_sec"]
    mc = rc.get("matched_precision_control")
    if mc and mc.get("matched_precision_ratio"):
        m, req128 = mc["matched_precision_ratio"], False
    elif mc:  # complex64 control exists but fails functionally
        m, req128 = off, True
    else:  # official candidate already complex64 => already matched
        m, req128 = off, False
    tasks.append(f"{nn:02d}")
    official.append(off)
    matched.append(m)
    needs_c128.append(req128)
    rewards.append(rw["reward"])
    cand_t.append(cand)
    ref_t.append(ref)

official = np.array(official)
matched = np.array(matched)
rewards = np.array(rewards)

# --- Figure 1: dual-axis scatter, unified matched-precision (complex64) ---
fig, ax = plt.subplots(figsize=(7.2, 4.6))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tasks)))
for x, y, t, c, req in zip(matched, rewards, tasks, colors, needs_c128):
    marker = "s" if req else "o"
    ax.scatter(x, y, s=110, color=c, marker=marker, edgecolor="black",
               linewidth=0.6, zorder=3)
    dy = -18 if t == "06" else 9  # keep the 06/11 labels from colliding
    ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=9)
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
ax.text(1.0, 0.62, " expert reference\n (T/T_ref = 1)", fontsize=8, color="gray")
ax.scatter([], [], marker="s", s=90, color="lightgray", edgecolor="black",
           label="requires complex128 (c64 control fails); plotted at official ratio")
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
ax.set_xscale("log")
ax.set_xlim(0.5, 30)
ax.set_ylim(0.55, 1.08)
ax.set_xlabel("artifact runtime ratio  T / T$_{ref}$  at matched precision, complex64  (log scale)")
ax.set_ylabel("official pass reward\n(functional x static x LLM audit)")
ax.set_title("ORBIT-Q agent axis: Fable 5 (Cursor agent) x TensorCircuit-NG\n"
             "all 12 tasks pass with reward = 1.0; matched-precision efficiency 0.74x - 18.1x",
             fontsize=10)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIGS / "fable5_dual_axis_scatter.png", dpi=180)

# --- Figure 2: per-task official bars + matched-precision control diamonds ---
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bar_colors = ["#2a9d8f" if r <= 1.0 else "#e9c46a" if r <= 4 else "#e76f51" for r in official]
bars = ax.bar(tasks, official, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
for b, r, ct, rt in zip(bars, official, cand_t, ref_t):
    ax.annotate(f"{r:.2f}\n{ct:.0f}s/{rt:.0f}s", (b.get_x() + b.get_width() / 2, r),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)
for i, (m, off, req) in enumerate(zip(matched, official, needs_c128)):
    if req:
        ax.annotate("c64 fails", (i, 0.47), ha="center", fontsize=8,
                    color="white", fontweight="bold")
    elif abs(m - off) > 1e-9:
        ax.scatter([i], [m], marker="D", s=42, color="white",
                   edgecolor="#264653", linewidth=1.4, zorder=4)
        ax.annotate(f"{m:.2f}", (i, m), textcoords="offset points",
                    xytext=(14, -3), fontsize=7.5, color="#264653", fontweight="bold")
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
print("official:", dict(zip(tasks, np.round(official, 2))))
print("matched :", dict(zip(tasks, np.round(matched, 2))))
