#!/usr/bin/env python3
"""Render one paired-speedup SVG for every measured Task 02 factor."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"
FIGURES = ROOT / "figures"

FACTORS = [
    ("e01-gate-fusion-paired.json", "e01-gate-fusion.svg", "Exact local gate fusion"),
    ("e02-training-scan-paired.json", "e02-training-scan.svg", "Whole-training K.jaxy_scan"),
    ("e03-purity-paired.json", "e03-frobenius-purity.svg", "Exact Frobenius purity"),
    ("e04-sparse-xxz-paired.json", "e04-sparse-xxz.svg", "TensorCircuit sparse XXZ"),
    ("e05-packed-params-paired.json", "e05-packed-params.svg", "Single-leaf parameter packing"),
    ("e06-entropy-vmap-paired.json", "e06-entropy-vmap.svg", "Checkpoint entropy K.vmap"),
    ("e07-batched-purity-paired.json", "e07-batched-purity.svg", "K.vmap + exact purity"),
]


def render_factor(profile_name, output_name, title):
    data = json.loads((PROFILES / profile_name).read_text())
    ratios = [pair["speedup"] for pair in data["pairs"]]
    mean = data["mean_paired_speedup"]
    low, high = data["paired_speedup_95pct_t_ci"]
    decision = data["decision"]
    color = "#2E7D32" if decision == "keep" else "#B26A00"
    if decision == "discard":
        color = "#B3261E"

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(1.0, color="#303030", linewidth=1.0)
    ax.scatter(range(1, 7), ratios, color=color, s=42, zorder=3)
    ax.errorbar(
        7,
        mean,
        yerr=[[mean - low], [high - mean]],
        fmt="D",
        color=color,
        capsize=6,
        markersize=7,
        linewidth=2,
    )
    span = max(max(ratios + [high, 1.0]) - min(ratios + [low, 1.0]), 0.08)
    ax.set_ylim(
        min(ratios + [low, 1.0]) - 0.16 * span,
        max(ratios + [high, 1.0]) + 0.22 * span,
    )
    ax.set_xticks(range(1, 8), ["1", "2", "3", "4", "5", "6", "mean"])
    ax.set_xlabel("Alternating matched pair")
    ax.set_ylabel("Speedup (parent runtime / factor runtime)")
    fig.suptitle(title, y=0.96, fontsize=13)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6)
    ax.text(
        0.02,
        0.94,
        f"mean {mean:.4f}×  ·  95% t-CI [{low:.4f}, {high:.4f}]  ·  {decision}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.text(6.85, 1.0, "no change", ha="right", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.82)
    fig.savefig(
        FIGURES / output_name,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "task-02-factor-ablations"
    plt.rcParams["font.family"] = "DejaVu Sans"
    for args in FACTORS:
        render_factor(*args)


if __name__ == "__main__":
    main()
