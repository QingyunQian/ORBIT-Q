#!/usr/bin/env python3
"""Build sanitized Task 03 ablation JSON and dependency-free SVG figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


T95_DF5 = 2.570581835636305


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(label, path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload["summary"]["comparison"]["tasks"]["03"]
    pairs = comparison["pairs"]
    speedups = [row["speedup"] for row in pairs]
    mean = statistics.mean(speedups)
    stderr = statistics.stdev(speedups) / math.sqrt(len(speedups))
    return {
        "factor": label,
        "raw_report_sha256": sha256(path),
        "passing_cells": sum(row["passed"] for row in payload["results"]),
        "total_cells": len(payload["results"]),
        "reference_mean_runtime_sec": comparison["reference_mean"],
        "reference_median_runtime_sec": comparison["reference_median"],
        "reference_stderr_sec": comparison["reference_stderr"],
        "candidate_mean_runtime_sec": comparison["candidate_mean"],
        "candidate_median_runtime_sec": comparison["candidate_median"],
        "candidate_stderr_sec": comparison["candidate_stderr"],
        "ratio_of_means_speedup": comparison["speedup"],
        "ratio_of_means_reduction_pct": comparison["improvement_pct"],
        "paired_speedups": speedups,
        "paired_speedup_mean": mean,
        "paired_speedup_median": statistics.median(speedups),
        "paired_speedup_stderr": stderr,
        "paired_speedup_ci95": [
            mean - T95_DF5 * stderr,
            mean + T95_DF5 * stderr,
        ],
        "wins": sum(value > 1.0 for value in speedups),
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "order": row["pair_order"],
                "reference_runtime_sec": row["reference_runtime_sec"],
                "candidate_runtime_sec": row["candidate_runtime_sec"],
                "speedup": row["speedup"],
            }
            for row in pairs
        ],
    }


def svg(summary, output):
    values = summary["paired_speedups"]
    mean = summary["paired_speedup_mean"]
    low, high = summary["paired_speedup_ci95"]
    upper = max(values + [high]) * 1.12
    left, top, width, height = 78, 46, 650, 250

    def y(value):
        return top + height - value / upper * height

    bars = []
    bar_width = width / len(values) * 0.58
    for index, value in enumerate(values):
        x = left + (index + 0.5) * width / len(values) - bar_width / 2
        bars.append(
            f'<rect x="{x:.2f}" y="{y(value):.2f}" width="{bar_width:.2f}" '
            f'height="{top + height - y(value):.2f}" fill="#2563eb"/>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{top + height + 22}" '
            f'text-anchor="middle" font-size="12">P{index + 1}</text>'
        )
    ci_x = left + width + 28
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="390" viewBox="0 0 860 390">
<rect width="860" height="390" fill="white"/>
<text x="28" y="25" font-family="sans-serif" font-size="17" font-weight="bold">{summary["factor"]}</text>
<line x1="{left}" y1="{y(1):.2f}" x2="{left + width}" y2="{y(1):.2f}" stroke="#dc2626" stroke-dasharray="5 4"/>
<text x="{left - 9}" y="{y(1) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">1.0x</text>
<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#111827"/>
{''.join(bars)}
<line x1="{ci_x}" y1="{y(low):.2f}" x2="{ci_x}" y2="{y(high):.2f}" stroke="#111827" stroke-width="3"/>
<line x1="{ci_x - 8}" y1="{y(low):.2f}" x2="{ci_x + 8}" y2="{y(low):.2f}" stroke="#111827" stroke-width="3"/>
<line x1="{ci_x - 8}" y1="{y(high):.2f}" x2="{ci_x + 8}" y2="{y(high):.2f}" stroke="#111827" stroke-width="3"/>
<circle cx="{ci_x}" cy="{y(mean):.2f}" r="6" fill="#f59e0b"/>
<text x="{ci_x + 16}" y="{y(mean) + 4:.2f}" font-family="sans-serif" font-size="12">mean {mean:.3f}x</text>
<text x="28" y="348" font-family="sans-serif" font-size="12">95% Student-t CI [{low:.3f}x, {high:.3f}x]; {summary["wins"]}/6 pairs won; {summary["passing_cells"]}/{summary["total_cells"]} cells passed.</text>
<text x="28" y="369" font-family="monospace" font-size="10">raw report sha256:{summary["raw_report_sha256"]}</text>
</svg>
"""
    output.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--local-vmap", type=Path, required=True)
    parser.add_argument("--training-scan", type=Path, required=True)
    parser.add_argument("--observable-vmap", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        summarize("Exact product-state reduction", args.product),
        summarize("Vectorized local conditional maps", args.local_vmap),
        summarize("Whole-training TensorCircuit scan", args.training_scan),
        summarize("Vectorized product-state observables", args.observable_vmap),
        summarize("Final candidate vs immutable expert", args.final),
    ]
    final_payload = json.loads(args.final.read_text(encoding="utf-8"))
    final_rows = final_payload["summary"]["comparison"]["tasks"]["03"]["pairs"][:5]
    rows[-1]["first_five"] = {
        "reference_mean_runtime_sec": statistics.mean(
            row["reference_runtime_sec"] for row in final_rows
        ),
        "candidate_mean_runtime_sec": statistics.mean(
            row["candidate_runtime_sec"] for row in final_rows
        ),
        "paired_speedup_mean": statistics.mean(
            row["speedup"] for row in final_rows
        ),
    }
    profiles = args.output_dir / "profiles"
    figures = args.output_dir / "figures"
    profiles.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (profiles / "ablation-summary.json").write_text(
        json.dumps({"schema_version": 1, "factors": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    for index, row in enumerate(rows, start=1):
        slug = (
            row["factor"].lower().replace(" ", "-").replace("/", "-")
        )
        (profiles / f"{index:02d}-{slug}.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        svg(row, figures / f"{index:02d}-{slug}.svg")


if __name__ == "__main__":
    main()
