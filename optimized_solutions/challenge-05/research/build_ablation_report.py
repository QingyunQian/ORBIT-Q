#!/usr/bin/env python3
"""Sanitize Task 05 benchmark reports and render dependency-free SVG plots."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


T95 = {5: 2.5705818366147395}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _comparison(
    path: Path,
    *,
    factor: str,
    reference_label: str,
    candidate_label: str,
    decision: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    comparison = report["summary"]["comparison"]["tasks"]["05"]
    pairs = [
        {
            "pair": int(row["pair_id"]),
            "order": row["pair_order"],
            "reference_runtime_sec": float(row["reference_runtime_sec"]),
            "candidate_runtime_sec": float(row["candidate_runtime_sec"]),
            "speedup": float(row["speedup"]),
            "improvement_pct": float(row["improvement_pct"]),
        }
        for row in comparison["pairs"]
    ]
    speedups = [row["speedup"] for row in pairs]
    stderr = float(comparison["speedup_stderr"])
    critical = T95[len(speedups) - 1]
    ci_half = critical * stderr
    rows = report["results"]
    image = rows[0]["environment_image_provenance"]
    role_hashes = {
        str(row["comparison_role"]): str(row["source_sha256"])
        for row in rows
    }
    return {
        "factor": factor,
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "decision": decision,
        "report_sha256": _sha256(path),
        "generated_at": report["generated_at"],
        "host_fingerprint_sha256": report["host"]["fingerprint_sha256"],
        "image": image["reference"],
        "image_id": image["id"],
        "environment": rows[0]["environment"],
        "cpus": 6,
        "memory": "7g",
        "timeout_sec": float(rows[0]["timeout_sec"]),
        "reference_source_sha256": role_hashes["reference"],
        "candidate_source_sha256": role_hashes["candidate"],
        "passing_cells": sum(bool(row["passed"]) for row in rows),
        "cells": len(rows),
        "pairs": pairs,
        "reference_mean_runtime_sec": float(comparison["reference_mean"]),
        "candidate_mean_runtime_sec": float(comparison["candidate_mean"]),
        "mean_paired_speedup": float(comparison["speedup_mean"]),
        "paired_speedup_stderr": stderr,
        "paired_speedup_ci95": [
            float(comparison["speedup_mean"]) - ci_half,
            float(comparison["speedup_mean"]) + ci_half,
        ],
        "paired_improvement_mean_pct": float(
            comparison["paired_improvement_mean"]
        ),
        "first_five": {
            "reference_mean_runtime_sec": _mean(
                [row["reference_runtime_sec"] for row in pairs[:5]]
            ),
            "candidate_mean_runtime_sec": _mean(
                [row["candidate_runtime_sec"] for row in pairs[:5]]
            ),
            "mean_paired_speedup": _mean(speedups[:5]),
        },
    }


def _baseline(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["results"]
    runtimes = [float(row["runtime_sec"]) for row in rows]
    return {
        "report_sha256": _sha256(path),
        "source_sha256": rows[0]["source_sha256"],
        "host_fingerprint_sha256": report["host"]["fingerprint_sha256"],
        "image_id": rows[0]["environment_image_provenance"]["id"],
        "passing_runs": sum(bool(row["passed"]) for row in rows),
        "runtimes_sec": runtimes,
        "mean_runtime_sec": _mean(runtimes),
        "stderr_runtime_sec": statistics.stdev(runtimes)
        / math.sqrt(len(runtimes)),
        "first_five_mean_runtime_sec": _mean(runtimes[:5]),
    }


def _svg(entry: dict[str, Any], path: Path) -> None:
    width, height = 880, 480
    left, right, top, bottom = 86, 836, 70, 382
    values = [float(row["speedup"]) for row in entry["pairs"]]
    ci_low, ci_high = entry["paired_speedup_ci95"]
    low = min([1.0, ci_low, *values])
    high = max([1.0, ci_high, *values])
    padding = max((high - low) * 0.16, 0.025)
    low = max(0.0, low - padding)
    high += padding

    def x(index: int) -> float:
        return left + (right - left) * index / (len(values) - 1)

    def y(value: float) -> float:
        return bottom - (bottom - top) * (value - low) / (high - low)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        (
            f'<text x="{left}" y="30" font-family="system-ui,sans-serif" '
            'font-size="20" font-weight="700" fill="#171923">'
            f'{html.escape(entry["factor"])}</text>'
        ),
        (
            f'<text x="{left}" y="52" font-family="system-ui,sans-serif" '
            'font-size="12" fill="#596275">'
            f'{html.escape(entry["reference_label"])} → '
            f'{html.escape(entry["candidate_label"])}; '
            f'{entry["passing_cells"]}/{entry["cells"]} cells passed</text>'
        ),
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        yy = y(value)
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" '
                    f'y2="{yy:.2f}" stroke="#dfe3eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{yy + 4:.2f}" '
                    'text-anchor="end" font-family="ui-monospace,monospace" '
                    f'font-size="11" fill="#596275">{value:.3f}×</text>'
                ),
            ]
        )
    baseline_y = y(1.0)
    lines.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{right}" '
        f'y2="{baseline_y:.2f}" stroke="#596275" stroke-width="2" '
        'stroke-dasharray="7 6"/>'
    )
    ci_top, ci_bottom = y(ci_high), y(ci_low)
    lines.append(
        f'<rect x="{left}" y="{ci_top:.2f}" width="{right-left}" '
        f'height="{ci_bottom-ci_top:.2f}" fill="#ff7f50" opacity="0.16"/>'
    )
    mean_y = y(entry["mean_paired_speedup"])
    lines.append(
        f'<line x1="{left}" y1="{mean_y:.2f}" x2="{right}" '
        f'y2="{mean_y:.2f}" stroke="#e4572e" stroke-width="3"/>'
    )
    points = " ".join(
        f"{x(index):.2f},{y(value):.2f}"
        for index, value in enumerate(values)
    )
    lines.append(
        f'<polyline points="{points}" fill="none" stroke="#2d6cdf" '
        'stroke-width="2" opacity="0.7"/>'
    )
    for index, value in enumerate(values):
        xx, yy = x(index), y(value)
        lines.extend(
            [
                f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="6" fill="#2d6cdf"/>',
                (
                    f'<text x="{xx:.2f}" y="{bottom + 24}" text-anchor="middle" '
                    'font-family="system-ui,sans-serif" font-size="11" '
                    f'fill="#596275">Pair {index + 1}</text>'
                ),
            ]
        )
    lines.extend(
        [
            (
                f'<text x="{left}" y="{height - 53}" '
                'font-family="system-ui,sans-serif" font-size="13" '
                'font-weight="650" fill="#171923">'
                f'Mean {entry["mean_paired_speedup"]:.4f}×; '
                f'95% t-CI [{ci_low:.4f}×, {ci_high:.4f}×]</text>'
            ),
            (
                f'<text x="{left}" y="{height - 30}" '
                'font-family="system-ui,sans-serif" font-size="11" '
                f'fill="#596275">Decision: {html.escape(entry["decision"])}; '
                f'report SHA-256 {entry["report_sha256"][:16]}…</text>'
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--overall", type=Path, required=True)
    parser.add_argument("--dense-fusion", type=Path, required=True)
    parser.add_argument("--mps-fusion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    entries = [
        _comparison(
            args.overall,
            factor="Overall: exact bounded-rank MPS",
            reference_label="Immutable dense expert",
            candidate_label="Exact no-QR MPS",
            decision="keep and promote",
        ),
        _comparison(
            args.dense_fusion,
            factor="Ablation: dense layer gate fusion",
            reference_label="Accepted dense parent",
            candidate_label="Fused dense filters",
            decision="discard; 95% CI crosses 1x",
        ),
        _comparison(
            args.mps_fusion,
            factor="Ablation: absorb RX into rank-2 MPS MPO",
            reference_label="Exact no-QR MPS",
            candidate_label="Fused MPS filters",
            decision="discard; consistent regression",
        ),
    ]
    payload = {
        "schema_version": 1,
        "task_id": "05",
        "protocol": {
            "pairs": 6,
            "order": "alternating matched pairs in one container",
            "timing": "evaluator-reported run_solution wall time",
            "confidence_interval": "two-sided 95% Student-t, df=5",
        },
        "reference_baseline": _baseline(args.baseline),
        "comparisons": entries,
        "unmeasured_factor": {
            "factor": "exact MPS versus accepted dense parent",
            "reason": (
                "The final Docker approval disconnected and explicitly "
                "prohibited an automatic retry. No paired value is inferred."
            ),
        },
    }
    output = args.output_dir / "ablation-summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    names = [
        "overall-exact-mps.svg",
        "factor-dense-layer-fusion.svg",
        "factor-mps-local-fusion.svg",
    ]
    for entry, name in zip(entries, names):
        _svg(entry, figures / name)


if __name__ == "__main__":
    main()
