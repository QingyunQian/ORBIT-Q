#!/usr/bin/env python3
"""Render deterministic, dependency-free SVGs from sanitized Task 01 data."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results-20260729" / "ablation-summary.json"
FIGURE_DIR = ROOT / "figures"

INK = "#172033"
MUTED = "#5c667a"
GRID = "#d9deea"
KEEP = "#138a72"
DISCARD = "#c14b4b"
POINT = "#3e67b1"


def esc(value: object) -> str:
    return html.escape(str(value))


def write_svg(path: Path, body: str, title: str, description: str) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(description)}</desc>
  <rect width="760" height="390" fill="#ffffff"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {INK}; }}
    .muted {{ fill: {MUTED}; }}
    .grid {{ stroke: {GRID}; stroke-width: 1; }}
  </style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def speedup_figure(factor: dict[str, object]) -> None:
    speedups = factor["paired_speedups"]
    mean = factor["paired_speedup_mean"]
    interval = factor["paired_speedup_ci95"]
    bounds = interval if interval is not None else [mean, mean]
    values = [1.0, *bounds, *speedups]
    padding = max(0.04, (max(values) - min(values)) * 0.18)
    x_min = min(values) - padding
    x_max = max(values) + padding
    left, right = 92.0, 706.0

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    color = KEEP if factor["decision"] == "keep" else DISCARD
    lines = [
        f'  <text x="38" y="42" font-size="22" font-weight="600">{esc(factor["id"].upper())}: {esc(factor["name"])}</text>',
        f'  <text x="38" y="70" font-size="14" class="muted">Parent: {esc(factor["parent"])} · decision: {esc(factor["decision"])}</text>',
        f'  <line x1="{left}" y1="316" x2="{right}" y2="316" class="grid"/>',
        f'  <line x1="{x(1.0):.2f}" y1="94" x2="{x(1.0):.2f}" y2="326" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"/>',
        f'  <text x="{x(1.0):.2f}" y="344" font-size="12" text-anchor="middle" class="muted">1.0× no change</text>',
    ]
    spacing = min(31, 172 / max(1, len(speedups) - 1)) if len(speedups) > 1 else 0
    for index, value in enumerate(speedups, start=1):
        y = 106 + (index - 1) * spacing
        lines.extend(
            [
                f'  <text x="70" y="{y + 4:.2f}" font-size="12" text-anchor="end" class="muted">pair {index}</text>',
                f'  <circle cx="{x(value):.2f}" cy="{y:.2f}" r="5.5" fill="{POINT}"/>',
                f'  <text x="{x(value):.2f}" y="{y - 10:.2f}" font-size="11" text-anchor="middle">{value:.3f}×</text>',
            ]
        )
    if interval is not None:
        low, high = interval
        lines.extend(
            [
                f'  <line x1="{x(low):.2f}" y1="292" x2="{x(high):.2f}" y2="292" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',
                f'  <line x1="{x(low):.2f}" y1="284" x2="{x(low):.2f}" y2="300" stroke="{color}" stroke-width="2"/>',
                f'  <line x1="{x(high):.2f}" y1="284" x2="{x(high):.2f}" y2="300" stroke="{color}" stroke-width="2"/>',
                f'  <polygon points="{x(mean):.2f},281 {x(mean) + 9:.2f},292 {x(mean):.2f},303 {x(mean) - 9:.2f},292" fill="{color}"/>',
                f'  <text x="38" y="374" font-size="13">mean {mean:.4f}× · 95% CI [{low:.4f}, {high:.4f}] · wins {factor["pair_wins"]}/{factor["pair_count"]}</text>',
            ]
        )
        description = "Paired speedups, no-change line, and mean 95 percent confidence interval."
    else:
        lines.extend(
            [
                f'  <polygon points="{x(mean):.2f},281 {x(mean) + 9:.2f},292 {x(mean):.2f},303 {x(mean) - 9:.2f},292" fill="{color}"/>',
                f'  <text x="38" y="374" font-size="13">screen {mean:.4f}× · one pair only · no confidence interval · wins {factor["pair_wins"]}/1</text>',
            ]
        )
        description = "One paired screening result; no confidence interval is claimed."
    write_svg(
        FIGURE_DIR / f'{factor["id"]}-{factor["decision"]}.svg',
        "\n".join(lines),
        f'{factor["id"].upper()} {factor["name"]}',
        description,
    )


def final_figure(final: dict[str, object]) -> None:
    refs = final["reference_times_sec"]
    candidates = final["candidate_times_sec"]
    y_top, y_bottom = 96.0, 310.0
    y_max = max(refs) * 1.08

    def y(value: float) -> float:
        return y_bottom - value / y_max * (y_bottom - y_top)

    def x(index: int) -> float:
        return 105 + index * 96

    lines = [
        '  <text x="38" y="42" font-size="22" font-weight="600">Final cumulative Task 01 comparison</text>',
        f'  <text x="38" y="70" font-size="14" class="muted">Six alternating matched pairs · mean speedup {final["paired_speedup_mean"]:.4f}×</text>',
    ]
    for tick in (0, 20, 40, 60):
        lines.extend(
            [
                f'  <line x1="78" y1="{y(tick):.2f}" x2="708" y2="{y(tick):.2f}" class="grid"/>',
                f'  <text x="66" y="{y(tick) + 4:.2f}" font-size="12" text-anchor="end" class="muted">{tick}s</text>',
            ]
        )
    for index, (reference, candidate) in enumerate(zip(refs, candidates), start=1):
        cx = x(index - 1)
        lines.extend(
            [
                f'  <line x1="{cx}" y1="{y(reference):.2f}" x2="{cx}" y2="{y(candidate):.2f}" stroke="{GRID}" stroke-width="5"/>',
                f'  <circle cx="{cx}" cy="{y(reference):.2f}" r="6" fill="{DISCARD}"/>',
                f'  <circle cx="{cx}" cy="{y(candidate):.2f}" r="6" fill="{KEEP}"/>',
                f'  <text x="{cx}" y="333" font-size="12" text-anchor="middle">pair {index}</text>',
                f'  <text x="{cx}" y="{y(reference) - 10:.2f}" font-size="11" text-anchor="middle">{reference:.2f}</text>',
                f'  <text x="{cx}" y="{y(candidate) + 18:.2f}" font-size="11" text-anchor="middle">{candidate:.2f}</text>',
            ]
        )
    low, high = final["paired_speedup_ci95"]
    lines.extend(
        [
            f'  <circle cx="490" cy="365" r="5" fill="{DISCARD}"/>',
            '  <text x="502" y="369" font-size="12">human expert</text>',
            f'  <circle cx="600" cy="365" r="5" fill="{KEEP}"/>',
            '  <text x="612" y="369" font-size="12">campaign best</text>',
            f'  <text x="38" y="369" font-size="13">95% CI [{low:.4f}, {high:.4f}]×</text>',
        ]
    )
    write_svg(
        FIGURE_DIR / "final-cumulative.svg",
        "\n".join(lines),
        "Final cumulative Task 01 comparison",
        "Each lane compares immutable expert and final candidate runtime for one matched pair.",
    )


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    FIGURE_DIR.mkdir(exist_ok=True)
    for factor in payload["factors"]:
        speedup_figure(factor)
    final_figure(payload["final_comparison"])


if __name__ == "__main__":
    main()
