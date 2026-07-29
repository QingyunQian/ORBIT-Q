#!/usr/bin/env python3
"""Render deterministic, dependency-free SVGs from sanitized Task 04 data."""

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
LIMIT = "#d89b2b"


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
    low, high = factor["paired_speedup_ci95"]
    values = [1.0, low, high, *speedups]
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
    for index, value in enumerate(speedups, start=1):
        y = 106 + (index - 1) * 31
        lines.extend(
            [
                f'  <text x="70" y="{y + 4}" font-size="12" text-anchor="end" class="muted">pair {index}</text>',
                f'  <circle cx="{x(value):.2f}" cy="{y}" r="5.5" fill="{POINT}"/>',
                f'  <text x="{x(value):.2f}" y="{y - 10}" font-size="11" text-anchor="middle">{value:.3f}×</text>',
            ]
        )
    lines.extend(
        [
            f'  <line x1="{x(low):.2f}" y1="292" x2="{x(high):.2f}" y2="292" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',
            f'  <line x1="{x(low):.2f}" y1="284" x2="{x(low):.2f}" y2="300" stroke="{color}" stroke-width="2"/>',
            f'  <line x1="{x(high):.2f}" y1="284" x2="{x(high):.2f}" y2="300" stroke="{color}" stroke-width="2"/>',
            f'  <polygon points="{x(mean):.2f},281 {x(mean) + 9:.2f},292 {x(mean):.2f},303 {x(mean) - 9:.2f},292" fill="{color}"/>',
            f'  <text x="38" y="374" font-size="13">mean {mean:.4f}× · 95% CI [{low:.4f}, {high:.4f}] · wins {factor["pair_wins"]}/{factor["pair_count"]}</text>',
        ]
    )
    write_svg(
        FIGURE_DIR / f'{factor["id"]}-{factor["decision"]}.svg',
        "\n".join(lines),
        f'{factor["id"].upper()} {factor["name"]}',
        "Six paired speedups, the no-change line, and the mean 95 percent confidence interval.",
    )


def memory_figure(factor: dict[str, object]) -> None:
    limit = factor["memory_limit_bytes"] / 1e9
    requested = factor["failed_allocation_bytes"] / 1e9
    scale = 580 / (requested * 1.12)
    body = "\n".join(
        [
            f'  <text x="38" y="42" font-size="22" font-weight="600">{esc(factor["id"].upper())}: {esc(factor["name"])}</text>',
            f'  <text x="38" y="70" font-size="14" class="muted">Parent: {esc(factor["parent"])} · decision: discard (resource gate)</text>',
            f'  <text x="38" y="137" font-size="13">Frozen memory limit</text>',
            f'  <rect x="145" y="112" width="{limit * scale:.2f}" height="36" rx="4" fill="{LIMIT}"/>',
            f'  <text x="{155 + limit * scale:.2f}" y="136" font-size="13">{limit:.2f} GB</text>',
            f'  <text x="38" y="222" font-size="13">Failed allocation</text>',
            f'  <rect x="145" y="197" width="{requested * scale:.2f}" height="36" rx="4" fill="{DISCARD}"/>',
            f'  <text x="{155 + requested * scale:.2f}" y="221" font-size="13">{requested:.2f} GB</text>',
            f'  <line x1="{145 + limit * scale:.2f}" y1="96" x2="{145 + limit * scale:.2f}" y2="254" stroke="{LIMIT}" stroke-width="2" stroke-dasharray="5 5"/>',
            f'  <text x="38" y="305" font-size="15">reuse=True exceeded the limit by {requested - limit:.2f} GB before a valid runtime existed.</text>',
            '  <text x="38" y="336" font-size="13" class="muted">No speedup is plotted because the canonical evaluator did not pass.</text>',
        ]
    )
    write_svg(
        FIGURE_DIR / f'{factor["id"]}-discard.svg',
        body,
        f'{factor["id"].upper()} {factor["name"]}',
        "The failed allocation exceeds the frozen seven GiB container limit.",
    )


def final_figure(final: dict[str, object]) -> None:
    refs = final["reference_times_sec"]
    cands = final["candidate_times_sec"]
    y_top, y_bottom = 96.0, 310.0
    y_max = max(refs) * 1.08

    def y(value: float) -> float:
        return y_bottom - value / y_max * (y_bottom - y_top)

    def x(index: int) -> float:
        return 105 + index * 96

    lines = [
        '  <text x="38" y="42" font-size="22" font-weight="600">Final cumulative comparison</text>',
        f'  <text x="38" y="70" font-size="14" class="muted">Six alternating matched pairs · mean speedup {final["paired_speedup_mean"]:.4f}×</text>',
    ]
    for tick in (0, 5, 10, 15):
        lines.extend(
            [
                f'  <line x1="78" y1="{y(tick):.2f}" x2="708" y2="{y(tick):.2f}" class="grid"/>',
                f'  <text x="66" y="{y(tick) + 4:.2f}" font-size="12" text-anchor="end" class="muted">{tick}s</text>',
            ]
        )
    for index, (ref, cand) in enumerate(zip(refs, cands), start=1):
        cx = x(index - 1)
        lines.extend(
            [
                f'  <line x1="{cx}" y1="{y(ref):.2f}" x2="{cx}" y2="{y(cand):.2f}" stroke="{GRID}" stroke-width="5"/>',
                f'  <circle cx="{cx}" cy="{y(ref):.2f}" r="6" fill="{DISCARD}"/>',
                f'  <circle cx="{cx}" cy="{y(cand):.2f}" r="6" fill="{KEEP}"/>',
                f'  <text x="{cx}" y="333" font-size="12" text-anchor="middle">pair {index}</text>',
                f'  <text x="{cx}" y="{y(ref) - 10:.2f}" font-size="11" text-anchor="middle">{ref:.2f}</text>',
                f'  <text x="{cx}" y="{y(cand) + 18:.2f}" font-size="11" text-anchor="middle">{cand:.2f}</text>',
            ]
        )
    lines.extend(
        [
            f'  <circle cx="490" cy="365" r="5" fill="{DISCARD}"/>',
            '  <text x="502" y="369" font-size="12">expert</text>',
            f'  <circle cx="570" cy="365" r="5" fill="{KEEP}"/>',
            '  <text x="582" y="369" font-size="12">campaign best</text>',
            f'  <text x="38" y="369" font-size="13">95% CI [{final["paired_speedup_ci95"][0]:.4f}, {final["paired_speedup_ci95"][1]:.4f}]×</text>',
        ]
    )
    write_svg(
        FIGURE_DIR / "final-cumulative.svg",
        "\n".join(lines),
        "Final cumulative Task 04 comparison",
        "Each vertical lane compares the immutable expert and final candidate runtime for one matched pair.",
    )


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    FIGURE_DIR.mkdir(exist_ok=True)
    for factor in payload["factors"]:
        if factor.get("failure") == "resource_gate":
            memory_figure(factor)
        else:
            speedup_figure(factor)
    final_figure(payload["final_comparison"])


if __name__ == "__main__":
    main()
