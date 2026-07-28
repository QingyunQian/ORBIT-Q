"""Cold-start benchmark: challenge-11 reference vs optimized solution.

Runs the official evaluator in a fresh process per trial (full 500 steps),
alternating between the solutions, and writes per-trial times plus summary
statistics to benchmark_results.json.
"""

import json
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALUATOR = REPO / "tasks/challenge-11/tests/evaluate_11.py"
PYTHON = REPO / ".venv-c12/bin/python"
HERE = Path(__file__).resolve().parent

SOLUTIONS = {
    "reference": str(REPO / "tasks/challenge-11/solution"),
    "optimized_fused": str(HERE),
}
MODULES = {"reference": "solution_11", "optimized_fused": "solution_11_fused"}

TIME_RE = re.compile(r"End-to-end solution time: ([0-9.]+)s")
GAP_RE = re.compile(r"Energy-density gap: ([0-9.e+-]+)")
MAE_RE = re.compile(r"String-order MAE: ([0-9.e+-]+)")


def run_once(kind):
    env = {
        "PYTHONPATH": SOLUTIONS[kind],
        "NUMBA_DISABLE_JIT": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
    }
    proc = subprocess.run(
        [str(PYTHON), str(EVALUATOR), "--solution", MODULES[kind]],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    out = proc.stdout
    assert "Overall: PASS" in out, f"{kind} FAILED:\n{out}\n{proc.stderr}"
    return {
        "time_sec": float(TIME_RE.search(out).group(1)),
        "energy_gap": float(GAP_RE.search(out).group(1)),
        "string_mae": float(MAE_RE.search(out).group(1)),
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    trials = {k: [] for k in SOLUTIONS}
    for i in range(n_trials):
        for kind in SOLUTIONS:
            r = run_once(kind)
            trials[kind].append(r)
            print(
                f"trial {i+1}/{n_trials} {kind}: {r['time_sec']:.2f}s "
                f"gap {r['energy_gap']:.4f} mae {r['string_mae']:.4f}",
                flush=True,
            )

    summary = {}
    for kind, rows in trials.items():
        times = [r["time_sec"] for r in rows]
        summary[kind] = {
            "times_sec": times,
            "mean_sec": round(statistics.mean(times), 3),
            "stdev_sec": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
            "min_sec": min(times),
            "max_sec": max(times),
            "energy_gaps": [r["energy_gap"] for r in rows],
            "string_maes": [r["string_mae"] for r in rows],
        }
    speedup = summary["reference"]["mean_sec"] / summary["optimized_fused"]["mean_sec"]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu_model": _cpu_model(),
        "python": platform.python_version(),
        "protocol": "official evaluate_11.py, 500 steps, fresh process per trial, alternating",
        "trials": summary,
        "mean_speedup_reference_over_optimized": round(speedup, 3),
    }
    (HERE / "benchmark_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    main()
