"""Cold-start benchmark: challenge-12 reference vs optimized solution.

Runs the official evaluator in a fresh process per trial (full 5000 steps),
alternating between the two solutions to spread machine drift, and writes
per-trial times plus summary statistics to benchmark_results.json.
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
EVALUATOR = REPO / "tasks/challenge-12/tests/evaluate_12.py"
PYTHON = REPO / ".venv-c12/bin/python"
HERE = Path(__file__).resolve().parent

SOLUTIONS = {
    "reference": str(REPO / "tasks/challenge-12/solution"),
    "optimized_batched": str(HERE),
    "optimized_fused": str(HERE),
}
MODULES = {
    "reference": "solution_12",
    "optimized_batched": "solution_12_batched",
    "optimized_fused": "solution_12_fused",
}

TIME_RE = re.compile(r"End-to-end solution time: ([0-9.]+)s")
FID_RE = re.compile(r"Final fidelity: ([0-9.e+-]+)")


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
        "final_fidelity": float(FID_RE.search(out).group(1)),
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
                f"fid {r['final_fidelity']:.6f}",
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
            "final_fidelities": [r["final_fidelity"] for r in rows],
        }
    speedups = {
        kind: round(summary["reference"]["mean_sec"] / summary[kind]["mean_sec"], 3)
        for kind in SOLUTIONS
        if kind != "reference"
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_model": _cpu_model(),
        "python": platform.python_version(),
        "protocol": "official evaluate_12.py, 5000 steps, fresh process per trial, alternating",
        "trials": summary,
        "mean_speedup_vs_reference": speedups,
    }
    out_path = HERE / "benchmark_results.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
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
