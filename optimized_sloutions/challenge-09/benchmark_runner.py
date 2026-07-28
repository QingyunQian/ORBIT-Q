"""Alternate official-evaluator benchmarks for challenge 09."""

from __future__ import annotations

import importlib.util
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "tasks" / "challenge-09" / "tests" / "evaluate_9.py"
REFERENCE = ROOT / "tasks" / "challenge-09" / "solution" / "solution_9.py"
CANDIDATE = Path(__file__).resolve().parent / "solution_9_cones.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_once(solution_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(EVALUATOR, tmp_path / "evaluate_9.py")
        shutil.copy(solution_path, tmp_path / "solution_9.py")
        sys.path.insert(0, str(tmp_path))
        try:
            if "evaluate_9" in sys.modules:
                del sys.modules["evaluate_9"]
            if "solution_9" in sys.modules:
                del sys.modules["solution_9"]
            evaluate_9 = _load(tmp_path / "evaluate_9.py", "evaluate_9_bench")
            config = evaluate_9.finalize_config(dict(evaluate_9.DEFAULT_CONFIG))
            # Capture printed metrics by re-running evaluate internals
            module = _load(tmp_path / "solution_9.py", "solution_9_bench")
            start = time.perf_counter()
            results = module.run_solution(config)
            elapsed = time.perf_counter() - start
            history = __import__("numpy").asarray(results["observable_history"], float)
            final = history[:, -1]
            initial = history[:, 0]
            return {
                "runtime_sec": elapsed,
                "mean_initial": float(initial.mean()),
                "mean_final": float(final.mean()),
                "best_final": float(final.max()),
                "success_fraction": float((final >= config["success_threshold"]).mean()),
                "pass": bool(
                    history.shape == (config["n_restarts"], config["max_steps"])
                    and final.mean() > initial.mean()
                    and final.max() >= config["success_threshold"]
                ),
            }
        finally:
            sys.path.pop(0)


def main(repeats: int = 5) -> None:
    rows = []
    for i in range(repeats):
        for label, path in (("reference", REFERENCE), ("cones", CANDIDATE)):
            print(f"[{i+1}/{repeats}] {label}", flush=True)
            row = run_once(path)
            row.update({"trial": i + 1, "solution": label})
            rows.append(row)
            print(
                f"  runtime={row['runtime_sec']:.3f}s pass={row['pass']} "
                f"mean_final={row['mean_final']:.6f}",
                flush=True,
            )
    summary = {}
    for label in ("reference", "cones"):
        times = [r["runtime_sec"] for r in rows if r["solution"] == label]
        summary[label] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "times": times,
            "pass": all(r["pass"] for r in rows if r["solution"] == label),
        }
    summary["speedup"] = summary["reference"]["mean"] / summary["cones"]["mean"]
    out = {"rows": rows, "summary": summary}
    out_path = Path(__file__).resolve().parent / "benchmark_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
