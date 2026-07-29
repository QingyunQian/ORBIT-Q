#!/usr/bin/env python3
"""Compare Task 02 Python-dispatched and K.jaxy_scan training histories."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "task-02" / "solution_2.py"
CANDIDATE = ROOT / "src" / "solutions" / "task-02" / "solution_2.py"
EVALUATOR = ROOT / "tasks" / "task-02" / "evaluator" / "evaluate_2.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load(REFERENCE, "task02_reference_scan_audit")
    candidate = load(CANDIDATE, "task02_candidate_scan_audit")
    evaluator = load(EVALUATOR, "task02_evaluator_scan_audit")
    config = dict(evaluator.DEFAULT_CONFIG)
    config["target_entropies"] = np.asarray(
        config["target_entropies"], dtype=np.float32
    )
    config["max_steps"] = args.steps

    expected = reference.run_solution(config)
    actual = candidate.run_solution(config)
    errors = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(expected[key], dtype=float)
                    - np.asarray(actual[key], dtype=float)
                )
            )
        )
        for key in sorted(expected)
    }
    shapes_match = {
        key: np.shape(expected[key]) == np.shape(actual[key])
        for key in sorted(expected)
    }
    report = {
        "schema_version": 1,
        "task_id": "02",
        "steps": args.steps,
        "max_abs_errors": errors,
        "shapes_match": shapes_match,
        "passed": all(shapes_match.values()) and max(errors.values()) <= 2e-4,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
