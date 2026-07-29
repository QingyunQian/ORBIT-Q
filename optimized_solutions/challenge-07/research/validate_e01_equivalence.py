#!/usr/bin/env python3
"""Compare one Task 07 trajectory through value, gradient, and one Adam update."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "n_data_qubits": 8,
    "n_ancilla_qubits": 8,
    "n_qubits": 16,
    "n_layers": 2,
    "n_trajectories": 64,
    "initial_parameter_scale": 0.1,
    "max_steps": 100,
    "learning_rate": 0.02,
    "seed": 2047,
    "transverse_field": 1.05,
}
TOLERANCES = {
    "energy_abs": 5e-5,
    "gradient_max_abs": 5e-4,
    "adam_parameter_max_abs": 2e-5,
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(module, params, status):
    one_trajectory = module.make_one_trajectory(CONFIG)
    value_and_grad = tc.backend.jit(
        tc.backend.value_and_grad(lambda p: one_trajectory(p, status))
    )
    started = time.perf_counter()
    value, gradient = value_and_grad(params)
    value_np = np.asarray(tc.backend.numpy(value))
    gradient_np = np.asarray(tc.backend.numpy(gradient))
    elapsed = time.perf_counter() - started

    optimizer = optax.adam(CONFIG["learning_rate"])
    state = optimizer.init(params)
    updates, _ = optimizer.update(gradient, state, params)
    next_params = optax.apply_updates(params, updates)
    next_params_np = np.asarray(tc.backend.numpy(next_params))
    return value_np, gradient_np, next_params_np, elapsed


def main():
    reference = load(
        ROOT / "references/task-07/solution_7.py", "task07_reference_equivalence"
    )
    candidate = load(
        ROOT / "src/solutions/task-07/solution_7.py", "task07_candidate_equivalence"
    )
    params = reference.initial_parameters(CONFIG)
    status = reference.trajectory_status(CONFIG)[0]

    rv, rg, rp, rt = evaluate(reference, params, status)
    cv, cg, cp, ct = evaluate(candidate, params, status)
    metrics = {
        "reference_energy": float(rv),
        "candidate_energy": float(cv),
        "energy_abs_error": float(np.abs(rv - cv)),
        "gradient_max_abs_error": float(np.max(np.abs(rg - cg))),
        "gradient_mean_abs_error": float(np.mean(np.abs(rg - cg))),
        "adam_parameter_max_abs_error": float(np.max(np.abs(rp - cp))),
        "adam_parameter_mean_abs_error": float(np.mean(np.abs(rp - cp))),
        "reference_elapsed_sec": rt,
        "candidate_elapsed_sec": ct,
    }
    checks = {
        "energy": metrics["energy_abs_error"] <= TOLERANCES["energy_abs"],
        "gradient": metrics["gradient_max_abs_error"]
        <= TOLERANCES["gradient_max_abs"],
        "one_adam_update": metrics["adam_parameter_max_abs_error"]
        <= TOLERANCES["adam_parameter_max_abs"],
    }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "07",
                "experiment": "e01",
                "trajectory_index": 0,
                "tolerances": TOLERANCES,
                "metrics": metrics,
                "checks": checks,
                "passed": all(checks.values()),
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
