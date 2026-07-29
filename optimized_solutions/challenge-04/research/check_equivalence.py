#!/usr/bin/env python3
"""Compare the Task 04 reference and current candidate on exact diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnostics(module, config):
    K = module.K
    true_p01 = K.convert_to_tensor(config["true_p01"])
    true_p10 = K.convert_to_tensor(config["true_p10"])
    if hasattr(module, "probe_states"):
        initial_states = module.probe_states(config)
        target = module.observable_table(
            true_p01, true_p10, initial_states, config
        )
    else:
        initial_states = None
        target = module.observable_table(true_p01, true_p10, config)
    params = module.initial_parameters(config)

    def loss_fn(p):
        if initial_states is None:
            return module.loss_and_observables(p, target, config)
        return module.loss_and_observables(
            p, target, initial_states, config
        )

    (loss, aux), grads = K.value_and_grad(loss_fn, has_aux=True)(params)
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    next_params = optax.apply_updates(params, updates)
    full_result = module.run_solution(config)
    return {
        "target": np.asarray(K.numpy(target)),
        "loss": np.asarray(K.numpy(loss)),
        "observables": np.asarray(K.numpy(aux[2])),
        "grads": np.asarray(K.numpy(grads)),
        "next_params": np.asarray(K.numpy(next_params)),
        "loss_history": np.asarray(full_result["loss_history"]),
        "final_probabilities": np.asarray(
            full_result["final_probabilities"]
        ),
        "fitted_expectations": np.asarray(
            full_result["fitted_expectations"]
        ),
    }


def max_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_module(
        ROOT / "references" / "task-04" / "solution_4.py",
        "task04_equivalence_reference",
    )
    candidate = load_module(
        ROOT / "src" / "solutions" / "task-04" / "solution_4.py",
        "task04_equivalence_candidate",
    )
    config = {
        "n_qubits": 4,
        "entangler_angle": 0.31,
        "true_p01": 0.034,
        "true_p10": 0.011,
        "initial_p01": 0.070,
        "initial_p10": 0.040,
        "max_steps": 3,
        "learning_rate": 0.04,
        "probability_absolute_tolerance": 2e-4,
    }
    ref = diagnostics(reference, config)
    cand = diagnostics(candidate, config)
    errors = {key: max_error(ref[key], cand[key]) for key in ref}
    threshold = 2e-6
    payload = {
        "schema_version": 1,
        "task_id": "04",
        "configuration": config,
        "errors": errors,
        "threshold": threshold,
        "passed": all(error <= threshold for error in errors.values()),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
