#!/usr/bin/env python3
"""Audit the batched exact Renyi-2 kernel for Task 02."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "task-02" / "solution_2.py"
CANDIDATE = ROOT / "src" / "solutions" / "task-02" / "solution_2.py"
EVALUATOR = ROOT / "tasks" / "task-02" / "evaluator" / "evaluate_2.py"
K = tc.set_backend("jax")
tc.set_dtype("complex64")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_error(a, b):
    aa = K.tree_flatten(a)[0]
    bb = K.tree_flatten(b)[0]
    return max(
        float(np.max(np.abs(K.numpy(x) - K.numpy(y))))
        for x, y in zip(aa, bb)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load(REFERENCE, "task02_reference_batched_purity_audit")
    candidate = load(CANDIDATE, "task02_candidate_batched_purity_audit")
    evaluator = load(EVALUATOR, "task02_evaluator_batched_purity_audit")
    config = dict(evaluator.DEFAULT_CONFIG)
    config["target_entropies"] = np.asarray(
        config["target_entropies"], dtype=np.float32
    )
    params = reference.initial_parameters(config)
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(1, config["n_qubits"], 2):
        circuit.x(i)
    input_state = circuit.state()
    target = K.convert_to_tensor(config["target_entropies"])
    href = reference.build_xxz_mvp(config)
    hcand = candidate.build_xxz_mvp(config)

    def ref_loss(p):
        return reference.observables(p, input_state, href, config, target)

    def cand_loss(p):
        return candidate.observables(p, input_state, hcand, config, target)

    (ref_loss_value, ref_aux), ref_grad = K.value_and_grad(
        ref_loss, has_aux=True
    )(params)
    (cand_loss_value, cand_aux), cand_grad = K.value_and_grad(
        cand_loss, has_aux=True
    )(params)
    config["max_steps"] = args.steps
    expected = reference.run_solution(config)
    actual = candidate.run_solution(config)
    history_errors = {
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
    report = {
        "schema_version": 1,
        "task_id": "02",
        "steps": args.steps,
        "loss_abs_error": float(
            abs(K.numpy(ref_loss_value - cand_loss_value))
        ),
        "aux_max_error": tree_error(ref_aux, cand_aux),
        "gradient_max_error": tree_error(ref_grad, cand_grad),
        "history_max_abs_errors": history_errors,
    }
    report["passed"] = (
        report["loss_abs_error"] <= 2e-6
        and report["aux_max_error"] <= 2e-6
        and report["gradient_max_error"] <= 1e-5
        and max(history_errors.values()) <= 2e-4
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
