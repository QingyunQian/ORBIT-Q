#!/usr/bin/env python3
"""Profile the final Task 01 candidate without changing its computation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import optax


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronized(function):
    start = time.perf_counter()
    value = function()
    leaves = value if isinstance(value, tuple) else (value,)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value, time.perf_counter() - start


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steady-repeats", type=int, default=5)
    args = parser.parse_args()

    source_path = ROOT / "src" / "solutions" / "task-01" / "solution_1.py"
    solution = load_module(source_path, "task01_candidate_profile")
    evaluator = load_module(
        ROOT / "tasks" / "task-01" / "evaluator" / "evaluate_1.py",
        "task01_candidate_evaluator_profile",
    )
    config = dict(evaluator.DEFAULT_CONFIG)
    dmrg_start = time.perf_counter()
    dmrg_state, dmrg_energy = evaluator.dmrg_initial_state(config)
    dmrg_seconds = time.perf_counter() - dmrg_start
    config["dmrg_state"] = dmrg_state
    config["dmrg_energy"] = dmrg_energy

    K = solution.K
    mps_input = solution.tc.quantum.quimb2qop(dmrg_state)
    mpo = solution.tfim_mpo(config)
    params = solution.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def energy_fn(p):
        return solution.circuit_energy(p, mps_input, config, mpo)

    value_and_grad = K.value_and_grad(energy_fn)

    def train_step(p, state):
        energy, grads = value_and_grad(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, energy

    compiled_step = K.jit(train_step)
    lower_start = time.perf_counter()
    lowered = compiled_step.lower(params, opt_state)
    lower_seconds = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    first, first_seconds = synchronized(lambda: executable(params, opt_state))
    params, opt_state = first[0], first[1]

    steady_seconds = []
    for _ in range(args.steady_repeats):
        result, elapsed = synchronized(lambda: executable(params, opt_state))
        params, opt_state = result[0], result[1]
        steady_seconds.append(elapsed)

    cost = executable.cost_analysis()
    cost = {
        key: cost[key]
        for key in ("flops", "bytes accessed", "transcendentals")
        if key in cost
    }
    payload = {
        "schema_version": 1,
        "task_id": "01",
        "solution_path": str(source_path.relative_to(ROOT)),
        "solution_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "configuration": dict(evaluator.DEFAULT_CONFIG),
        "structural_counts": {
            "parameters": solution.parameter_count(config),
            "fused_two_qubit_gate_nodes": 62,
            "fused_boundary_one_qubit_gate_nodes": 4,
            "tfim_mpo_bond_dimension": 3,
            "training_steps": 500,
        },
        "evaluator_side_dmrg_seconds_excluded_from_runtime": dmrg_seconds,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_compiled_step_seconds": first_seconds,
        "steady_step_seconds": steady_seconds,
        "steady_step_mean_seconds": statistics.mean(steady_seconds),
        "steady_step_median_seconds": statistics.median(steady_seconds),
        "projected_500_step_seconds": 500 * statistics.mean(steady_seconds),
        "stablehlo_line_count": len(lowered.as_text().splitlines()),
        "cost_analysis": jsonable(cost),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
