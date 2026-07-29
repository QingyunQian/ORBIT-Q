#!/usr/bin/env python3
"""Phase-split profiler for the immutable Task 10 expert objective."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import jax
import optax


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "task-10" / "solution_10.py"
EVALUATOR = ROOT / "tasks" / "task-10" / "evaluator" / "evaluate_10.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_tree(tree) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-steps", type=int, default=20)
    args = parser.parse_args()

    solution = load_module("task10_reference_profile", REFERENCE)
    evaluator = load_module("task10_evaluator_profile", EVALUATOR)
    config = dict(evaluator.DEFAULT_CONFIG)
    config["selected_qubits"] = list(config["selected_qubits"])
    config["initial_ones"] = list(config["initial_ones"])

    start = time.perf_counter()
    params = solution.initial_parameters(config)
    params.block_until_ready()
    parameter_setup_sec = time.perf_counter() - start

    start = time.perf_counter()
    mpo = solution.build_tfim_mpo(config)
    mpo_setup_sec = time.perf_counter() - start

    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return solution.energy_density(p, mpo, config)

    def step(p, state):
        value, grads = solution.K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, value

    jitted = solution.K.jit(step)

    start = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    lower_sec = time.perf_counter() - start
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - start

    start = time.perf_counter()
    current = compiled(params, opt_state)
    block_tree(current)
    first_execution_sec = time.perf_counter() - start

    current_params, current_state, _ = current
    start = time.perf_counter()
    for _ in range(args.steady_steps):
        current = compiled(current_params, current_state)
        current_params, current_state, _ = current
    block_tree(current)
    steady_total_sec = time.perf_counter() - start

    start = time.perf_counter()
    results = solution.run_solution(config)
    full_run_solution_sec = time.perf_counter() - start

    payload = {
        "generated_by": "research/task-10/profile_reference.py",
        "reference_sha256": (
            "0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172"
        ),
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "profile": {
            "parameter_setup_sec": parameter_setup_sec,
            "mpo_setup_sec": mpo_setup_sec,
            "jit_lower_sec": lower_sec,
            "xla_compile_sec": compile_sec,
            "stablehlo_line_count": len(stablehlo.splitlines()),
            "first_execution_sec": first_execution_sec,
            "steady_steps": args.steady_steps,
            "steady_total_sec": steady_total_sec,
            "steady_step_ms": 1000.0 * steady_total_sec / args.steady_steps,
            "full_run_solution_sec": full_run_solution_sec,
            "initial_energy_density": float(results["energy_history"][0]),
            "final_energy_density": float(results["energy_history"][-1]),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
