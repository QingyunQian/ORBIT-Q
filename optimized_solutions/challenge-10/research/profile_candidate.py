#!/usr/bin/env python3
"""Phase-split profiler for the editable Task 10 candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import jax
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "src" / "solutions" / "task-10" / "solution_10.py"
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
    parser.add_argument("--steady-runs", type=int, default=5)
    args = parser.parse_args()

    solution = load_module("task10_candidate_profile", CANDIDATE)
    evaluator = load_module("task10_evaluator_profile", EVALUATOR)
    config = dict(evaluator.DEFAULT_CONFIG)
    config["selected_qubits"] = list(config["selected_qubits"])
    config["initial_ones"] = list(config["initial_ones"])

    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])

    start = time.perf_counter()
    initial_mps = solution._initial_mps(n_qubits, config["initial_ones"])
    cmz_mpo = solution._cmz_mpo(n_qubits, config["selected_qubits"])
    tfim_mpo = solution._tfim_mpo(
        n_qubits,
        float(config["zz_strength"]),
        float(config["x_strength"]),
    )
    tensor_setup_sec = time.perf_counter() - start

    def energy_density(params):
        circuit = tc.MPSCircuit(
            n_qubits,
            tensors=initial_mps,
            center_position=0,
        )
        tensors = circuit.get_tensors()
        for layer in range(n_layers):
            tensors = [
                solution._apply_one_site(
                    tensor,
                    solution._rotation(params[layer, q]),
                )
                for q, tensor in enumerate(tensors)
            ]
            tensors = solution._apply_mpo(tensors, cmz_mpo)
        circuit = tc.MPSCircuit(
            n_qubits,
            tensors=tensors,
            center_position=0,
        )
        return solution._mpo_expectation(circuit.get_tensors(), tfim_mpo) / n_qubits

    params = solution.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train(carry, _):
        current, state = carry
        value, grads = solution.K.value_and_grad(energy_density)(current)
        updates, state = optimizer.update(grads, state, current)
        return (optax.apply_updates(current, updates), state), value

    def optimize(current, state):
        return solution.K.jaxy_scan(
            train,
            (current, state),
            solution.K.arange(max_steps),
        )

    jitted = solution.K.jit(optimize)

    start = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    lower_sec = time.perf_counter() - start
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - start

    start = time.perf_counter()
    first = compiled(params, opt_state)
    block_tree(first)
    first_execution_sec = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(args.steady_runs):
        current = compiled(params, opt_state)
        block_tree(current)
    steady_total_sec = time.perf_counter() - start

    start = time.perf_counter()
    results = solution.run_solution(config)
    full_run_solution_after_profile_sec = time.perf_counter() - start

    payload = {
        "generated_by": "research/task-10/profile_candidate.py",
        "candidate_sha256": (
            "ad3154ccdfec1a329493e1dc7bbe6e3d30ee4e5d0cc7ac16c9922c57976d1262"
        ),
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "profile": {
            "tensor_setup_sec": tensor_setup_sec,
            "jit_lower_sec": lower_sec,
            "xla_compile_sec": compile_sec,
            "stablehlo_line_count": len(stablehlo.splitlines()),
            "first_execution_sec": first_execution_sec,
            "steady_full_training_runs": args.steady_runs,
            "steady_full_training_total_sec": steady_total_sec,
            "steady_full_training_ms": (
                1000.0 * steady_total_sec / args.steady_runs
            ),
            "full_run_solution_after_profile_sec": (
                full_run_solution_after_profile_sec
            ),
            "initial_energy_density": float(results["energy_history"][0]),
            "final_energy_density": float(results["energy_history"][-1]),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
