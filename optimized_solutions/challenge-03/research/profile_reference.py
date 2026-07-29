#!/usr/bin/env python3
"""Profile Task 03 lowering, compilation, and compiled optimizer updates."""

from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import jax
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "references" / "task-03" / "solution_3.py"
CONFIG = {
    "n_qubits": 12,
    "transverse_field": 0.9,
    "n_steps": 10,
    "log_probability_weight": 0.05,
    "max_steps": 300,
    "learning_rate": 0.01,
}


def load_reference():
    spec = importlib.util.spec_from_file_location("task03_reference_profile", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synchronize(tree):
    leaf = jax.tree.leaves(tree)[0]
    leaf.block_until_ready()


def main():
    solution = load_reference()
    K = tc.backend
    params = solution.initial_parameters(CONFIG)
    circuit = tc.Circuit(CONFIG["n_qubits"])
    for i in range(CONFIG["n_qubits"]):
        circuit.h(i)
    input_state = circuit.state()
    hamiltonian = solution.build_tfim_mvp(CONFIG)
    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return solution.observables(p, input_state, hamiltonian, CONFIG)

    def train_step(p, state):
        (loss, aux), grads = K.value_and_grad(loss_fn, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    jitted = K.jit(train_step)
    start = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    lowering_sec = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation_sec = time.perf_counter() - start
    start = time.perf_counter()
    result = compiled(params, opt_state)
    synchronize(result)
    first_execution_sec = time.perf_counter() - start

    durations = []
    params, opt_state = result[0], result[1]
    for _ in range(20):
        start = time.perf_counter()
        result = compiled(params, opt_state)
        synchronize(result)
        durations.append(time.perf_counter() - start)
        params, opt_state = result[0], result[1]

    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    payload = {
        "source": "references/task-03/solution_3.py",
        "lowering_sec": lowering_sec,
        "compilation_sec": compilation_sec,
        "first_execution_sec": first_execution_sec,
        "steady_update_sec": {
            "count": len(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "stdev": statistics.stdev(durations),
            "values": durations,
        },
        "projected_300_update_sec": 300 * statistics.mean(durations),
        "stablehlo_line_count": len(lowered.as_text().splitlines()),
        "cost_analysis": {
            key: float(value)
            for key, value in (cost or {}).items()
            if isinstance(value, (int, float))
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
