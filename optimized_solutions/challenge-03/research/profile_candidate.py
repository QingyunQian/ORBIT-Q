#!/usr/bin/env python3
"""Profile lowering, compilation, and execution of the final Task 03 scan."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import jax
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "solutions" / "task-03" / "solution_3.py"
CONFIG = {
    "n_qubits": 12,
    "transverse_field": 0.9,
    "n_steps": 10,
    "log_probability_weight": 0.05,
    "max_steps": 300,
    "learning_rate": 0.01,
}


def load_candidate():
    spec = importlib.util.spec_from_file_location("task03_candidate_profile", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    solution = load_candidate()
    K = tc.backend
    params = solution.initial_parameters(CONFIG)
    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(
            solution.observables, has_aux=True
        )(p, CONFIG)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        energy, success, mean_log = aux
        return (p, state), K.stack([energy, success, mean_log, loss])

    def train(p, state):
        return K.jaxy_scan(
            train_step, (p, state), K.arange(CONFIG["max_steps"])
        )

    jitted = K.jit(train)
    start = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    lowering_sec = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation_sec = time.perf_counter() - start
    start = time.perf_counter()
    result = compiled(params, opt_state)
    jax.tree.leaves(result)[0].block_until_ready()
    execution_sec = time.perf_counter() - start
    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    payload = {
        "source": "src/solutions/task-03/solution_3.py",
        "lowering_sec": lowering_sec,
        "compilation_sec": compilation_sec,
        "complete_300_update_execution_sec": execution_sec,
        "stablehlo_line_count": len(lowered.as_text().splitlines()),
        "cost_analysis": {
            key: float(value)
            for key, value in (cost or {}).items()
            if key in {"flops", "transcendentals", "bytes accessed"}
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
