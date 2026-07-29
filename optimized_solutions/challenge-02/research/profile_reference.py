#!/usr/bin/env python3
"""Profile the immutable Task 02 expert without changing benchmark code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import jax
import numpy as np
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "task-02" / "solution_2.py"
EVALUATOR = ROOT / "tasks" / "task-02" / "evaluator" / "evaluate_2.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync(value):
    return jax.block_until_ready(value)


def timed_calls(fn, args, repeat: int):
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        sync(fn(*args))
        values.append(time.perf_counter() - start)
    return {
        "values_sec": values,
        "mean_sec": statistics.mean(values),
        "median_sec": statistics.median(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expert = load_module(REFERENCE, "task02_reference_profile")
    evaluator = load_module(EVALUATOR, "task02_evaluator_profile")
    config = dict(evaluator.DEFAULT_CONFIG)
    config["target_entropies"] = np.asarray(
        evaluator.DEFAULT_CONFIG["target_entropies"], dtype=np.float32
    )

    K = tc.set_backend("jax")
    tc.set_dtype("complex64")
    params = expert.initial_parameters(config)
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(1, config["n_qubits"], 2):
        circuit.x(i)
    input_state = circuit.state()
    hamiltonian_mvp = expert.build_xxz_mvp(config)
    target = K.convert_to_tensor(config["target_entropies"])
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return expert.observables(
            p, input_state, hamiltonian_mvp, config, target
        )

    def train_step(p, state):
        (loss, aux), grads = K.value_and_grad(loss_fn, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    def trajectory(p):
        return expert.block_states(p, input_state, config)

    trajectory_jit = K.jit(trajectory)
    loss_jit = K.jit(loss_fn)
    step_jit = K.jit(train_step)

    profile = {}
    for name, fn, fn_args in (
        ("trajectory", trajectory_jit, (params,)),
        ("loss", loss_jit, (params,)),
        ("train_step", step_jit, (params, opt_state)),
    ):
        start = time.perf_counter()
        lowered = fn.lower(*fn_args)
        lower_sec = time.perf_counter() - start
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        start = time.perf_counter()
        compiled = lowered.compile()
        compile_sec = time.perf_counter() - start
        start = time.perf_counter()
        sync(compiled(*fn_args))
        cold_execute_sec = time.perf_counter() - start
        profile[name] = {
            "lower_sec": lower_sec,
            "compile_sec": compile_sec,
            "cold_execute_sec": cold_execute_sec,
            "stablehlo_lines": len(stablehlo.splitlines()),
            "steady": timed_calls(compiled, fn_args, args.repeat),
        }

    final_state, _ = trajectory_jit(params)
    sync(final_state)
    energy_jit = K.jit(lambda s: expert.xxz_energy(s, hamiltonian_mvp))
    entropy_jit = K.jit(lambda s: expert.renyi2_entropy(s, config))
    for name, fn in (("energy", energy_jit), ("one_entropy", entropy_jit)):
        lowered = fn.lower(final_state)
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        start = time.perf_counter()
        compiled = lowered.compile()
        compile_sec = time.perf_counter() - start
        profile[name] = {
            "compile_sec": compile_sec,
            "stablehlo_lines": len(stablehlo.splitlines()),
            "steady": timed_calls(compiled, (final_state,), args.repeat),
        }

    report = {
        "schema_version": 1,
        "task_id": "02",
        "reference_sha256": sha256(REFERENCE),
        "evaluator_sha256": sha256(EVALUATOR),
        "jax": jax.__version__,
        "tensorcircuit": tc.__version__,
        "repeat": args.repeat,
        "profile": profile,
        "projected_500_train_steps_sec": (
            profile["train_step"]["steady"]["mean_sec"] * config["max_steps"]
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
