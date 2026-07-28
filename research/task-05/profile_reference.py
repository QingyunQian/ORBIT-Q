#!/usr/bin/env python3
"""Profile the immutable public Task 05 expert without changing its semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import optax


CONFIG = {
    "n_qubits": 18,
    "transverse_field": 1.10,
    "n_layers": 10,
    "initial_filter_strength": 0.01,
    "max_steps": 600,
    "learning_rate": 0.02,
    "maximum_energy_density_gap": 0.5,
}


def load_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("task05_immutable_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import immutable reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        value,
    )


def seconds(call: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = ready(call())
    return value, time.perf_counter() - started


def numeric_mapping(value: Any) -> dict[str, float | int | str]:
    if not isinstance(value, dict):
        return {"repr": str(value)}
    result: dict[str, float | int | str] = {}
    for key, item in value.items():
        if str(key) not in {"flops", "transcendentals", "bytes accessed"}:
            continue
        if isinstance(item, bool):
            result[str(key)] = int(item)
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            result[str(key)] = item
        else:
            result[str(key)] = str(item)
    return result


def memory_mapping(value: Any) -> dict[str, int | str]:
    if value is None:
        return {}
    result: dict[str, int | str] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        item = getattr(value, name)
        if isinstance(item, (int, float, str)):
            result[name] = int(item) if isinstance(item, float) else item
    return result or {"repr": str(value)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/workspace/references/task-05/solution_5.py"),
    )
    parser.add_argument("--steady-steps", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    reference = load_reference(args.reference)
    params = reference.initial_parameters(CONFIG)
    input_state = reference.initial_state(CONFIG)
    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)
    setup_sec = time.perf_counter() - started

    def make_loss() -> Any:
        # PauliStringSum2MVP owns a mutable dtype cache. Use a fresh closure for
        # every independent JAX trace so a tracer cannot escape one transform
        # and contaminate another.
        hamiltonian_mvp = reference.build_tfim_mvp(CONFIG)

        def loss_fn(p: Any) -> Any:
            return reference.energy_density(
                p,
                input_state,
                hamiltonian_mvp,
                CONFIG,
            )

        return loss_fn

    analysis_loss = make_loss()
    analysis_jitted_loss = reference.K.jit(analysis_loss)
    lower_started = time.perf_counter()
    lowered_loss = analysis_jitted_loss.lower(params)
    loss_lower_sec = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    compiled_loss = lowered_loss.compile()
    loss_compile_sec = time.perf_counter() - compile_started

    execution_loss = reference.K.jit(make_loss())
    (initial_energy, loss_first_call_sec) = seconds(lambda: execution_loss(params))
    (_, loss_steady_exec_sec) = seconds(lambda: execution_loss(params))

    analysis_step_loss = make_loss()

    def analysis_train_step(p: Any, state: Any) -> tuple[Any, Any, Any]:
        energy, grads = reference.K.value_and_grad(analysis_step_loss)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, energy

    analysis_jitted_step = reference.K.jit(analysis_train_step)
    lower_started = time.perf_counter()
    lowered_step = analysis_jitted_step.lower(params, opt_state)
    step_lower_sec = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    compiled_step = lowered_step.compile()
    step_compile_sec = time.perf_counter() - compile_started

    execution_step_loss = make_loss()

    def execution_train_step(p: Any, state: Any) -> tuple[Any, Any, Any]:
        energy, grads = reference.K.value_and_grad(execution_step_loss)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, energy

    execution_jitted_step = reference.K.jit(execution_train_step)
    ((params, opt_state, first_energy), step_first_call_sec) = seconds(
        lambda: execution_jitted_step(params, opt_state)
    )

    steady: list[float] = []
    energies: list[float] = []
    for _ in range(args.steady_steps):
        step_started = time.perf_counter()
        params, opt_state, energy = ready(
            execution_jitted_step(params, opt_state)
        )
        steady.append(time.perf_counter() - step_started)
        energies.append(float(energy))

    report = {
        "schema_version": 1,
        "task_id": "05",
        "reference_path": str(args.reference),
        "config": CONFIG,
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "setup_sec": setup_sec,
        "loss": {
            "lower_sec": loss_lower_sec,
            "compile_sec": loss_compile_sec,
            "first_call_compile_and_exec_sec": loss_first_call_sec,
            "steady_exec_sec": loss_steady_exec_sec,
            "initial_energy_density": float(initial_energy),
            "cost_analysis": numeric_mapping(compiled_loss.cost_analysis()),
            "memory_analysis": memory_mapping(compiled_loss.memory_analysis()),
        },
        "train_step": {
            "lower_sec": step_lower_sec,
            "compile_sec": step_compile_sec,
            "first_call_compile_and_exec_sec": step_first_call_sec,
            "first_energy_density": float(first_energy),
            "steady_measurements": len(steady),
            "steady_runtime_sec": steady,
            "steady_mean_runtime_sec": statistics.mean(steady),
            "steady_median_runtime_sec": statistics.median(steady),
            "steady_stdev_runtime_sec": statistics.stdev(steady)
            if len(steady) > 1
            else 0.0,
            "projected_600_exec_sec": 600 * statistics.mean(steady),
            "last_profile_energy_density": energies[-1],
            "cost_analysis": numeric_mapping(compiled_step.cost_analysis()),
            "memory_analysis": memory_mapping(compiled_step.memory_analysis()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
