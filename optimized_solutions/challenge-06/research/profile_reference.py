#!/usr/bin/env python3
"""Profile the immutable Task 06 expert without changing its semantics."""

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
    "n_qubits": 14,
    "n_blocks": 4,
    "t_min": 0.05,
    "t_max": 0.50,
    "ode_rtol": 1e-6,
    "ode_atol": 1e-6,
    "ode_max_steps": 16,
    "max_steps": 100,
    "learning_rate": 0.12,
    "maximum_energy_density_gap": 1.0,
}


def load_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("task06_immutable_reference", path)
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
        default=Path("/workspace/references/task-06/solution_6.py"),
    )
    parser.add_argument("--steady-steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    reference = load_reference(args.reference)
    hxy, hfield, htarget = reference.build_hamiltonians(CONFIG)
    psi0 = reference.initial_state(CONFIG)
    params = reference.initial_parameters(CONFIG)
    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)
    setup_sec = time.perf_counter() - started

    def loss_fn(p: Any) -> Any:
        return reference.forward(p, psi0, hxy, hfield, htarget, CONFIG)

    def train_step(p: Any, state: Any) -> tuple[Any, Any, Any]:
        energy, grads = reference.K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, energy

    analysis_step = reference.K.jit(train_step)
    started = time.perf_counter()
    lowered = analysis_step.lower(params, opt_state)
    lower_sec = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - started

    execution_step = reference.K.jit(train_step)
    started = time.perf_counter()
    params, opt_state, first_energy = ready(execution_step(params, opt_state))
    first_call_sec = time.perf_counter() - started

    steady: list[float] = []
    energies: list[float] = []
    for _ in range(args.steady_steps):
        started = time.perf_counter()
        params, opt_state, energy = ready(execution_step(params, opt_state))
        steady.append(time.perf_counter() - started)
        energies.append(float(energy))

    report = {
        "schema_version": 1,
        "task_id": "06",
        "reference_path": str(args.reference),
        "config": CONFIG,
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "setup_sec": setup_sec,
        "train_step": {
            "lower_sec": lower_sec,
            "compile_sec": compile_sec,
            "first_call_compile_and_exec_sec": first_call_sec,
            "first_energy_density": float(first_energy),
            "steady_measurements": len(steady),
            "steady_runtime_sec": steady,
            "steady_mean_runtime_sec": statistics.mean(steady),
            "steady_median_runtime_sec": statistics.median(steady),
            "steady_stdev_runtime_sec": statistics.stdev(steady)
            if len(steady) > 1
            else 0.0,
            "projected_100_exec_sec": 100 * statistics.mean(steady),
            "last_profile_energy_density": energies[-1],
            "cost_analysis": numeric_mapping(compiled.cost_analysis()),
            "memory_analysis": memory_mapping(compiled.memory_analysis()),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
