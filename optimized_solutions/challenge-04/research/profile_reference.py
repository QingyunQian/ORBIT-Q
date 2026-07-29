#!/usr/bin/env python3
"""Profile the immutable Task 04 expert without changing benchmark semantics."""

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


def synchronized_seconds(function):
    start = time.perf_counter()
    value = function()
    if isinstance(value, tuple):
        leaves = value
    else:
        leaves = (value,)
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
    parser.add_argument("--steady-repeats", type=int, default=8)
    parser.add_argument(
        "--solution",
        choices=("reference", "optimized"),
        default="reference",
    )
    args = parser.parse_args()

    solution_path = (
        ROOT / "references" / "task-04" / "solution_4.py"
        if args.solution == "reference"
        else ROOT / "src" / "solutions" / "task-04" / "solution_4.py"
    )
    solution = load_module(
        solution_path,
        "task04_reference_profile",
    )
    solution_source = solution_path.read_text(encoding="utf-8")
    paired_channel = any(
        marker in solution_source
        for marker in ("paired_kraus = [", "noisy_kraus = [")
    )
    fused_entangler = "noisy_kraus = [" in solution_source
    evaluator = load_module(
        ROOT / "tasks" / "task-04" / "evaluator" / "evaluate_4.py",
        "task04_evaluator_profile",
    )
    config = dict(evaluator.DEFAULT_CONFIG)
    K = solution.K

    true_p01 = K.convert_to_tensor(config["true_p01"])
    true_p10 = K.convert_to_tensor(config["true_p10"])
    if hasattr(solution, "probe_states"):
        initial_states = solution.probe_states(config)
        target_function = lambda: solution.observable_table(  # noqa: E731
            true_p01, true_p10, initial_states, config
        )
    else:
        initial_states = None
        target_function = lambda: solution.observable_table(  # noqa: E731
            true_p01, true_p10, config
        )
    target, target_seconds = synchronized_seconds(target_function)

    params = solution.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        if initial_states is None:
            return solution.loss_and_observables(p, target, config)
        return solution.loss_and_observables(
            p, target, initial_states, config
        )

    value_and_grad = K.value_and_grad(loss_fn, has_aux=True)

    def train_step(p, state):
        (loss, aux), grads = value_and_grad(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    compiled_step = K.jit(train_step)
    lower_start = time.perf_counter()
    lowered = compiled_step.lower(params, opt_state)
    lower_seconds = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start

    first, first_seconds = synchronized_seconds(
        lambda: executable(params, opt_state)
    )
    params, opt_state = first[0], first[1]
    steady_seconds = []
    for _ in range(args.steady_repeats):
        result, elapsed = synchronized_seconds(
            lambda: executable(params, opt_state)
        )
        params, opt_state = result[0], result[1]
        steady_seconds.append(elapsed)

    cost_analysis = executable.cost_analysis()
    cost_analysis = {
        key: cost_analysis[key]
        for key in ("flops", "bytes accessed", "transcendentals")
        if key in cost_analysis
    }
    payload = {
        "schema_version": 1,
        "task_id": "04",
        "solution": args.solution,
        "solution_path": str(solution_path.relative_to(ROOT)),
        "solution_sha256": hashlib.sha256(solution_path.read_bytes()).hexdigest(),
        "configuration": config,
        "structural_counts": {
            "probes_per_table": 4,
            "channel_nodes_per_probe": 11 if paired_channel else 22,
            "channel_nodes_per_table": 44 if paired_channel else 88,
            "kraus_branches_per_node": 9 if paired_channel else 3,
            "rxx_nodes_per_probe": 0 if fused_entangler else 11,
            "rxx_nodes_per_table": 0 if fused_entangler else 44,
            "observables_per_probe": 13,
            "observables_per_table": 52,
            "training_steps": 120,
        },
        "target_observable_table_seconds": target_seconds,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_compiled_step_seconds": first_seconds,
        "steady_step_seconds": steady_seconds,
        "steady_step_mean_seconds": statistics.mean(steady_seconds),
        "steady_step_median_seconds": statistics.median(steady_seconds),
        "projected_120_step_seconds": (
            120 * statistics.mean(steady_seconds)
        ),
        "stablehlo_line_count": len(lowered.as_text().splitlines()),
        "cost_analysis": jsonable(cost_analysis),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
