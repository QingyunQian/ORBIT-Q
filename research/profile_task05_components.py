#!/usr/bin/env python3
"""Split immutable Task 05 forward cost into trajectory and Hamiltonian work."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import jax

from profile_task05_reference import (
    CONFIG,
    load_reference,
    memory_mapping,
    numeric_mapping,
    ready,
    seconds,
)


def profile_function(
    factory: Callable[[], Callable[..., Any]],
    args: tuple[Any, ...],
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    analysis_function = jax.jit(factory())
    started = time.perf_counter()
    lowered = analysis_function.lower(*args)
    lower_sec = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - started

    execution_function = jax.jit(factory())
    (value, first_call_sec) = seconds(lambda: execution_function(*args))
    steady: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = ready(execution_function(*args))
        steady.append(time.perf_counter() - started)

    return value, {
        "lower_sec": lower_sec,
        "compile_sec": compile_sec,
        "first_call_compile_and_exec_sec": first_call_sec,
        "steady_measurements": repeats,
        "steady_runtime_sec": steady,
        "steady_mean_runtime_sec": statistics.mean(steady),
        "steady_median_runtime_sec": statistics.median(steady),
        "steady_stdev_runtime_sec": statistics.stdev(steady)
        if repeats > 1
        else 0.0,
        "cost_analysis": numeric_mapping(compiled.cost_analysis()),
        "memory_analysis": memory_mapping(compiled.memory_analysis()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/workspace/references/task-05/solution_5.py"),
    )
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    params = reference.initial_parameters(CONFIG)
    input_state = reference.initial_state(CONFIG)

    def trajectory_factory() -> Callable[[Any], Any]:
        def trajectory(p: Any) -> Any:
            return reference.cooling_trajectory(p, input_state, CONFIG)

        return trajectory

    final_state, trajectory = profile_function(
        trajectory_factory,
        (params,),
        args.repeats,
    )

    def hamiltonian_factory() -> Callable[[Any], Any]:
        # The MVP owns a mutable dtype cache, so each independent trace gets a
        # fresh closure.
        hamiltonian_mvp = reference.build_tfim_mvp(CONFIG)

        def energy(state: Any) -> Any:
            return (
                reference.tfim_energy(state, hamiltonian_mvp)
                / CONFIG["n_qubits"]
            )

        return energy

    energy, hamiltonian = profile_function(
        hamiltonian_factory,
        (final_state,),
        args.repeats,
    )

    report = {
        "schema_version": 1,
        "task_id": "05",
        "reference_path": str(args.reference),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "initial_parameter_energy_density": float(energy),
        "final_state_norm": float(reference.K.norm(final_state)),
        "trajectory": trajectory,
        "hamiltonian_energy": hamiltonian,
        "combined_component_mean_sec": (
            trajectory["steady_mean_runtime_sec"]
            + hamiltonian["steady_mean_runtime_sec"]
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
