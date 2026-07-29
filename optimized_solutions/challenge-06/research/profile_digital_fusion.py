#!/usr/bin/env python3
"""Audit and profile exact Task 06 digital Euler-gate fusion."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
import tensorcircuit as tc

from profile_reference import CONFIG, load_reference, ready


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def current_state(psi: Any, rotations: Any) -> Any:
    circuit = tc.Circuit(CONFIG["n_qubits"], inputs=psi)
    for i in range(CONFIG["n_qubits"]):
        circuit.rz(i, theta=rotations[i, 0])
        circuit.ry(i, theta=rotations[i, 1])
        circuit.rz(i, theta=rotations[i, 2])
    return circuit.state()


def fused_state(psi: Any, rotations: Any) -> Any:
    circuit = tc.Circuit(CONFIG["n_qubits"], inputs=psi)
    for i in range(CONFIG["n_qubits"]):
        alpha, beta, gamma = rotations[i]
        phase = K.exp(-0.5j * K.cast(alpha + gamma, tc.dtypestr))
        unitary = phase * tc.gates.u_gate(
            theta=beta, phi=gamma, lbd=alpha
        ).tensor
        circuit.any(i, unitary=unitary)
    return circuit.state()


def measure(
    function: Callable[[Any], Any], argument: Any, repeats: int
) -> tuple[Any, dict[str, Any]]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    value = ready(compiled(argument))
    first_sec = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = ready(compiled(argument))
        samples.append(time.perf_counter() - started)
    return value, {
        "first_compile_and_exec_sec": first_sec,
        "steady_runtime_sec": samples,
        "steady_mean_sec": statistics.mean(samples),
        "steady_median_sec": statistics.median(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/workspace/references/task-06/solution_6.py"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    psi = reference.initial_state(CONFIG)
    rotations = reference.initial_parameters(CONFIG)["rot"][0]
    _, _, htarget = reference.build_hamiltonians(CONFIG)

    current_value, current = measure(
        lambda r: current_state(psi, r), rotations, args.repeats
    )
    fused_value, fused = measure(
        lambda r: fused_state(psi, r), rotations, args.repeats
    )

    def energy(state_function: Callable[[Any, Any], Any], r: Any) -> Any:
        state = state_function(psi, r)
        return K.real(K.tensordot(K.conj(state), htarget(state), 1))

    current_energy_grad, current_grad_profile = measure(
        jax.value_and_grad(lambda r: energy(current_state, r)),
        rotations,
        args.repeats,
    )
    fused_energy_grad, fused_grad_profile = measure(
        jax.value_and_grad(lambda r: energy(fused_state, r)),
        rotations,
        args.repeats,
    )
    current_energy, current_grad = current_energy_grad
    fused_energy, fused_grad = fused_energy_grad

    report = {
        "schema_version": 1,
        "task_id": "06",
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "state_max_abs_error": float(
            np.max(np.abs(np.asarray(current_value) - np.asarray(fused_value)))
        ),
        "energy_abs_error": float(
            abs(float(current_energy) - float(fused_energy))
        ),
        "gradient_max_abs_error": float(
            np.max(np.abs(np.asarray(current_grad) - np.asarray(fused_grad)))
        ),
        "state_current": current,
        "state_fused": fused,
        "state_steady_speedup": (
            current["steady_mean_sec"] / fused["steady_mean_sec"]
        ),
        "energy_gradient_current": current_grad_profile,
        "energy_gradient_fused": fused_grad_profile,
        "energy_gradient_steady_speedup": (
            current_grad_profile["steady_mean_sec"]
            / fused_grad_profile["steady_mean_sec"]
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
