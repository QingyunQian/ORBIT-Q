#!/usr/bin/env python3
"""Component ablations for the merged Task 11 candidate.

This profiler keeps the candidate math fixed while comparing:

* batched fixed Pade(3,3) entanglers with batched adaptive ``K.expm``;
* the precomputed diagonal onsite vector with 12 framework expectations;
* one compiled training scan with repeated dispatch of the identical compiled
  candidate step.

The measurements are diagnostic component ablations, not replacements for the
canonical six-pair evaluator result.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import jax
import numpy as np
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "src/solutions/task-11/solution_11.py"
K = tc.set_backend("jax")
tc.set_dtype("complex64")

CONFIG = {
    "n_sites": 12,
    "n_layers": 5,
    "beta": 0.20,
    "single_ion_anisotropy": 0.15,
    "max_steps": 10,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.05,
    "seed": 2041,
}


def load_solution():
    spec = importlib.util.spec_from_file_location("task11_candidate", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SOLUTION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(value) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])


def compile_and_time(fn, args, repeats=5):
    lowered_started = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lowered_sec = time.perf_counter() - lowered_started
    compiled_started = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - compiled_started
    ready(compiled(*args))
    started = time.perf_counter()
    for _ in range(repeats):
        value = compiled(*args)
    ready(value)
    return {
        "lower_sec": lowered_sec,
        "compile_sec": compile_sec,
        "mean_execution_sec": (time.perf_counter() - started) / repeats,
        "compiled": compiled,
    }


def main() -> None:
    candidate = load_solution()
    params = candidate.initial_parameters(CONFIG)
    theta = params["even_theta"][0]
    phi = params["even_phi"][0]

    def fixed_pade(t, p):
        return candidate.entangler_batch(t, p, CONFIG["beta"])

    def adaptive_expm(t, p):
        generator = (
            K.cast(t, "complex64")[:, None, None] * candidate.DOT_BOND
            + K.cast(p - t, "complex64")[:, None, None] * candidate.ZZ_BOND
            + K.cast(CONFIG["beta"], "complex64")
            * candidate.DOT_BOND_SQUARED
        )
        return K.vmap(K.expm)(-1.0j * generator)

    fixed_value = fixed_pade(theta, phi)
    adaptive_value = adaptive_expm(theta, phi)
    ready((fixed_value, adaptive_value))
    entangler_error = float(
        np.max(np.abs(np.asarray(fixed_value) - np.asarray(adaptive_value)))
    )
    fixed_profile = compile_and_time(fixed_pade, (theta, phi), repeats=10)
    adaptive_profile = compile_and_time(adaptive_expm, (theta, phi), repeats=10)

    rng = np.random.default_rng(0)
    state_np = rng.normal(size=(candidate.DIM**CONFIG["n_sites"],)) + 1.0j * rng.normal(
        size=(candidate.DIM**CONFIG["n_sites"],)
    )
    state = K.convert_to_tensor(
        (state_np / np.linalg.norm(state_np)).astype(np.complex64)
    )
    onsite_gate = tc.gates.Gate(
        K.convert_to_tensor(np.diag([1.0, 0.0, 1.0]).astype(np.complex64))
    )
    digits = candidate.basis_digit_table(CONFIG["n_sites"])
    onsite_coeffs = K.convert_to_tensor(
        (
            CONFIG["single_ion_anisotropy"]
            * candidate.SZ2_DIAG[digits].sum(axis=1)
        ).astype(np.float32)
    )

    def onsite_expectations(psi):
        circuit = tc.QuditCircuit(CONFIG["n_sites"], dim=candidate.DIM, inputs=psi)
        total = K.cast(0.0, "complex64")
        for site in range(CONFIG["n_sites"]):
            total += CONFIG["single_ion_anisotropy"] * circuit.expectation(
                (onsite_gate, [site])
            )
        return K.real(total)

    def onsite_diagonal(psi):
        return K.sum(onsite_coeffs * K.abs(psi) ** 2)

    onsite_reference = onsite_expectations(state)
    onsite_candidate = onsite_diagonal(state)
    ready((onsite_reference, onsite_candidate))
    onsite_error = float(
        np.abs(np.asarray(onsite_reference) - np.asarray(onsite_candidate))
    )
    onsite_expectation_profile = compile_and_time(
        onsite_expectations, (state,), repeats=20
    )
    onsite_diagonal_profile = compile_and_time(onsite_diagonal, (state,), repeats=20)

    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return candidate.energy_density_from_state(
            candidate.build_state(p, CONFIG),
            CONFIG,
            onsite_coeffs,
        )

    def train_step(carry, _):
        p, state_ = carry
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state_ = optimizer.update(grads, state_, p)
        return (optax.apply_updates(p, updates), state_), value

    def scan_train(carry):
        return K.jaxy_scan(
            train_step,
            carry,
            K.zeros([CONFIG["max_steps"]]),
        )

    def one_step(carry):
        return train_step(carry, None)

    init = (params, opt_state)
    scan_profile = compile_and_time(scan_train, (init,), repeats=3)
    step_profile = compile_and_time(one_step, (init,), repeats=1)
    compiled_step = step_profile["compiled"]
    loop_times = []
    loop_history = None
    loop_final = None
    for _ in range(3):
        carry = init
        history = []
        started = time.perf_counter()
        for _ in range(CONFIG["max_steps"]):
            carry, value = compiled_step(carry)
            history.append(value)
        ready((carry, history[-1]))
        loop_times.append(time.perf_counter() - started)
        loop_final = carry
        loop_history = K.stack(history)
    scan_final, scan_history = scan_profile["compiled"](init)
    ready((scan_final, scan_history))
    history_error = float(
        np.max(np.abs(np.asarray(scan_history) - np.asarray(loop_history)))
    )

    def public_profile(row):
        return {
            key: value
            for key, value in row.items()
            if key != "compiled"
        }

    report = {
        "generated_by": "research/task-11/profile_factor_ablation.py",
        "scope": "diagnostic component ablation; canonical claim unchanged",
        "configuration": CONFIG,
        "entangler": {
            "fixed_pade": public_profile(fixed_profile),
            "adaptive_expm": public_profile(adaptive_profile),
            "max_abs_gate_error": entangler_error,
        },
        "onsite": {
            "diagonal_vector": public_profile(onsite_diagonal_profile),
            "framework_expectations": public_profile(onsite_expectation_profile),
            "absolute_value_error": onsite_error,
        },
        "training_control_flow": {
            "steps": CONFIG["max_steps"],
            "scan": public_profile(scan_profile),
            "python_loop_compile": public_profile(step_profile),
            "python_loop_execution_sec": loop_times,
            "python_loop_mean_execution_sec": float(np.mean(loop_times)),
            "history_max_abs_error": history_error,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
