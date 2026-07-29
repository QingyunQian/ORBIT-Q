#!/usr/bin/env python3
"""Whole-training scan removal ablation for the merged Task 12 candidate."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
PROFILE_REFERENCE = ROOT / "research/task-12/profile_reference.py"
SOLUTION = ROOT / "src/solutions/task-12/solution_12.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(value) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])


def lower_compile(fn, *args):
    started = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lower_sec = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - started
    return compiled, {
        "lower_sec": lower_sec,
        "compile_sec": compile_sec,
        "stablehlo_line_count": lowered.as_text().count("\n"),
    }


def main() -> None:
    profile_reference = load(PROFILE_REFERENCE, "task12_profile_reference")
    solution = load(SOLUTION, "task12_candidate")
    config = dict(profile_reference.CONFIG)
    dmrg_state = profile_reference.build_dmrg_state()

    K = tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("omeco")

    n_qubits = config["n_qubits"]
    layer_bonds = [
        list(range(layer % 2, n_qubits - 1, 2))
        for layer in range(config["n_layers"])
    ]
    n_gates = sum(len(bonds) for bonds in layer_bonds)
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
    gens = jnp.asarray(solution._GENERATORS, dtype=jnp.complex64)
    optimizer = optax.adam(config["learning_rate"])

    def objective(p):
        gates = solution._su4_batch(p.reshape(n_gates, 15), gens)
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        k = 0
        for bonds in layer_bonds:
            for i in bonds:
                circuit.any(i, i + 1, unitary=gates[k])
                k += 1
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    def body(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        return (optax.apply_updates(p, updates), state), (loss,) + aux

    def one_step(carry):
        return body(carry, None)

    def scan_train(carry):
        return jax.lax.scan(
            body,
            carry,
            None,
            length=config["max_steps"],
        )

    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(
        rng.normal(
            scale=config["initial_parameter_scale"],
            size=(15 * n_gates,),
        ).astype(np.float32)
    )
    init = (params, optimizer.init(params))
    compiled_scan, scan_compile = lower_compile(scan_train, init)
    compiled_step, step_compile = lower_compile(one_step, init)

    scan_times = []
    scan_output = None
    for _ in range(3):
        started = time.perf_counter()
        scan_output = compiled_scan(init)
        ready(scan_output)
        scan_times.append(time.perf_counter() - started)

    loop_times = []
    loop_history = None
    loop_final = None
    for _ in range(3):
        carry = init
        losses, fidelities, overlaps = [], [], []
        started = time.perf_counter()
        for _ in range(config["max_steps"]):
            carry, values = compiled_step(carry)
            loss, fidelity, overlap = values
            losses.append(loss)
            fidelities.append(fidelity)
            overlaps.append(overlap)
        ready((carry, losses[-1]))
        loop_times.append(time.perf_counter() - started)
        loop_final = carry
        loop_history = (
            K.stack(losses),
            K.stack(fidelities),
            K.stack(overlaps),
        )

    scan_final, scan_history = scan_output
    history_errors = [
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(scan_history, loop_history)
    ]
    parameter_error = float(
        np.max(
            np.abs(
                np.asarray(scan_final[0])
                - np.asarray(loop_final[0])
            )
        )
    )
    report = {
        "generated_by": "research/task-12/profile_factor_ablation.py",
        "scope": "scan removal with identical candidate objective",
        "steps": config["max_steps"],
        "scan_compile": scan_compile,
        "step_compile": step_compile,
        "scan_execution_sec": scan_times,
        "scan_mean_execution_sec": float(np.mean(scan_times)),
        "python_loop_execution_sec": loop_times,
        "python_loop_mean_execution_sec": float(np.mean(loop_times)),
        "history_max_abs_errors": history_errors,
        "parameter_max_abs_error": parameter_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
