#!/usr/bin/env python3
"""Compare the final batched gates with native TensorCircuit gate products."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "src" / "solutions" / "task-01" / "solution_1.py"
EVALUATOR = ROOT / "tasks" / "task-01" / "evaluator" / "evaluate_1.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(value):
    leaves = value if isinstance(value, tuple) else (value,)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def one_native(solution, theta):
    K = solution.K
    return (
        K.reshape(solution.tc.gates.rz(theta=theta[2]).tensor, [2, 2])
        @ K.reshape(solution.tc.gates.ry(theta=theta[1]).tensor, [2, 2])
        @ K.reshape(solution.tc.gates.rz(theta=theta[0]).tensor, [2, 2])
    )


def two_native(solution, theta):
    K = solution.K
    return (
        K.reshape(solution.tc.gates.rzz(theta=theta[2]).tensor, [4, 4])
        @ K.reshape(solution.tc.gates.ryy(theta=theta[1]).tensor, [4, 4])
        @ K.reshape(solution.tc.gates.rxx(theta=theta[0]).tensor, [4, 4])
    )


def native_energy(solution, params, mps_input, config, mpo):
    circuit = solution.tc.Circuit(config["n_qubits"], mps_inputs=mps_input)
    offset = 0
    for layer in range(config["n_layers"]):
        singles = []
        for _ in range(config["n_qubits"]):
            singles.append(one_native(solution, params[offset : offset + 3]))
            offset += 3
        for i in range(layer % 2, config["n_qubits"] - 1, 2):
            pair = two_native(solution, params[offset : offset + 3])
            offset += 3
            unitary = pair @ solution.K.kron(singles[i], singles[i + 1])
            circuit.unitary(
                i,
                i + 1,
                unitary=solution.tc.gates.Gate(
                    solution.K.reshape(unitary, [2, 2, 2, 2])
                ),
            )
        if layer % 2:
            for i in (0, config["n_qubits"] - 1):
                circuit.unitary(
                    i, unitary=solution.tc.gates.Gate(singles[i])
                )
    return solution.mpo_expectation(circuit, mpo)


def trajectory(solution, loss_fn, config, steps):
    params = solution.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(params)

    def train_step(p, s):
        energy, grads = solution.K.value_and_grad(loss_fn)(p)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, energy

    train_step = solution.K.jit(train_step)
    history = []
    start = time.perf_counter()
    for _ in range(steps):
        params, state, energy = train_step(params, state)
        history.append(energy)
    ready((params, history[-1]))
    return (
        np.asarray(params),
        np.asarray(solution.K.stack(history)),
        time.perf_counter() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    solution = load(SOLUTION, "task01_final_equivalence")
    evaluator = load(EVALUATOR, "task01_final_evaluator")
    config = dict(evaluator.DEFAULT_CONFIG)
    dmrg_state, dmrg_energy = evaluator.dmrg_initial_state(config)
    config["dmrg_state"] = dmrg_state
    config["dmrg_energy"] = dmrg_energy
    native_mps_input = solution.tc.quantum.quimb2qop(dmrg_state)
    native_mpo = solution.tfim_mpo(config)
    candidate_mps_input = solution.tc.quantum.quimb2qop(dmrg_state)
    candidate_mpo = solution.tfim_mpo(config)

    theta = solution.K.convert_to_tensor(
        np.array([[0.31, -0.22, 0.17], [-0.40, 0.20, 0.70]], np.float32)
    )
    one_closed = solution.one_qubit_gates(theta)
    two_closed = solution.two_qubit_gates(theta)
    one_reference = solution.K.stack(
        [one_native(solution, row) for row in theta]
    )
    two_reference = solution.K.stack(
        [two_native(solution, row) for row in theta]
    )
    ready((one_closed, two_closed, one_reference, two_reference))

    def reference_loss(params):
        return native_energy(
            solution, params, native_mps_input, config, native_mpo
        )

    def candidate_loss(params):
        return solution.circuit_energy(
            params, candidate_mps_input, config, candidate_mpo
        )

    steps = config["max_steps"]
    reference_params, reference_history, reference_seconds = trajectory(
        solution, reference_loss, config, steps
    )
    candidate_params, candidate_history, candidate_seconds = trajectory(
        solution, candidate_loss, config, steps
    )
    payload = {
        "schema_version": 1,
        "steps": steps,
        "dmrg_energy": dmrg_energy,
        "candidate_sha256": hashlib.sha256(SOLUTION.read_bytes()).hexdigest(),
        "one_qubit_gate_max_absolute_difference": float(
            np.max(np.abs(np.asarray(one_closed) - np.asarray(one_reference)))
        ),
        "two_qubit_gate_max_absolute_difference": float(
            np.max(np.abs(np.asarray(two_closed) - np.asarray(two_reference)))
        ),
        "history_max_absolute_difference": float(
            np.max(np.abs(reference_history - candidate_history))
        ),
        "history_max_relative_difference": float(
            np.max(
                np.abs(reference_history - candidate_history)
                / np.maximum(np.abs(reference_history), 1e-7)
            )
        ),
        "parameter_max_absolute_difference_after_steps": float(
            np.max(np.abs(reference_params - candidate_params))
        ),
        "native_compile_plus_steps_seconds": reference_seconds,
        "candidate_compile_plus_steps_seconds": candidate_seconds,
        "native_initial_final": [
            float(reference_history[0]),
            float(reference_history[-1]),
        ],
        "candidate_initial_final": [
            float(candidate_history[0]),
            float(candidate_history[-1]),
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
