#!/usr/bin/env python3
"""Audit the exact classical-ancilla reduction proposed for Task 07."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import optax
import tensorcircuit as tc


CONFIG = {
    "n_data_qubits": 8,
    "n_ancilla_qubits": 8,
    "n_qubits": 16,
    "n_layers": 2,
    "n_trajectories": 64,
    "initial_parameter_scale": 0.1,
    "max_steps": 100,
    "learning_rate": 0.02,
    "seed": 2047,
    "transverse_field": 1.05,
    "minimum_improvement": 0.3,
    "target_final_energy": -8.3,
}

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco-1-1")


def load_solution(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("task07_current_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {path}")
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


def exact_full_bits(params: Any, status: Any) -> Any:
    c = tc.Circuit(CONFIG["n_qubits"])
    pidx = 0
    sidx = 0
    layers = []
    for _ in range(CONFIG["n_layers"]):
        for q in range(CONFIG["n_data_qubits"]):
            c.ry(q, theta=params[pidx + q])
        pidx += CONFIG["n_data_qubits"]
        for a in range(CONFIG["n_ancilla_qubits"]):
            c.ry(CONFIG["n_data_qubits"] + a, theta=params[pidx + a])
        pidx += CONFIG["n_ancilla_qubits"]
        for a in range(CONFIG["n_ancilla_qubits"]):
            c.rzz(
                CONFIG["n_data_qubits"] + a,
                a,
                theta=params[pidx + a],
            )
        pidx += CONFIG["n_ancilla_qubits"]
        for a in range(CONFIG["n_ancilla_qubits"] - 1):
            c.cnot(
                CONFIG["n_data_qubits"] + a,
                CONFIG["n_data_qubits"] + a + 1,
            )
        theta0 = params[pidx : pidx + CONFIG["n_ancilla_qubits"]]
        pidx += CONFIG["n_ancilla_qubits"]
        theta1 = params[pidx : pidx + CONFIG["n_ancilla_qubits"]]
        pidx += CONFIG["n_ancilla_qubits"]
        bits = []
        for a in range(CONFIG["n_ancilla_qubits"]):
            bit = c.cond_measure(
                CONFIG["n_data_qubits"] + a, status=status[sidx]
            )
            bitf = K.cast(bit, "float32")
            feedback = theta0[a] + bitf * (theta1[a] - theta0[a])
            c.rz(a, theta=(1.0 - 2.0 * bitf) * feedback)
            bits.append(bit)
            sidx += 1
        for q in range(CONFIG["n_data_qubits"] - 1):
            c.cnot(q, q + 1)
        for q in range(CONFIG["n_data_qubits"]):
            c.rz(q, theta=params[pidx + q])
        pidx += CONFIG["n_data_qubits"]
        layers.append(K.stack(bits))
    return K.stack(layers)


def analytic_bits(params: Any, status: Any) -> tuple[Any, Any]:
    previous_measured = K.zeros([CONFIG["n_ancilla_qubits"]], dtype="int32")
    measured_layers = []
    pre_ladder_layers = []
    sidx = 0
    for layer in range(CONFIG["n_layers"]):
        offset = layer * 48
        ancilla_angles = params[offset + 8 : offset + 16]
        base_probability_one = K.sin(ancilla_angles / 2.0) ** 2
        previous_float = K.cast(previous_measured, "float32")
        probability_one = base_probability_one + previous_float * (
            1.0 - 2.0 * base_probability_one
        )
        measured = []
        pre_ladder = []
        previous_output = K.convert_to_tensor(0, dtype="int32")
        for a in range(CONFIG["n_ancilla_qubits"]):
            previous_output_float = K.cast(previous_output, "float32")
            measured_probability_one = probability_one[a] + previous_output_float * (
                1.0 - 2.0 * probability_one[a]
            )
            bit = K.cast(
                status[sidx] > (1.0 - measured_probability_one), "int32"
            )
            source_bit = bit + previous_output - 2 * bit * previous_output
            measured.append(bit)
            pre_ladder.append(source_bit)
            previous_output = bit
            sidx += 1
        previous_measured = K.stack(measured)
        measured_layers.append(previous_measured)
        pre_ladder_layers.append(K.stack(pre_ladder))
    return K.stack(measured_layers), K.stack(pre_ladder_layers)


def make_reduced_energy() -> Any:
    strings = []
    weights = []
    for i in range(CONFIG["n_data_qubits"] - 1):
        term = [0] * CONFIG["n_data_qubits"]
        term[i] = 3
        term[i + 1] = 3
        strings.append(term)
        weights.append(-1.0)
    for i in range(CONFIG["n_data_qubits"]):
        term = [0] * CONFIG["n_data_qubits"]
        term[i] = 1
        strings.append(term)
        weights.append(-CONFIG["transverse_field"])
    hamiltonian = tc.quantum.PauliStringSum2COO(strings, weights)

    def reduced_energy(params: Any, pattern: Any) -> Any:
        measured, pre_ladder = pattern
        c = tc.Circuit(CONFIG["n_data_qubits"])
        for layer in range(CONFIG["n_layers"]):
            offset = layer * 48
            for q in range(CONFIG["n_data_qubits"]):
                c.ry(q, theta=params[offset + q])
            theta0 = params[offset + 24 : offset + 32]
            theta1 = params[offset + 32 : offset + 40]
            for q in range(CONFIG["n_data_qubits"]):
                measured_float = K.cast(measured[layer, q], "float32")
                source_float = K.cast(pre_ladder[layer, q], "float32")
                feedback = theta0[q] + measured_float * (
                    theta1[q] - theta0[q]
                )
                angle = (
                    (1.0 - 2.0 * source_float) * params[offset + 16 + q]
                    + (1.0 - 2.0 * measured_float) * feedback
                )
                c.rz(q, theta=angle)
            for q in range(CONFIG["n_data_qubits"] - 1):
                c.cnot(q, q + 1)
            for q in range(CONFIG["n_data_qubits"]):
                c.rz(q, theta=params[offset + 40 + q])
        return tc.templates.measurements.operator_expectation(c, hamiltonian)

    return reduced_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path("/workspace/src/solutions/task-07/solution_7.py"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    solution = load_solution(args.solution)
    params = solution.initial_parameters(CONFIG)
    statuses = solution.trajectory_status(CONFIG)

    full_bits = ready(
        K.jit(K.vmap(exact_full_bits, vectorized_argnums=1))(params, statuses)
    )
    analytic = K.jit(K.vmap(analytic_bits, vectorized_argnums=1))
    analytic_measured, analytic_pre_ladder = ready(analytic(params, statuses))
    patterns = K.stack([analytic_measured, analytic_pre_ladder], axis=1)

    pattern_array = np.asarray(patterns, dtype=np.int32)
    flat_patterns = pattern_array.reshape(CONFIG["n_trajectories"], -1)
    unique_flat, inverse, counts = np.unique(
        flat_patterns, axis=0, return_inverse=True, return_counts=True
    )
    unique_patterns = K.convert_to_tensor(
        unique_flat.reshape(-1, 2, CONFIG["n_layers"], 8)
    )
    inverse_tensor = K.convert_to_tensor(inverse, dtype="int32")
    counts_tensor = K.convert_to_tensor(counts, dtype="float32")

    full_one = solution.make_one_trajectory(CONFIG)
    full_batch = K.jit(K.vmap(full_one, vectorized_argnums=1))
    reduced_one = make_reduced_energy()
    reduced_batch = K.jit(K.vmap(reduced_one, vectorized_argnums=1))

    def full_loss(p: Any) -> Any:
        return K.mean(full_batch(p, statuses))

    def reduced_loss(p: Any) -> Any:
        values = reduced_batch(p, unique_patterns)
        return K.sum(values * counts_tensor) / CONFIG["n_trajectories"]

    full_energy, full_grad = ready(
        K.jit(K.value_and_grad(full_loss))(params)
    )
    reduced_energy, reduced_grad = ready(
        K.jit(K.value_and_grad(reduced_loss))(params)
    )
    full_values = ready(full_batch(params, statuses))
    unique_values = ready(reduced_batch(params, unique_patterns))
    reduced_values = unique_values[inverse_tensor]

    ancilla_indices = np.array(
        [*range(8, 16), *range(56, 64)], dtype=np.int32
    )
    non_ancilla_mask = np.ones(96, dtype=bool)
    non_ancilla_mask[ancilla_indices] = False
    full_grad_np = np.asarray(full_grad)
    reduced_grad_np = np.asarray(reduced_grad)

    optimizer = optax.adam(CONFIG["learning_rate"])
    full_state = optimizer.init(params)
    reduced_state = optimizer.init(params)
    full_updates, full_state = optimizer.update(full_grad, full_state, params)
    reduced_updates, reduced_state = optimizer.update(
        reduced_grad, reduced_state, params
    )
    full_post = optax.apply_updates(params, full_updates)
    reduced_post = optax.apply_updates(params, reduced_updates)
    full_post_energy = ready(K.jit(full_loss)(full_post))
    reduced_post_energy = ready(K.jit(reduced_loss)(reduced_post))

    report = {
        "schema_version": 1,
        "task_id": "07",
        "full_vs_analytic_bits_equal": bool(
            np.array_equal(np.asarray(full_bits), np.asarray(analytic_measured))
        ),
        "unique_pattern_count": int(len(unique_flat)),
        "pattern_counts": [int(value) for value in counts],
        "rare_trajectory_indices": [
            int(index) for index in np.where(inverse != inverse[0])[0]
        ],
        "initial_energy": {
            "full": float(full_energy),
            "reduced": float(reduced_energy),
            "abs_error": abs(float(full_energy) - float(reduced_energy)),
        },
        "trajectory_energy_max_abs_error": float(
            np.max(np.abs(np.asarray(full_values) - np.asarray(reduced_values)))
        ),
        "gradient_max_abs_error": float(
            np.max(np.abs(full_grad_np - reduced_grad_np))
        ),
        "non_ancilla_gradient_max_abs_error": float(
            np.max(
                np.abs(
                    full_grad_np[non_ancilla_mask]
                    - reduced_grad_np[non_ancilla_mask]
                )
            )
        ),
        "full_ancilla_gradient_max_abs": float(
            np.max(np.abs(full_grad_np[ancilla_indices]))
        ),
        "reduced_ancilla_gradient_max_abs": float(
            np.max(np.abs(reduced_grad_np[ancilla_indices]))
        ),
        "post_update_parameter_max_abs_error": float(
            np.max(np.abs(np.asarray(full_post) - np.asarray(reduced_post)))
        ),
        "post_update_energy": {
            "full": float(full_post_energy),
            "reduced": float(reduced_post_energy),
            "abs_error": abs(
                float(full_post_energy) - float(reduced_post_energy)
            ),
        },
    }
    report["passed"] = bool(
        report["full_vs_analytic_bits_equal"]
        and report["unique_pattern_count"] == 2
        and report["initial_energy"]["abs_error"] <= 5e-5
        and report["non_ancilla_gradient_max_abs_error"] <= 5e-4
        and report["post_update_energy"]["abs_error"] <= 2e-3
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
