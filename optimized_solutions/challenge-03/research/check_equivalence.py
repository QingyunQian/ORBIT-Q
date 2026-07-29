#!/usr/bin/env python3
"""Audit the exact Task 03 product-state reduction against the expert."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import optax
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "n_qubits": 12,
    "transverse_field": 0.9,
    "n_steps": 10,
    "log_probability_weight": 0.05,
    "max_steps": 3,
    "learning_rate": 0.01,
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def max_tree_error(left, right):
    return max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def main():
    reference = load("task03_reference_audit", "references/task-03/solution_3.py")
    candidate = load("task03_candidate_audit", "src/solutions/task-03/solution_3.py")
    K = tc.backend
    params = reference.initial_parameters(CONFIG)
    circuit = tc.Circuit(CONFIG["n_qubits"])
    for q in range(CONFIG["n_qubits"]):
        circuit.h(q)
    input_state = circuit.state()
    hamiltonian = reference.build_tfim_mvp(CONFIG)

    reference_state, reference_logs = reference.cooling_trajectory(
        params, input_state, CONFIG
    )
    product_states, candidate_logs = candidate.cooling_product(params, CONFIG)
    zero = K.convert_to_tensor(np.array([1.0, 0.0], dtype=np.complex64))
    candidate_state = zero
    for state in product_states:
        candidate_state = K.kron(candidate_state, state)
        if K.shape_tuple(candidate_state)[0] < 2**CONFIG["n_qubits"]:
            candidate_state = K.kron(candidate_state, zero)
    candidate_state = K.reshape(candidate_state, [-1])

    def reference_loss(p):
        return reference.observables(p, input_state, hamiltonian, CONFIG)

    def candidate_loss(p):
        return candidate.observables(p, CONFIG)

    (ref_loss, ref_aux), ref_grad = K.value_and_grad(
        reference_loss, has_aux=True
    )(params)
    (cand_loss, cand_aux), cand_grad = K.value_and_grad(
        candidate_loss, has_aux=True
    )(params)

    optimizer = optax.adam(CONFIG["learning_rate"])
    ref_opt = optimizer.init(params)
    cand_opt = optimizer.init(params)
    ref_updates, ref_opt = optimizer.update(ref_grad, ref_opt, params)
    cand_updates, cand_opt = optimizer.update(cand_grad, cand_opt, params)
    ref_next = optax.apply_updates(params, ref_updates)
    cand_next = optax.apply_updates(params, cand_updates)
    material_update_error = 0.0
    worst_update = None
    for parity in ("even", "odd"):
        for gate in ("rx", "xx", "zz"):
            rg = np.asarray(ref_grad[parity][gate])
            cg = np.asarray(cand_grad[parity][gate])
            rp = np.asarray(ref_next[parity][gate])
            cp = np.asarray(cand_next[parity][gate])
            difference = np.abs(rp - cp)
            index = np.unravel_index(np.argmax(difference), difference.shape)
            if worst_update is None or difference[index] > worst_update["error"]:
                worst_update = {
                    "leaf": f"{parity}.{gate}",
                    "index": [int(value) for value in index],
                    "error": float(difference[index]),
                    "reference_gradient": float(rg[index]),
                    "candidate_gradient": float(cg[index]),
                }
            material = np.maximum(np.abs(rg), np.abs(cg)) > 1e-7
            if np.any(material):
                material_update_error = max(
                    material_update_error, float(np.max(difference[material]))
                )
    ref_after_loss, ref_after_aux = reference_loss(ref_next)
    cand_after_loss, cand_after_aux = candidate_loss(cand_next)

    ref_history = reference.run_solution(CONFIG)
    cand_history = candidate.run_solution(CONFIG)
    history_errors = {
        key: float(
            np.max(np.abs(np.asarray(ref_history[key]) - np.asarray(value)))
        )
        for key, value in cand_history.items()
    }
    payload = {
        "state_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(reference_state)
                    - np.asarray(candidate_state)
                )
            )
        ),
        "log_probability_sum_abs_error": float(
            np.abs(
                np.asarray(K.sum(reference_logs))
                - np.asarray(K.sum(candidate_logs))
            )
        ),
        "loss_abs_error": float(np.abs(np.asarray(ref_loss) - np.asarray(cand_loss))),
        "aux_max_abs_error": max(
            float(np.abs(np.asarray(a) - np.asarray(b)))
            for a, b in zip(ref_aux, cand_aux)
        ),
        "gradient_max_abs_error": max_tree_error(ref_grad, cand_grad),
        "one_adam_update_max_abs_error": max_tree_error(ref_next, cand_next),
        "one_adam_update_material_gradient_max_abs_error": material_update_error,
        "worst_update_location": worst_update,
        "post_update_loss_abs_error": float(
            np.abs(np.asarray(ref_after_loss) - np.asarray(cand_after_loss))
        ),
        "post_update_aux_max_abs_error": max(
            float(np.abs(np.asarray(a) - np.asarray(b)))
            for a, b in zip(ref_after_aux, cand_after_aux)
        ),
        "optimizer_state_max_abs_error": max_tree_error(ref_opt, cand_opt),
        "history_max_abs_error": history_errors,
        "thresholds": {
            "state_and_observable": 2e-6,
            "gradient_and_update": 2e-5,
            "history": 2e-4,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
