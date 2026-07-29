#!/usr/bin/env python3
"""Compare the exact Task 05 MPS representation with the dense expert."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_from_mps(module, tensors):
    state = tensors[0][0]
    for tensor in tensors[1:]:
        state = module.K.einsum("...i,iaj->...aj", state, tensor)
    return module.K.reshape(state, (-1,))


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main() -> None:
    workload = json.loads(
        (ROOT / "datasets/public/task-05/canonical.json").read_text(
            encoding="utf-8"
        )
    )
    config = workload["input_config"]
    reference = _load(
        "task05_dense_reference",
        ROOT / "references/task-05/solution_5.py",
    )
    candidate = _load(
        "task05_exact_mps_candidate",
        ROOT / "src/solutions/task-05/solution_5.py",
    )
    params = reference.initial_parameters(config)
    dense_input = reference.initial_state(config)
    dense_hamiltonian = reference.build_tfim_mvp(config)
    mps_hamiltonian = candidate._tfim_mpo(
        config["n_qubits"],
        config["transverse_field"],
    )

    dense_state = reference.cooling_trajectory(params, dense_input, config)
    mps_tensors = candidate.cooling_tensors(params, config)
    reconstructed = _dense_from_mps(candidate, mps_tensors)

    def dense_loss(current):
        return reference.energy_density(
            current,
            dense_input,
            dense_hamiltonian,
            config,
        )

    def mps_loss(current):
        return candidate.energy_density(current, mps_hamiltonian, config)

    dense_energy, dense_grads = reference.K.value_and_grad(dense_loss)(params)
    mps_energy, mps_grads = candidate.K.value_and_grad(mps_loss)(params)
    optimizer = optax.adam(config["learning_rate"])
    dense_opt = optimizer.init(params)
    mps_opt = optimizer.init(params)
    dense_updates, _ = optimizer.update(dense_grads, dense_opt, params)
    mps_updates, _ = optimizer.update(mps_grads, mps_opt, params)
    dense_next = optax.apply_updates(params, dense_updates)
    mps_next = optax.apply_updates(params, mps_updates)
    conditioned_update_errors = []
    near_zero_gradient_count = 0
    for key in ("a", "b"):
        dense_gradient = np.asarray(reference.K.numpy(dense_grads[key]))
        update_error = np.abs(
            np.asarray(reference.K.numpy(dense_next[key]))
            - np.asarray(candidate.K.numpy(mps_next[key]))
        )
        stable = np.abs(dense_gradient) > 1e-5
        conditioned_update_errors.extend(update_error[stable].tolist())
        near_zero_gradient_count += int(np.size(stable) - np.count_nonzero(stable))

    payload = {
        "state_max_abs_error": _max_abs(
            reference.K.numpy(dense_state),
            candidate.K.numpy(reconstructed),
        ),
        "state_norm_abs_error": abs(
            float(reference.K.norm(dense_state))
            - float(candidate._mps_norm(mps_tensors))
        ),
        "initial_energy_abs_error": abs(float(dense_energy) - float(mps_energy)),
        "gradient_max_abs_error": max(
            _max_abs(
                reference.K.numpy(dense_grads[key]),
                candidate.K.numpy(mps_grads[key]),
            )
            for key in ("a", "b")
        ),
        "one_update_parameter_max_abs_error": max(
            _max_abs(
                reference.K.numpy(dense_next[key]),
                candidate.K.numpy(mps_next[key]),
            )
            for key in ("a", "b")
        ),
        "conditioned_one_update_max_abs_error": max(
            conditioned_update_errors,
            default=0.0,
        ),
        "near_zero_reference_gradient_count": near_zero_gradient_count,
        "maximum_exact_bond_dimension": max(
            max(candidate.K.shape_tuple(tensor)[0::2])
            for tensor in mps_tensors
        ),
    }
    payload["passed"] = (
        payload["state_max_abs_error"] <= 2e-5
        and payload["state_norm_abs_error"] <= 2e-5
        and payload["initial_energy_abs_error"] <= 3e-4
        and payload["gradient_max_abs_error"] <= 2e-4
        and payload["conditioned_one_update_max_abs_error"] <= 2e-5
        and payload["maximum_exact_bond_dimension"] <= 32
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
