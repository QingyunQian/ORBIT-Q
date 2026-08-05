"""Variational circuit to DMRG-MPS overlap optimization (TensorCircuit-NG)."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax


def _to_tc_tensors(mps):
    """Convert quimb MPS arrays to TC (left, phys, right) tensors."""
    mps = mps.copy()
    try:
        mps.permute_arrays("lpr")
    except Exception:
        pass
    arrays = list(mps.arrays)
    out = []
    nsite = len(arrays)
    for i, arr in enumerate(arrays):
        a = np.asarray(arr, dtype=np.complex128)
        if a.ndim == 2:
            if i == 0:
                a = a.reshape(1, a.shape[0], a.shape[1])
            else:
                a = a.reshape(a.shape[0], a.shape[1], 1)
        elif a.ndim != 3:
            raise ValueError(f"Unexpected MPS tensor rank {a.ndim} at site {i}/{nsite}")
        out.append(jnp.asarray(a))
    return out


def _neel_tensors(n_qubits: int):
    """MPS tensors for the product state |0101...01>."""
    tensors = []
    for i in range(n_qubits):
        t = np.zeros((1, 2, 1), dtype=np.complex128)
        t[0, i % 2, 0] = 1.0
        tensors.append(jnp.asarray(t))
    return tensors


def _n_gates(n_qubits: int, n_layers: int) -> int:
    total = 0
    for layer in range(n_layers):
        start = 0 if (layer % 2 == 0) else 1
        total += len(range(start, n_qubits - 1, 2))
    return total


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    target_tensors = _to_tc_tensors(config["dmrg_state"])
    target_bra = tc.Circuit(n_qubits, tensors=target_tensors).get_quvector().adjoint()
    init_tensors = _neel_tensors(n_qubits)

    n_params = _n_gates(n_qubits, n_layers) * 15
    rng = np.random.RandomState(seed)
    params = jnp.asarray(rng.normal(scale=scale, size=(n_params,)))

    def overlap(params):
        thetas = jnp.reshape(params, (-1, 15))
        circuit = tc.Circuit(n_qubits, tensors=init_tensors)
        idx = 0
        for layer in range(n_layers):
            start = 0 if (layer % 2 == 0) else 1
            for q in range(start, n_qubits - 1, 2):
                circuit.su4(q, q + 1, theta=thetas[idx])
                idx += 1
        return (target_bra @ circuit.get_quvector()).eval()

    def loss_fn(params):
        ov = overlap(params)
        fid = jnp.real(jnp.conj(ov) * ov)
        return 1.0 - fid

    value_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def _step(carry, _):
        params, opt_state = carry
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # loss/fid recorded for pre-update parameters
        return (params, opt_state), loss

    # Compile via a scan over all optimizer updates.
    (params, opt_state), loss_hist = jax.lax.scan(
        _step, (params, opt_state), xs=None, length=max_steps
    )

    loss_history = np.asarray(loss_hist, dtype=np.float64)
    fidelity_history = 1.0 - loss_history

    ov_final = complex(overlap(params))
    phase = float(np.angle(ov_final))

    return {
        "loss_history": loss_history,
        "fidelity_history": np.asarray(fidelity_history, dtype=np.float64),
        "final_parameters": np.asarray(params, dtype=np.float64),
        "final_overlap_phase": phase,
    }
