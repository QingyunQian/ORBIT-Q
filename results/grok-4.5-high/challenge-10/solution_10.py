"""22-qubit VQE with an 18-qubit controlled-Z hyperedge (TensorCircuit-NG)."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc
from jax import lax
import jax
import jax.numpy as jnp

tc.set_backend("jax")
tc.set_dtype("complex64")


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    selected_qubits = list(config["selected_qubits"])
    initial_ones = list(config["initial_ones"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_parameter_scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    dim = 1 << n_qubits
    idx = jnp.arange(dim)
    # TensorCircuit bit order: qubit 0 is the most significant bit.
    bit_shifts = jnp.arange(n_qubits - 1, -1, -1)
    bits = (idx[None, :] >> bit_shifts[:, None]) & 1
    z_eigs = (1 - 2 * bits).astype(jnp.float32)
    zz_diag = jnp.zeros(dim, dtype=jnp.float32)
    for i in range(n_qubits - 1):
        zz_diag = zz_diag - zz_strength * z_eigs[i] * z_eigs[i + 1]
    x_masks = jnp.array(
        [1 << (n_qubits - 1 - q) for q in range(n_qubits)], dtype=idx.dtype
    )

    def apply_hamiltonian(psi):
        out = zz_diag.astype(psi.dtype) * psi
        flipped = jax.vmap(lambda mask: psi[idx ^ mask])(x_masks)
        out = out - x_strength * jnp.sum(flipped, axis=0)
        return out

    def energy_density_from_state(psi):
        return jnp.real(jnp.vdot(psi, apply_hamiltonian(psi))) / n_qubits

    def ansatz_state(params):
        circuit = tc.Circuit(n_qubits)
        for q in initial_ones:
            circuit.x(q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                circuit.rx(q, theta=params[layer, q, 0])
                circuit.rz(q, theta=params[layer, q, 1])
                circuit.ry(q, theta=params[layer, q, 2])
            circuit.cmz(*selected_qubits)
        return circuit.state()

    def energy_density(params):
        return energy_density_from_state(ansatz_state(params))

    value_and_grad = jax.value_and_grad(energy_density)

    def adam_step(carry, _):
        params, m, v, t = carry
        value, grad = value_and_grad(params)
        t = t + 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        return (params, m, v, t), value

    @jax.jit
    def optimize(params0):
        m0 = jnp.zeros_like(params0)
        v0 = jnp.zeros_like(params0)
        t0 = jnp.array(0.0, dtype=params0.dtype)
        (params_f, _, _, _), history = lax.scan(
            adam_step, (params0, m0, v0, t0), xs=None, length=max_steps
        )
        return params_f, history

    key = jax.random.PRNGKey(seed)
    params0 = initial_parameter_scale * jax.random.normal(
        key, (n_layers, n_qubits, 3), dtype=jnp.float32
    )
    final_parameters, energy_history = optimize(params0)

    return {
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "final_parameters": np.asarray(final_parameters, dtype=np.float64),
    }
