"""Variational non-unitary filter cooling for the open TFIM chain."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    g = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_strength = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    K = tc.backend

    # Sparse Pauli-sum representation of H for efficient <psi|H|psi>.
    structs = []
    weights = []
    for i in range(n_qubits - 1):
        s = [0] * n_qubits
        s[i] = 3
        s[i + 1] = 3
        structs.append(s)
        weights.append(np.float32(-1.0))
    for i in range(n_qubits):
        s = [0] * n_qubits
        s[i] = 1
        structs.append(s)
        weights.append(np.float32(-g))
    mvp = tc.quantum.PauliStringSum2MVP(structs, weights)

    x_mat = tc.gates._x_matrix
    zz_mat = tc.gates._zz_matrix

    def energy_density(params):
        """Build the cooling circuit, renormalize each layer, return <H>/n."""
        a = params[:, 0]
        b = params[:, 1]
        c = tc.Circuit(n_qubits)
        for i in range(n_qubits):
            c.h(i)

        for layer in range(n_layers):
            # exp(a X) = exp1(theta=i a, unitary=X) since U^2=I.
            theta_a = 1j * a[layer]
            for i in range(n_qubits):
                c.exp1(i, theta=theta_a, unitary=x_mat)

            # exp(b ZZ) on brickwork bonds of this layer.
            theta_b = 1j * b[layer]
            if layer % 2 == 0:
                bonds = range(0, n_qubits, 2)
            else:
                bonds = range(1, n_qubits - 1, 2)
            for i in bonds:
                c.exp1(i, i + 1, theta=theta_b, unitary=zz_mat)

            psi = c.state()
            psi = psi / K.norm(psi)
            c = tc.Circuit(n_qubits, inputs=psi)

        psi = c.state()
        energy = K.real(K.sum(K.conj(psi) * mvp(psi)))
        return energy / n_qubits

    params = jnp.asarray(
        init_strength * np.ones((n_layers, 2), dtype=np.float32)
    )
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        val, grad = jax.value_and_grad(energy_density)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val

    history = np.empty(max_steps, dtype=np.float64)
    for step in range(max_steps):
        params, opt_state, val = train_step(params, opt_state)
        history[step] = float(val)

    params_np = np.asarray(params, dtype=np.float64)
    final_a = params_np[:, 0].reshape(n_layers // 2, 2)
    final_b = params_np[:, 1].reshape(n_layers // 2, 2)

    return {
        "final_a": final_a,
        "final_b": final_b,
        "energy_density_history": history,
    }
