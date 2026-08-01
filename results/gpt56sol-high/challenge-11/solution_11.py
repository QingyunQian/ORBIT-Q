import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    K = tc.backend
    n = int(config["n_sites"])
    layers = int(config["n_layers"])
    beta = float(config["beta"])
    anis = float(config["single_ion_anisotropy"])
    steps = int(config["max_steps"])

    sx = jnp.asarray(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.complex64
    ) / np.sqrt(2.0)
    sy = jnp.asarray(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=jnp.complex64
    ) / np.sqrt(2.0)
    sz = jnp.diag(jnp.asarray([1, 0, -1], dtype=jnp.complex64))
    sz2 = sz @ sz
    xy = jnp.kron(sx, sx) + jnp.kron(sy, sy)
    zz = jnp.kron(sz, sz)
    dot = xy + zz
    biquad = dot @ dot
    bond_h = dot + beta * biquad
    z_values = jnp.asarray([1.0, 0.0, -1.0])
    neel_flip = jnp.asarray(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=jnp.complex64
    )

    def op(matrix):
        return tc.gates.any(unitary=matrix, dim=3)

    bond_op = op(bond_h)
    anis_op = op(sz2)
    per_layer = 3 * n + 2 * (n - 1)

    def make_circuit(params):
        circuit = tc.QuditCircuit(n, dim=3)
        for site in range(1, n, 2):
            circuit.any(site, unitary=neel_flip)
        params = jnp.reshape(params, (layers, per_layer))
        for layer in range(layers):
            row = params[layer]
            rotations = jnp.reshape(row[: 3 * n], (n, 3))
            for site in range(n):
                alpha, angle, gamma = rotations[site]
                rza = jnp.diag(jnp.exp(-1j * alpha * z_values))
                rzg = jnp.diag(jnp.exp(-1j * gamma * z_values))
                ry = K.expm(-1j * angle * sy)
                circuit.any(site, unitary=rzg @ ry @ rza)
            offset = 3 * n
            for parity in (0, 1):
                for site in range(parity, n - 1, 2):
                    theta, phi = row[offset], row[offset + 1]
                    offset += 2
                    generator = theta * xy + phi * zz + beta * biquad
                    circuit.any(site, site + 1, unitary=K.expm(-1j * generator))
        return circuit

    def energy_density(params):
        circuit = make_circuit(params)
        energy = jnp.asarray(0.0j, dtype=jnp.complex64)
        for site in range(n - 1):
            energy = energy + circuit.expectation((bond_op, [site, site + 1]))
        for site in range(n):
            energy = energy + anis * circuit.expectation((anis_op, [site]))
        return jnp.real(energy) / n

    value_and_grad = jax.jit(jax.value_and_grad(energy_density))
    rng = np.random.default_rng(int(config["seed"]))
    params = jnp.asarray(
        rng.normal(
            0.0,
            float(config["initial_parameter_scale"]),
            layers * per_layer,
        ),
        dtype=jnp.float32,
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)
    history = []
    for _ in range(steps):
        value, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(value)

    final_energy, _ = value_and_grad(params)
    parity_op = jnp.diag(jnp.asarray([-1, 1, -1], dtype=jnp.complex64))

    def string_orders(params):
        circuit = make_circuit(params)
        values = []
        for left, right in ((0, 11), (1, 10), (2, 9)):
            operators = [(op(sz), [left])]
            operators += [(op(parity_op), [site]) for site in range(left + 1, right)]
            operators += [(op(sz), [right])]
            values.append(jnp.real(circuit.expectation(*operators)))
        return jnp.stack(values)

    final_strings = jax.jit(string_orders)(params)
    history_np, energy_np, strings_np = jax.device_get(
        (jnp.stack(history), final_energy, final_strings)
    )
    return {
        "energy_density_history": np.asarray(history_np),
        "final_energy_density": np.float64(energy_np),
        "final_string_orders": np.asarray(strings_np),
    }
