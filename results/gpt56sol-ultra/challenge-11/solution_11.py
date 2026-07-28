import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("greedy", preprocessing=True)


def run_solution(config):
    n = int(config["n_sites"])
    nl = int(config["n_layers"])
    beta = float(config["beta"])
    anis = float(config["single_ion_anisotropy"])
    r2 = np.sqrt(2.0)

    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.complex64) / r2
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], np.complex64) / r2
    sz = np.diag([1, 0, -1]).astype(np.complex64)
    ident = np.eye(3, dtype=np.complex64)
    dot = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    interaction = dot + beta * (dot @ dot)

    bond_hamiltonians = []
    for i in range(n - 1):
        wl = 1.0 if i == 0 else 0.5
        wr = 1.0 if i == n - 2 else 0.5
        h = interaction + anis * (
            wl * np.kron(sz @ sz, ident) + wr * np.kron(ident, sz @ sz)
        )
        bond_hamiltonians.append(jnp.asarray(h.reshape(3, 3, 3, 3)))

    initial_circuit = tc.QuditCircuit(n, dim=3)
    for i in range(1, n, 2):
        initial_circuit.x(i)
        initial_circuit.x(i)
    initial_state = initial_circuit.state()

    def rotation(alpha, theta, gamma):
        c, s = jnp.cos(theta), jnp.sin(theta)
        ry = jnp.array(
            [
                [(1 + c) / 2, -s / r2, (1 - c) / 2],
                [s / r2, c, -s / r2],
                [(1 - c) / 2, s / r2, (1 + c) / 2],
            ],
            dtype=jnp.complex64,
        )
        left = jnp.exp(-1j * jnp.array([gamma, 0.0, -gamma]))
        right = jnp.exp(-1j * jnp.array([alpha, 0.0, -alpha]))
        return left[:, None] * ry * right[None, :]

    def entangler(theta, phi):
        u = jnp.zeros((9, 9), dtype=jnp.complex64)
        ep = jnp.exp(-1j * (phi + beta))
        ea = jnp.exp(-1j * (-phi + beta))
        e1 = jnp.exp(-1j * beta)
        u = u.at[0, 0].set(ep).at[8, 8].set(ep)
        for a, b in ((1, 3), (5, 7)):
            d = e1 * jnp.cos(theta)
            o = -1j * e1 * jnp.sin(theta)
            u = u.at[a, a].set(d).at[b, b].set(d)
            u = u.at[a, b].set(o).at[b, a].set(o)

        aa, dd = -phi + 3 * beta, 2 * beta
        x = r2 * (theta - beta)
        t, z = (aa + dd) / 2, (aa - dd) / 2
        radius = jnp.sqrt(z * z + x * x)
        sinc = jnp.sinc(radius / jnp.pi)
        phase = jnp.exp(-1j * t)
        us = phase * (jnp.cos(radius) - 1j * sinc * z)
        ud = phase * (jnp.cos(radius) + 1j * sinc * z)
        ux = phase * (-1j * sinc * x)
        u = u.at[2, 2].set((us + ea) / 2).at[6, 6].set((us + ea) / 2)
        u = u.at[2, 6].set((us - ea) / 2).at[6, 2].set((us - ea) / 2)
        u = u.at[4, 4].set(ud)
        for a, b in ((2, 4), (6, 4), (4, 2), (4, 6)):
            u = u.at[a, b].set(ux / r2)
        return u.reshape(3, 3, 3, 3)

    def apply_layer(state, p):
        circuit = tc.QuditCircuit(n, dim=3, inputs=state)
        k = 0
        for i in range(n):
            circuit.any(i, unitary=rotation(*p[k : k + 3]))
            k += 3
        for i in range(0, n - 1, 2):
            circuit.any(i, i + 1, unitary=entangler(*p[k : k + 2]))
            k += 2
        for i in range(1, n - 1, 2):
            circuit.any(i, i + 1, unitary=entangler(*p[k : k + 2]))
            k += 2
        return circuit.state(), None

    def evolved_state(params):
        return jax.lax.scan(apply_layer, initial_state, params)[0]

    def energy_from_state(state):
        circuit = tc.QuditCircuit(n, dim=3, inputs=state)
        energy = 0.0
        for i, h in enumerate(bond_hamiltonians):
            energy += jnp.real(circuit.expectation((h, [i, i + 1]), reuse=True))
        return energy / n

    def energy(params):
        return energy_from_state(evolved_state(params))

    per_layer = 3 * n + 2 * (n - 1)
    rng = np.random.default_rng(int(config["seed"]))
    params = jnp.asarray(
        rng.normal(
            0.0,
            float(config["initial_parameter_scale"]),
            size=(nl, per_layer),
        ).astype(np.float32)
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)

    def update(carry, _):
        p, state = carry
        value, gradient = jax.value_and_grad(energy)(p)
        updates, state = optimizer.update(gradient, state, p)
        return (optax.apply_updates(p, updates), state), value

    @jax.jit
    def optimize(p, state):
        return jax.lax.scan(
            update, (p, state), None, length=int(config["max_steps"])
        )

    (params, _), history = optimize(params, opt_state)

    parity = jnp.diag(jnp.array([-1.0, 1.0, -1.0], dtype=jnp.complex64))
    szj = jnp.asarray(sz)

    @jax.jit
    def final_observables(p):
        state = evolved_state(p)
        final_energy = energy_from_state(state)
        circuit = tc.QuditCircuit(n, dim=3, inputs=state)
        strings = []
        for i, j in ((0, n - 1), (1, n - 2), (2, n - 3)):
            ops = [(szj, [i])]
            ops.extend((parity, [k]) for k in range(i + 1, j))
            ops.append((szj, [j]))
            strings.append(jnp.real(circuit.expectation(*ops, reuse=True)))
        return final_energy, jnp.stack(strings)

    final_energy, strings = final_observables(params)
    return {
        "energy_density_history": np.asarray(history),
        "final_energy_density": float(final_energy),
        "final_string_orders": np.asarray(strings),
    }
