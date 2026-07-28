import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import optax
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO
from tensorcircuit.templates.measurements import sparse_expectation


tc.set_backend("jax")


def _hamiltonians(n):
    xy_terms, z_terms, target_terms, target_weights = [], [], [], []
    for i in range(n - 1):
        for pauli, weight in ((1, 0.7), (2, 0.7), (3, 1.1)):
            term = [0] * n
            term[i] = term[i + 1] = pauli
            target_terms.append(term)
            target_weights.append(weight)
        for pauli in (1, 2):
            term = [0] * n
            term[i] = term[i + 1] = pauli
            xy_terms.append(term)
    for i in range(n):
        term = [0] * n
        term[i] = 3
        z_terms.append(term)
        target_terms.append(term)
        target_weights.append(0.25 * (-1.0) ** i)
    hxy = PauliStringSum2COO(xy_terms).sum_duplicates(remove_zeros=True)
    hz = PauliStringSum2COO(
        z_terms, [(-1.0) ** i for i in range(n)]
    ).sum_duplicates(remove_zeros=True)
    target = PauliStringSum2COO(
        target_terms, target_weights
    ).sum_duplicates(remove_zeros=True)
    return hxy, hz, target


def run_solution(config):
    n, blocks = int(config["n_qubits"]), int(config["n_blocks"])
    tmin, tmax = float(config["t_min"]), float(config["t_max"])
    rtol, atol = float(config["ode_rtol"]), float(config["ode_atol"])
    max_ode_steps = int(config["ode_max_steps"])
    hxy, hz, target = _hamiltonians(n)

    rng = np.random.default_rng(0)
    angles = rng.normal(0.0, 0.1, (blocks, n, 3))
    params = jnp.asarray(
        np.concatenate(
            [np.zeros(blocks), np.full(2 * blocks, 0.1), angles.ravel()]
        ),
        dtype=jnp.float32,
    )

    def energy_density(p):
        s, raw_j, raw_d = p[:blocks], p[blocks : 2 * blocks], p[2 * blocks : 3 * blocks]
        rotations = p[3 * blocks :].reshape(blocks, n, 3)
        times = tmin + (tmax - tmin) * jax.nn.sigmoid(s)
        circuit = tc.AnalogCircuit(n)
        for qubit in range(1, n, 2):
            circuit.x(qubit)
        for layer in range(blocks):
            coupling, detuning = jnp.tanh(raw_j[layer]), jnp.tanh(raw_d[layer])

            def schrodinger_rhs(state, time, coupling=coupling, detuning=detuning):
                del time
                return -1j * (
                    coupling * (hxy @ state) + detuning * (hz @ state)
                )

            circuit.add_analog_block(
                schrodinger_rhs,
                times[layer],
                mode="raw",
                ode_backend="diffrax",
                solver="Tsit5",
                dt0=tmin,
                rtol=rtol,
                atol=atol,
                max_steps=max_ode_steps,
            )
            for qubit in range(n):
                circuit.rz(qubit, theta=rotations[layer, qubit, 0])
                circuit.ry(qubit, theta=rotations[layer, qubit, 1])
                circuit.rz(qubit, theta=rotations[layer, qubit, 2])
        final_circuit = tc.Circuit(n, inputs=circuit.state())
        return sparse_expectation(final_circuit, target) / n

    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(energy_density)

    def update(carry, unused):
        del unused
        p, state = carry
        value, gradient = value_and_grad(p)
        updates, state = optimizer.update(gradient, state, p)
        return (optax.apply_updates(p, updates), state), value

    @jax.jit
    def train(p, state):
        return jax.lax.scan(
            update, (p, state), xs=None, length=int(config["max_steps"])
        )

    (params, _), history = train(params, opt_state)
    history.block_until_ready()
    raw = np.asarray(jax.device_get(params), dtype=np.float64)
    sigmoid_s = 1.0 / (1.0 + np.exp(-raw[:blocks]))
    return {
        "final_analog_times": tmin + (tmax - tmin) * sigmoid_s,
        "final_analog_couplings": np.tanh(raw[blocks : 2 * blocks]),
        "final_analog_detunings": np.tanh(raw[2 * blocks : 3 * blocks]),
        "energy_density_history": np.asarray(jax.device_get(history)),
    }
