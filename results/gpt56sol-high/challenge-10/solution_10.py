import os

import numpy as np


def run_solution(config):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc
    from tensorcircuit.quantum import PauliStringSum2MVP

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    selected = tuple(config["selected_qubits"])
    initial_ones = tuple(config["initial_ones"])

    paulis = []
    weights = []
    for q in range(n - 1):
        term = [0] * n
        term[q] = term[q + 1] = 3
        paulis.append(term)
        weights.append(-float(config["zz_strength"]))
    for q in range(n):
        term = [0] * n
        term[q] = 1
        paulis.append(term)
        weights.append(-float(config["x_strength"]))
    hamiltonian_mvp = PauliStringSum2MVP(paulis, weights)

    def energy_density(parameters):
        circuit = tc.Circuit(n)
        for q in initial_ones:
            circuit.x(q)
        for layer in range(layers):
            for q in range(n):
                circuit.rx(q, theta=parameters[layer, q, 0])
                circuit.rz(q, theta=parameters[layer, q, 1])
                circuit.ry(q, theta=parameters[layer, q, 2])
            circuit.cmz(*selected)
        state = circuit.state()
        h_state = hamiltonian_mvp(state)
        return tc.backend.real(
            tc.backend.sum(tc.backend.conj(state) * h_state)
        ) / n

    rng = np.random.default_rng(int(config["seed"]))
    parameters = jnp.asarray(
        rng.normal(
            scale=float(config["initial_parameter_scale"]),
            size=(layers, n, 3),
        ),
        dtype=jnp.float32,
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    value_and_grad = jax.value_and_grad(energy_density)

    def update(carry, unused):
        params, opt_state = carry
        energy, gradient = value_and_grad(params)
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), energy

    @jax.jit
    def optimize(params):
        initial_state = optimizer.init(params)
        return jax.lax.scan(
            update, (params, initial_state), None, length=int(config["max_steps"])
        )

    (parameters, _), history = optimize(parameters)
    return {
        "energy_history": np.asarray(history),
        "final_parameters": np.asarray(parameters),
    }
