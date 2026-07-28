import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    backend = tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy")

    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    qubits = tuple(range(n))
    selected = tuple(config["selected_qubits"])
    initial_ones = tuple(config["initial_ones"])

    pauli_strings = []
    weights = []
    for i in range(n - 1):
        term = [0] * n
        term[i] = term[i + 1] = 3
        pauli_strings.append(term)
        weights.append(-float(config["zz_strength"]))
    for i in range(n):
        term = [0] * n
        term[i] = 1
        pauli_strings.append(term)
        weights.append(-float(config["x_strength"]))

    matvec = tc.quantum.PauliStringSum2MVP(pauli_strings, weights)
    hamiltonian = tc.aslinearoperator(
        matvec, shape=(2**n, 2**n), dtype=jnp.complex64
    )

    def energy_density(parameters):
        circuit = tc.Circuit(n)
        circuit.x(initial_ones)
        for layer in range(layers):
            circuit.rx(qubits, theta=parameters[layer, :, 0])
            circuit.rz(qubits, theta=parameters[layer, :, 1])
            circuit.ry(qubits, theta=parameters[layer, :, 2])
            circuit.cmz(*selected)
        state = circuit.state()
        h_state = hamiltonian @ state
        return backend.real(backend.sum(backend.conj(state) * h_state)) / n

    rng = np.random.default_rng(int(config["seed"]))
    parameters = jnp.asarray(
        rng.normal(
            scale=float(config["initial_parameter_scale"]),
            size=(layers, n, 3),
        ),
        dtype=jnp.float32,
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(parameters)
    value_and_grad = jax.value_and_grad(energy_density)

    def adam_step(carry, unused):
        params, state = carry
        value, gradient = value_and_grad(params)
        updates, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    optimize = jax.jit(
        lambda p, s: jax.lax.scan(
            adam_step, (p, s), None, length=int(config["max_steps"])
        )
    )
    (parameters, _), history = optimize(parameters, opt_state)
    return {
        "energy_history": np.asarray(history),
        "final_parameters": np.asarray(parameters),
    }
