import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    nl = int(config["n_layers"])
    steps = int(config["max_steps"])
    field = float(config["transverse_field"])
    lr = float(config["learning_rate"])
    bonds = (tuple(range(0, n - 1, 2)), tuple(range(1, n - 1, 2)))

    # TensorCircuit's Pauli-string operator API gives the Hamiltonian action.
    strings, weights = [], []
    for i in range(n - 1):
        term = [0] * n
        term[i] = term[i + 1] = 3
        strings.append(term)
        weights.append(-1.0)
    for i in range(n):
        term = [0] * n
        term[i] = 1
        strings.append(term)
        weights.append(-field)
    hamiltonian_action = tc.quantum.PauliStringSum2MVP(strings, weights)

    def energy(params):
        circuit = tc.Circuit(n)
        circuit.h(range(n))
        state = circuit.state()
        for layer in range(nl):
            a, b = params[layer]
            one = jnp.array(
                [[jnp.cosh(a), jnp.sinh(a)], [jnp.sinh(a), jnp.cosh(a)]],
                dtype=jnp.complex64,
            )
            two = jnp.diag(
                jnp.array([jnp.exp(b), jnp.exp(-b), jnp.exp(-b), jnp.exp(b)],
                          dtype=jnp.complex64)
            )
            circuit = tc.Circuit(n, inputs=state)
            for qubit in range(n):
                circuit.any(qubit, unitary=one)
            for left in bonds[layer % 2]:
                circuit.any(left, left + 1, unitary=two)
            state = circuit.state()
            state = state / jnp.linalg.norm(state)
        return jnp.real(jnp.vdot(state, hamiltonian_action(state))) / n

    value_and_grad = jax.jit(jax.value_and_grad(energy))
    params = jnp.full((nl, 2), float(config["initial_filter_strength"]))
    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    history = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        value, gradient = value_and_grad(params)
        history[step] = float(value)  # value is deliberately before this update
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        t = step + 1
        first_hat = first_moment / (1.0 - 0.9**t)
        second_hat = second_moment / (1.0 - 0.999**t)
        params = params - lr * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)

    final = np.asarray(params)
    return {
        "final_a": final[:, 0].reshape(nl // 2, 2),
        "final_b": final[:, 1].reshape(nl // 2, 2),
        "energy_density_history": history,
    }
