import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    k = tc.backend
    g = tc.gates
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    layers = int(config["n_layers"])

    paulis = []
    weights = []
    for i in range(n - 1):
        term = [0] * n
        term[i] = term[i + 1] = 3
        paulis.append(term)
        weights.append(-1.0)
    for i in range(n):
        term = [0] * n
        term[i] = 1
        paulis.append(term)
        weights.append(-field)
    hamiltonian = tc.quantum.aslinearoperator(
        tc.quantum.PauliStringSum2MVP(paulis, weights),
        shape=(2**n, 2**n),
    )

    def energy_density(params):
        circuit = tc.Circuit(n)
        for i in range(n):
            circuit.H(i)
        for layer in range(layers):
            for i in range(n):
                circuit.exp1(
                    i, theta=1j * params[0, layer], unitary=g._x_matrix
                )
            for i in range(layer % 2, n - 1, 2):
                circuit.exp1(
                    i,
                    i + 1,
                    theta=1j * params[1, layer],
                    unitary=g._zz_matrix,
                )
            state = circuit.state()
            state = state / k.norm(state)
            circuit = tc.Circuit(n, inputs=state)
        state = circuit.state()
        return k.real(k.sum(k.conj(state) * (hamiltonian @ state))) / n

    optimizer = optax.adam(float(config["learning_rate"]))

    def train(params):
        opt_state = optimizer.init(params)

        def update(carry, _):
            current, state = carry
            value, gradient = jax.value_and_grad(energy_density)(current)
            delta, state = optimizer.update(gradient, state, current)
            current = optax.apply_updates(current, delta)
            return (current, state), value

        return jax.lax.scan(
            update,
            (params, opt_state),
            None,
            length=int(config["max_steps"]),
        )

    initial = jnp.full(
        (2, layers),
        float(config["initial_filter_strength"]),
        dtype=jnp.float32,
    )
    (final, _), history = jax.jit(train)(initial)
    final = np.asarray(final)
    return {
        "final_a": final[0].reshape(layers // 2, 2),
        "final_b": final[1].reshape(layers // 2, 2),
        "energy_density_history": np.asarray(history),
    }
