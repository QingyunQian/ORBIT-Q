import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    K = tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("plain")

    n = int(config["n_qubits"])
    nblocks = int(config["n_layers"]) // 2
    subsystem = tuple(range(int(config["subsystem_size"])))
    targets = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    entropy_weight = float(config["entropy_weight"])

    even_bonds = tuple(range(0, n - 1, 2))
    odd_bonds = tuple(range(1, n - 1, 2))
    block_size = 4 * n + 3 * (len(even_bonds) + len(odd_bonds))
    if targets.shape != (nblocks,):
        raise ValueError("target_entropies must contain one value per block")

    initial_circuit = tc.Circuit(n)
    for q in range(1, n, 2):
        initial_circuit.x(q)
    initial_state = initial_circuit.state()

    pauli_strings, coefficients = [], []
    for q in range(n - 1):
        for pauli, coefficient in (
            (1, 1.0),
            (2, 1.0),
            (3, float(config["zz_anisotropy"])),
        ):
            term = [0] * n
            term[q] = term[q + 1] = pauli
            pauli_strings.append(term)
            coefficients.append(coefficient)
    for q in range(n):
        term = [0] * n
        term[q] = 3
        pauli_strings.append(term)
        coefficients.append(float(config["staggered_field"]) * (-1.0) ** q)
    hamiltonian_mvp = tc.quantum.PauliStringSum2MVP(
        pauli_strings, coefficients
    )

    def apply_block(state, parameters):
        circuit = tc.Circuit(n, inputs=state)
        k = 0
        for q in range(n):
            circuit.ry(q, theta=parameters[k])
            circuit.rz(q, theta=parameters[k + 1])
            k += 2
        for q in even_bonds:
            circuit.rxx(q, q + 1, theta=2.0 * parameters[k])
            circuit.ryy(q, q + 1, theta=2.0 * parameters[k + 1])
            circuit.rzz(q, q + 1, theta=2.0 * parameters[k + 2])
            k += 3
        for q in range(n):
            circuit.ry(q, theta=parameters[k])
            circuit.rz(q, theta=parameters[k + 1])
            k += 2
        for q in odd_bonds:
            circuit.rxx(q, q + 1, theta=2.0 * parameters[k])
            circuit.ryy(q, q + 1, theta=2.0 * parameters[k + 1])
            circuit.rzz(q, q + 1, theta=2.0 * parameters[k + 2])
            k += 3
        state = circuit.state()
        rho = tc.quantum.reduced_density_matrix(
            state, subsystem_to_keep=subsystem
        )
        return state, tc.quantum.renyi_entropy(rho, 2)

    def objective(parameters):
        state, entropies = jax.lax.scan(
            apply_block, initial_state, parameters
        )
        h_state = hamiltonian_mvp(state)
        energy_density = K.real(K.sum(K.conj(state) * h_state)) / n
        entropy_mse = K.mean((entropies - targets) ** 2)
        loss = energy_density + entropy_weight * entropy_mse
        return loss, (energy_density, entropy_mse, entropies)

    optimizer = optax.adam(float(config["learning_rate"]))

    def train(parameters):
        optimizer_state = optimizer.init(parameters)

        def update(carry, _):
            parameters, optimizer_state = carry
            (loss, (energy, mse, entropies)), gradient = jax.value_and_grad(
                objective, has_aux=True
            )(parameters)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, parameters
            )
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, optimizer_state), (energy, loss, mse, entropies)

        _, histories = jax.lax.scan(
            update,
            (parameters, optimizer_state),
            None,
            length=int(config["max_steps"]),
        )
        return histories

    parameters = 0.02 * jax.random.normal(
        jax.random.PRNGKey(0), (nblocks, block_size), dtype=jnp.float32
    )
    energy, loss, mse, entropy = jax.jit(train)(parameters)
    return {
        "energy_density_history": np.asarray(energy),
        "loss_history": np.asarray(loss),
        "entropy_mse_history": np.asarray(mse),
        "entropy_history": np.asarray(entropy),
    }
