import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")

    n = int(config["n_qubits"])
    n_blocks = int(config["n_layers"]) // 2
    keep = tuple(range(int(config["subsystem_size"])))
    targets = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    entropy_weight = float(config["entropy_weight"])
    steps = int(config["max_steps"])
    even_bonds = tuple(range(0, n - 1, 2))
    odd_bonds = tuple(range(1, n - 1, 2))
    block_size = 4 * n + 3 * (len(even_bonds) + len(odd_bonds))

    paulis, coefficients = [], []
    for q in range(n - 1):
        for label, coefficient in ((1, 1.0), (2, 1.0), (3, config["zz_anisotropy"])):
            term = [0] * n
            term[q] = term[q + 1] = label
            paulis.append(term)
            coefficients.append(coefficient)
    for q in range(n):
        term = [0] * n
        term[q] = 3
        paulis.append(term)
        coefficients.append(config["staggered_field"] * (-1) ** q)
    hamiltonian = tc.quantum.PauliStringSum2COO(paulis, coefficients)

    initial_circuit = tc.Circuit(n)
    for q in range(1, n, 2):
        initial_circuit.x(q)
    initial_state = initial_circuit.state()
    generators = (tc.gates._xx_matrix, tc.gates._yy_matrix, tc.gates._zz_matrix)

    def circuit_block(state, block_parameters):
        circuit = tc.Circuit(n, inputs=state)
        k = 0
        for q in range(n):
            circuit.ry(q, theta=block_parameters[k])
            circuit.rz(q, theta=block_parameters[k + 1])
            k += 2
        for q in even_bonds:
            for generator in generators:
                circuit.exp1(q, q + 1, unitary=generator, theta=block_parameters[k])
                k += 1
        for q in range(n):
            circuit.ry(q, theta=block_parameters[k])
            circuit.rz(q, theta=block_parameters[k + 1])
            k += 2
        for q in odd_bonds:
            for generator in generators:
                circuit.exp1(q, q + 1, unitary=generator, theta=block_parameters[k])
                k += 1
        state = circuit.state()
        rho = tc.quantum.reduced_density_matrix(state, subsystem_to_keep=keep)
        entropy = -jnp.log(jnp.real(jnp.trace(rho @ rho)))
        return state, entropy

    def objective(parameters):
        final_state, entropies = jax.lax.scan(circuit_block, initial_state, parameters)
        final_circuit = tc.Circuit(n, inputs=final_state)
        energy_density = (
            tc.templates.measurements.operator_expectation(final_circuit, hamiltonian) / n
        )
        entropy_mse = jnp.mean((entropies - targets) ** 2)
        loss = energy_density + entropy_weight * entropy_mse
        return loss, (energy_density, entropy_mse, entropies)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    optimizer = optax.adam(float(config["learning_rate"]))
    parameters = 0.02 * jax.random.normal(
        jax.random.PRNGKey(0), (n_blocks, block_size), dtype=jnp.float32
    )
    optimizer_state = optimizer.init(parameters)

    def update(carry, unused):
        parameters, optimizer_state = carry
        (loss, auxiliary), gradient = value_and_grad(parameters)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        energy_density, entropy_mse, entropies = auxiliary
        return (parameters, optimizer_state), (energy_density, loss, entropy_mse, entropies)

    @jax.jit
    def train(parameters, optimizer_state):
        return jax.lax.scan(update, (parameters, optimizer_state), None, length=steps)

    _, histories = train(parameters, optimizer_state)
    energy, loss, entropy_mse, entropy = jax.device_get(histories)
    return {
        "energy_density_history": np.asarray(energy),
        "loss_history": np.asarray(loss),
        "entropy_mse_history": np.asarray(entropy_mse),
        "entropy_history": np.asarray(entropy),
    }
