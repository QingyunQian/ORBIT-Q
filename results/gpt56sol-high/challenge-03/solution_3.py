import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    """Optimize the explicitly post-selected TensorCircuit trajectory."""
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    depth = int(config["n_steps"])
    field = float(config["transverse_field"])
    weight = float(config["log_probability_weight"])
    steps = int(config["max_steps"])

    # TensorCircuit sparse operator for -sum(ZZ) - field*sum(X).
    paulis, coefficients = [], []
    for i in range(n - 1):
        term = [0] * n
        term[i] = term[i + 1] = 3
        paulis.append(term)
        coefficients.append(-1.0)
    for i in range(n):
        term = [0] * n
        term[i] = 1
        paulis.append(term)
        coefficients.append(-field)
    hamiltonian = tc.quantum.PauliStringSum2COO(paulis, coefficients)

    bonds_per_step = [(n - (t & 1)) // 2 for t in range(depth)]
    n_bonds = sum(bonds_per_step)
    n_parameters = 2 * n_bonds + depth * n

    initial_circuit = tc.Circuit(n)
    initial_circuit.h(range(n))
    plus_state = initial_circuit.state()

    def objective(parameters):
        state = plus_state
        log_probabilities = []
        bond_parameter = 0
        for t in range(depth):
            circuit = tc.Circuit(n, inputs=state)
            for q in range(t & 1, n - 1, 2):
                circuit.exp1(
                    q,
                    q + 1,
                    unitary=tc.gates._xx_matrix,
                    theta=parameters[bond_parameter],
                )
                circuit.exp1(
                    q,
                    q + 1,
                    unitary=tc.gates._zz_matrix,
                    theta=parameters[n_bonds + bond_parameter],
                )
                bond_parameter += 1
            start = 2 * n_bonds + t * n
            circuit.rx(range(n), theta=parameters[start : start + n])
            state = circuit.state()

            # Each normalization makes the next probability explicitly conditional.
            for q in range(0, n, 2):
                selected = tc.Circuit(n, inputs=state)
                selected.mid_measurement(q, keep=0)
                projected_state = selected.state()
                probability = jnp.real(jnp.vdot(projected_state, projected_state))
                log_probabilities.append(jnp.log(probability + 1.0e-12))
                state = projected_state / jnp.sqrt(probability)

        final_circuit = tc.Circuit(n, inputs=state)
        energy_density = (
            tc.templates.measurements.operator_expectation(final_circuit, hamiltonian)
            / n
        )
        logs = jnp.stack(log_probabilities)
        mean_log_probability = jnp.mean(logs)
        success_probability = jnp.exp(logs.size * mean_log_probability)
        loss = energy_density - weight * mean_log_probability
        return loss, (energy_density, success_probability, mean_log_probability)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    optimizer = optax.adam(float(config["learning_rate"]))
    # Small deterministic perturbations avoid the zero-gradient symmetric point.
    parameters = 0.02 * jax.random.normal(jax.random.PRNGKey(3), (n_parameters,))
    optimizer_state = optimizer.init(parameters)

    def update(carry, unused):
        parameters, optimizer_state = carry
        (loss, metrics), gradient = value_and_grad(parameters)
        updates, optimizer_state = optimizer.update(
            gradient, optimizer_state, parameters
        )
        parameters = optax.apply_updates(parameters, updates)
        return (parameters, optimizer_state), (*metrics, loss)

    @jax.jit
    def train(parameters, optimizer_state):
        return jax.lax.scan(update, (parameters, optimizer_state), None, length=steps)

    (_, _), histories = train(parameters, optimizer_state)
    energy, success, mean_log, loss = (np.asarray(x) for x in histories)
    return {
        "energy_density_history": energy,
        "success_probability_history": success,
        "mean_log_probability_history": mean_log,
        "loss_history": loss,
    }
