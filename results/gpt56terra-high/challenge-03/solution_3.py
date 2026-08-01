import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from functools import partial


def run_solution(config):
    """Optimize the explicitly post-selected brickwork cooling circuit."""
    tc.set_backend("jax")
    n = config["n_qubits"]
    nt = config["n_steps"]
    field = config["transverse_field"]
    weight = config["log_probability_weight"]
    eps = 1e-12

    @partial(jax.jit, static_argnums=2)
    def cooling_step(state, layer, even):
        circuit = tc.Circuit(n, inputs=state)
        bonds = range(0, n - 1, 2) if even else range(1, n - 1, 2)
        for k, q in enumerate(bonds):
            circuit.exp1(q, q + 1, unitary=tc.gates._xx_matrix, theta=layer[k, 0])
            circuit.exp1(q, q + 1, unitary=tc.gates._zz_matrix, theta=layer[k, 1])
        for q in range(n):
            circuit.rx(q, theta=layer[n // 2, q])
        old_norm = jnp.array(1.0)
        logs = []
        for q in range(0, n, 2):
            circuit.mid_measurement(q, keep=0)
            new_norm = tc.backend.norm(circuit.state()) ** 2
            logs.append(jnp.log(new_norm / old_norm + eps))
            old_norm = new_norm
        return circuit.state() / tc.backend.sqrt(old_norm), jnp.stack(logs)

    @jax.jit
    def energy_density(state):
        energy = 0.0
        for q in range(n - 1):
            energy -= tc.backend.real(
                tc.expectation((tc.gates.z(), q), (tc.gates.z(), q + 1), ket=state,
                               normalization=True)
            )
        for q in range(n):
            energy -= field * tc.backend.real(
                tc.expectation((tc.gates.x(), q), ket=state, normalization=True)
            )
        return energy / n

    initial_circuit = tc.Circuit(n)
    for q in range(n):
        initial_circuit.h(q)
    initial_state = initial_circuit.state()

    def metrics(params):
        state, event_logs = initial_state, []
        for t in range(nt):
            state, logs = cooling_step(state, params[t], t % 2 == 0)
            event_logs.append(logs)
        mean_log = jnp.mean(jnp.concatenate(event_logs))
        return energy_density(state), mean_log

    def objective(params):
        energy, mean_log = metrics(params)
        return energy - weight * mean_log, (energy, mean_log)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    key = jax.random.PRNGKey(42)
    params = 0.1 * jax.random.normal(key, (nt, n // 2 + 1, n))
    first_moment, second_moment = jnp.zeros_like(params), jnp.zeros_like(params)
    energies, mean_logs, losses = [], [], []
    beta1, beta2 = 0.9, 0.999

    for step in range(config["max_steps"]):
        (loss, (energy, mean_log)), gradient = value_and_grad(params)
        energies.append(energy)
        mean_logs.append(mean_log)
        losses.append(loss)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        first_hat = first_moment / (1.0 - beta1 ** (step + 1))
        second_hat = second_moment / (1.0 - beta2 ** (step + 1))
        params = params - config["learning_rate"] * first_hat / (jnp.sqrt(second_hat) + 1e-8)

    energy_history = np.asarray(jnp.stack(energies))
    mean_log_history = np.asarray(jnp.stack(mean_logs))
    return {
        "energy_density_history": energy_history,
        "success_probability_history": np.asarray(jnp.exp(60.0 * mean_log_history)),
        "mean_log_probability_history": mean_log_history,
        "loss_history": np.asarray(jnp.stack(losses)),
    }
