import numpy as np
import jax
import optax
import tensorcircuit as tc


tc.set_backend("jax")
tc.set_dtype("complex64")
K = tc.backend


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    penalty = float(config["log_probability_weight"])
    updates_n = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    initial = tc.Circuit(n)
    initial.h(range(n))
    plus_state = initial.state()

    graph = tc.templates.graphs.Line1D(n, pbc=False)
    hamiltonian = tc.quantum.heisenberg_hamiltonian(
        graph, hzz=-1.0, hxx=0.0, hyy=0.0, hx=-field, sparse=True
    )
    n_bonds = sum(len(range(t % 2, n - 1, 2)) for t in range(n_steps))

    key = jax.random.PRNGKey(7)
    params = {
        "bond": 0.08 * jax.random.normal(key, (n_bonds, 2)),
        "rx": 0.08
        * jax.random.normal(jax.random.fold_in(key, 1), (n_steps, n)),
    }

    def objective(p):
        state = plus_state
        event_logs = []
        k = 0
        for t in range(n_steps):
            circuit = tc.Circuit(n, inputs=state)
            for q in range(t % 2, n - 1, 2):
                circuit.rxx(q, q + 1, theta=2.0 * p["bond"][k, 0])
                circuit.rzz(q, q + 1, theta=2.0 * p["bond"][k, 1])
                k += 1
            circuit.rx(range(n), theta=p["rx"][t])

            prefix_probability = K.convert_to_tensor(np.float32(1.0))
            for q in range(0, n, 2):
                circuit.mid_measurement(q, keep=0)
                projected = circuit.state()
                joint_probability = K.real(K.norm(projected) ** 2)
                event_probability = joint_probability / prefix_probability
                event_logs.append(K.log(event_probability + 1.0e-12))
                prefix_probability = joint_probability
            state = projected / K.sqrt(prefix_probability)

        final_circuit = tc.Circuit(n, inputs=state)
        energy_density = (
            tc.templates.measurements.operator_expectation(
                final_circuit, hamiltonian
            )
            / n
        )
        event_logs = K.stack(event_logs)
        mean_log_probability = K.mean(event_logs)
        success_probability = K.exp(K.sum(event_logs))
        loss = energy_density - penalty * mean_log_probability
        return loss, (
            energy_density,
            success_probability,
            mean_log_probability,
        )

    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(objective, has_aux=True)

    def update(carry, _):
        p, opt_state = carry
        (loss, aux), gradient = value_and_grad(p)
        delta, opt_state = optimizer.update(gradient, opt_state, p)
        p = optax.apply_updates(p, delta)
        return (p, opt_state), (*aux, loss)

    def train(p, opt_state):
        return jax.lax.scan(
            update, (p, opt_state), xs=None, length=updates_n
        )[1]

    energy, success, mean_log, loss = jax.jit(train)(params, optimizer_state)
    return {
        "energy_density_history": np.asarray(energy),
        "success_probability_history": np.asarray(success),
        "mean_log_probability_history": np.asarray(mean_log),
        "loss_history": np.asarray(loss),
    }
