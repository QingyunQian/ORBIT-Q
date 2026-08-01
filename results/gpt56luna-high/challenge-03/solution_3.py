import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    b = tc.backend
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    steps = int(config["n_steps"])
    log_weight = float(config["log_probability_weight"])
    updates = int(config["max_steps"])
    lr = float(config["learning_rate"])
    even_bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]
    max_bonds = len(even_bonds)
    width = 2 * max_bonds + n

    def make_circuit(state, p, bonds):
        c = tc.Circuit(n, inputs=state)
        for j, (q0, q1) in enumerate(bonds):
            c.exp1(q0, q1, unitary=tc.gates._xx_matrix, theta=p[2 * j])
            c.exp1(q0, q1, unitary=tc.gates._zz_matrix, theta=p[2 * j + 1])
        for q in range(n):
            c.rx(q, theta=p[2 * max_bonds + q])
        return c

    def step(carry, values):
        state, norm2 = carry
        p, odd = values

        def run(bonds):
            c = make_circuit(state, p, bonds)
            logs = []
            current = norm2
            for q in range(0, n, 2):
                c.mid_measurement(q, keep=0)
                state_after = c.wavefunction()
                next_norm = b.real(b.sum(b.abs(state_after) ** 2))
                logs.append(b.log(next_norm / current + 1.0e-12))
                current = next_norm
            return (state_after, current), b.stack(logs)

        return jax.lax.cond(
            odd, lambda _: run(odd_bonds), lambda _: run(even_bonds), operand=None
        )

    def objective(params):
        initial = tc.Circuit(n)
        initial.h(range(n))
        state0 = initial.wavefunction()
        carry, event_logs = jax.lax.scan(
            step, (state0, b.convert_to_tensor(1.0)),
            (params, jnp.arange(steps, dtype=jnp.int32) % 2),
        )
        state, norm2 = carry
        final = tc.Circuit(n, inputs=state)
        energy = b.convert_to_tensor(0.0)
        for q in range(n - 1):
            energy -= b.real(final.expectation_ps(z=[q, q + 1], reuse=True))
        for q in range(n):
            energy -= field * b.real(final.expectation_ps(x=[q], reuse=True))
        energy_density = energy / (n * norm2)
        mean_log = b.mean(event_logs)
        loss = energy_density - log_weight * mean_log
        success = b.exp((n // 2) * steps * mean_log)
        return loss, (energy_density, success, mean_log)

    rng = np.random.default_rng(1234)
    params = b.convert_to_tensor(rng.normal(0.0, 0.05, (steps, width)).astype(np.float32))
    m = b.zeros_like(params)
    v = b.zeros_like(params)
    value_and_grad = b.jit(b.value_and_grad(objective, has_aux=True))
    energy_history = []
    success_history = []
    mean_log_history = []
    loss_history = []
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    for update in range(1, updates + 1):
        (loss, metrics), grads = value_and_grad(params)
        energy, success, mean_log = metrics
        energy_history.append(energy)
        success_history.append(success)
        mean_log_history.append(mean_log)
        loss_history.append(loss)
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads * grads
        mhat = m / (1.0 - beta1**update)
        vhat = v / (1.0 - beta2**update)
        params = params - lr * mhat / (b.sqrt(vhat) + eps)
    return {
        "energy_density_history": np.asarray(energy_history),
        "success_probability_history": np.asarray(success_history),
        "mean_log_probability_history": np.asarray(mean_log_history),
        "loss_history": np.asarray(loss_history),
    }
