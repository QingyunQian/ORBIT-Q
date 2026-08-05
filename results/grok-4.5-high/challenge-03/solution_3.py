import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def run_solution(config):
    n = int(config["n_qubits"])
    g = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    log_w = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    def init_params(key):
        params = []
        for t in range(n_steps):
            nb = n // 2 if t % 2 == 0 else n // 2 - 1
            key, k1, k2 = jax.random.split(key, 3)
            bond = jax.random.normal(k1, (nb, 2), dtype=jnp.float32) * 0.1
            rx = jax.random.normal(k2, (n,), dtype=jnp.float32) * 0.1
            params.append({"bond": bond, "rx": rx})
        return params

    def energy_density(c):
        e = 0.0
        for i in range(n - 1):
            e = e - tc.backend.real(c.expectation_ps(z=[i, i + 1]))
        for i in range(n):
            e = e - g * tc.backend.real(c.expectation_ps(x=[i]))
        return e / n

    def evaluate(params):
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)

        logps = []
        for t in range(n_steps):
            if t % 2 == 0:
                bonds = [(2 * i, 2 * i + 1) for i in range(n // 2)]
            else:
                bonds = [(2 * i + 1, 2 * i + 2) for i in range(n // 2 - 1)]

            bp = params[t]["bond"]
            for b_idx, (i, j) in enumerate(bonds):
                c.exp1(i, j, theta=bp[b_idx, 0], unitary=tc.gates._xx_matrix)
                c.exp1(i, j, theta=bp[b_idx, 1], unitary=tc.gates._zz_matrix)

            rx = params[t]["rx"]
            for i in range(n):
                c.rx(i, theta=rx[i])

            for q in range(0, n, 2):
                ez = tc.backend.real(c.expectation_ps(z=[q]))
                p0 = 0.5 * (1.0 + ez)
                logps.append(jnp.log(p0 + 1e-12))
                c.post_select(q, keep=0)
                s = c.state()
                norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.real(jnp.conj(s) * s)), 1e-30))
                c = tc.Circuit(n, inputs=s / norm)

        logps = jnp.stack(logps)
        mean_logp = jnp.mean(logps)
        success_p = jnp.exp(jnp.sum(logps))
        ed = energy_density(c)
        loss = ed - log_w * mean_logp
        return loss, (ed, success_p, mean_logp)

    value_and_grad = jax.jit(jax.value_and_grad(evaluate, has_aux=True))

    optimizer = optax.adam(lr)
    params = init_params(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    energy_hist = np.empty(max_steps, dtype=np.float64)
    succ_hist = np.empty(max_steps, dtype=np.float64)
    meanlog_hist = np.empty(max_steps, dtype=np.float64)
    loss_hist = np.empty(max_steps, dtype=np.float64)

    for step in range(max_steps):
        (loss, (ed, sp, mlp)), grads = value_and_grad(params)
        energy_hist[step] = float(ed)
        succ_hist[step] = float(sp)
        meanlog_hist[step] = float(mlp)
        loss_hist[step] = float(loss)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "energy_density_history": energy_hist,
        "success_probability_history": succ_hist,
        "mean_log_probability_history": meanlog_hist,
        "loss_history": loss_hist,
    }
