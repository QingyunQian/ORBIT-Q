import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")
K = tc.backend


def run_solution(config):
    n = int(config["n_qubits"])
    g = np.float32(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_s = np.float32(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    lr = np.float32(config["learning_rate"])

    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)

    def filter_x(a):
        ca = jnp.cosh(a)
        sa = jnp.sinh(a)
        return jnp.stack([jnp.stack([ca, sa]), jnp.stack([sa, ca])]).astype(
            jnp.complex64
        )

    def filter_zz(b):
        z = jnp.zeros((), dtype=b.dtype)
        eb = jnp.exp(b)
        emb = jnp.exp(-b)
        return jnp.stack(
            [
                jnp.stack([eb, z, z, z]),
                jnp.stack([z, emb, z, z]),
                jnp.stack([z, z, emb, z]),
                jnp.stack([z, z, z, eb]),
            ]
        ).astype(jnp.complex64)

    def energy_density(state):
        c = tc.Circuit(n, inputs=state)
        e = jnp.float32(0.0)
        for i in range(n - 1):
            e = e - jnp.real(c.expectation_ps(z=[i, i + 1]))
        for i in range(n):
            e = e - g * jnp.real(c.expectation_ps(x=[i]))
        return e / jnp.float32(n)

    def loss(params):
        params_a = params[0]
        params_b = params[1]
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
        state = c.state()
        for l in range(n_layers):
            c = tc.Circuit(n, inputs=state)
            gx = filter_x(params_a[l])
            gz = filter_zz(params_b[l])
            for i in range(n):
                c.unitary(i, unitary=gx)
            bonds = range(0, n - 1, 2) if (l % 2 == 0) else range(1, n - 1, 2)
            for i in bonds:
                c.unitary(i, i + 1, unitary=gz)
            state = c.state()
            state = state / jnp.linalg.norm(state)
        return energy_density(state)

    loss_vag = K.value_and_grad(loss)

    @jax.jit
    def adam_step(params, m, v, t):
        val, grad = loss_vag(params)
        t = t + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
        return params, m, v, t, val

    params = jnp.ones((2, n_layers), dtype=jnp.float32) * init_s
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    t = jnp.array(0, dtype=jnp.float32)

    # Warmup / compile, then reset so history starts from the true initial params.
    params, m, v, t, warmup_val = adam_step(params, m, v, t)
    warmup_val.block_until_ready()

    params = jnp.ones((2, n_layers), dtype=jnp.float32) * init_s
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    t = jnp.array(0, dtype=jnp.float32)

    hist = []
    for _ in range(max_steps):
        params, m, v, t, val = adam_step(params, m, v, t)
        hist.append(val)

    energy_density_history = np.asarray(jnp.stack(hist), dtype=np.float64)
    params_np = np.asarray(params, dtype=np.float64)
    final_a = params_np[0].reshape(n_layers // 2, 2)
    final_b = params_np[1].reshape(n_layers // 2, 2)

    return {
        "final_a": final_a,
        "final_b": final_b,
        "energy_density_history": energy_density_history,
    }
