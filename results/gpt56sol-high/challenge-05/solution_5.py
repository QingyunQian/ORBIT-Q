import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy", preprocessing=True)
    K = tc.backend
    n = int(config["n_qubits"])
    nlayers = int(config["n_layers"])
    field = float(config["transverse_field"])

    c0 = tc.Circuit(n)
    c0.h(range(n))
    initial_state = c0.state()

    def energy_density(params):
        state = initial_state
        a = params[0].reshape(-1)
        b = params[1].reshape(-1)
        for layer in range(nlayers):
            c = tc.Circuit(n, inputs=state)
            for q in range(n):
                c.exp1(q, theta=1j * a[layer], unitary=tc.gates._x_matrix)
            for q in range(layer & 1, n - 1, 2):
                c.exp1(q, q + 1, theta=1j * b[layer], unitary=tc.gates._zz_matrix)
            state = c.state()
            state = state / K.norm(state)

        c = tc.Circuit(n, inputs=state)
        energy = 0.0
        for q in range(n - 1):
            energy -= c.expectation_ps(z=[q, q + 1])
        for q in range(n):
            energy -= field * c.expectation_ps(x=[q])
        return K.real(energy) / n

    steps = int(config["max_steps"])
    params = jnp.full(
        (2, nlayers // 2, 2),
        float(config["initial_filter_strength"]),
        dtype=jnp.float32,
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(energy_density)

    def update(carry, _):
        p, state = carry
        loss, grad = value_and_grad(p)
        updates, state = optimizer.update(grad, state, p)
        return (optax.apply_updates(p, updates), state), loss

    (params, _), history = jax.jit(
        lambda p, s: jax.lax.scan(update, (p, s), None, length=steps)
    )(params, opt_state)
    params, history = jax.device_get((params, history))
    return {
        "final_a": np.asarray(params[0]),
        "final_b": np.asarray(params[1]),
        "energy_density_history": np.asarray(history),
    }
