"""TensorCircuit-NG solution for the measurement-feedback VQE benchmark."""

import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    tc.set_backend("jax")
    n = config["n_data_qubits"]
    na = config["n_ancilla_qubits"]
    nl = config["n_layers"]
    nt = config["n_trajectories"]
    nq = config["n_qubits"]
    h = config["transverse_field"]

    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(
        rng.normal(0.0, config["initial_parameter_scale"], (6, nl, n)),
        dtype=jnp.float32,
    )
    uniforms = jnp.asarray(rng.random((nt, nl, na)), dtype=jnp.float32)
    projectors = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]

    def trajectory(p, u):
        c = tc.Circuit(nq)
        for layer in range(nl):
            for i in range(n):
                c.ry(i, theta=p[0, layer, i])
            for i in range(na):
                c.ry(n + i, theta=p[1, layer, i])
            for i in range(n):
                c.rzz(n + i, i, theta=p[2, layer, i])
            for i in range(na - 1):
                c.cnot(n + i, n + i + 1)
            bits, probability = c.measure_jit(
                *range(n, n + na), with_prob=True, status=u[layer]
            )
            scaled = [x / jnp.sqrt(probability) for x in projectors]
            c.conditional_gate(bits[0], scaled, n)
            for i in range(1, na):
                c.conditional_gate(bits[i], projectors, n + i)
            for i in range(n):
                angle = jnp.where(bits[i] == 0, p[3, layer, i], p[4, layer, i])
                c.rzz(n + i, i, theta=angle)
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rz(i, theta=p[5, layer, i])
        e = sum(-jnp.real(c.expectation_ps(z=[i, i + 1])) for i in range(n - 1))
        e += sum(-h * jnp.real(c.expectation_ps(x=[i])) for i in range(n))
        return e

    energies = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))
    loss = lambda p: jnp.mean(energies(p, uniforms))
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(params)

    def step(carry, _):
        p, s = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, s = optimizer.update(grad, s, p)
        return (optax.apply_updates(p, updates), s), value

    optimize = jax.jit(lambda p, s: jax.lax.scan(
        step, (p, s), None, length=config["max_steps"]
    ))
    (params, state), history = optimize(params, state)
    final = energies(params, uniforms)
    return {
        "energy_history": np.asarray(jax.device_get(history)),
        "final_trajectory_energies": np.asarray(jax.device_get(final)),
    }
