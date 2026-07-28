import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = int(config["n_data_qubits"])
    nq = int(config["n_qubits"])
    layers = int(config["n_layers"])
    trajectories = int(config["n_trajectories"])
    steps = int(config["max_steps"])
    field = float(config["transverse_field"])

    rng = np.random.default_rng(int(config["seed"]))
    params = jnp.asarray(
        rng.normal(
            0.0,
            float(config["initial_parameter_scale"]),
            (6, layers, n),
        ),
        dtype=jnp.float32,
    )
    uniforms = jnp.asarray(
        rng.random((trajectories, layers, n)), dtype=jnp.float32
    )

    # RZZ is diagonal, so it cannot change ancilla Z probabilities.  This
    # ancilla-only circuit therefore samples exactly the same projective bits
    # as the full circuit, including the CNOT-ladder correlations.
    def sample_layer(angles, status, previous):
        c = tc.Circuit(n)
        for i in range(n):
            c.conditional_gate(previous[i], [tc.gates.i(), tc.gates.x()], i)
        for i in range(n):
            c.ry(i, theta=angles[i])
        for i in range(n - 1):
            c.cnot(i, i + 1)
        measured, _ = c.measure(*range(n), status=status)
        return tc.backend.cast(measured, "int32")

    sample_batch = jax.jit(
        jax.vmap(sample_layer, in_axes=(None, 0, 0))
    )
    previous = jnp.zeros((trajectories, n), dtype=jnp.int32)
    measured_layers = []
    for layer in range(layers):
        previous = sample_batch(
            params[1, layer], uniforms[:, layer], previous
        )
        measured_layers.append(previous)
    measured = jnp.stack(measured_layers, axis=1)

    # For the ordered CNOT ladder, m_i=b_0 xor ... xor b_i.  Projecting the
    # corresponding pre-ladder bit b_i turns RZZ(a_i,d_i) exactly into
    # RZ((-1)**b_i theta) on d_i; the discarded scalar is removed by the
    # projective normalization.
    pre_bits = jnp.concatenate(
        [
            measured[:, :, :1],
            jnp.bitwise_xor(measured[:, :, 1:], measured[:, :, :-1]),
        ],
        axis=2,
    )

    def trajectory_energy(p, bits, before_ladder):
        c = tc.Circuit(nq)
        old_bits = jnp.zeros(n, dtype=jnp.int32)
        for layer in range(layers):
            for i in range(n):
                c.ry(i, theta=p[0, layer, i])
            for i in range(n):
                sign = 1 - 2 * before_ladder[layer, i]
                c.rz(i, theta=sign * p[2, layer, i])
            for i in range(n):
                changed = jnp.bitwise_xor(old_bits[i], bits[layer, i])
                c.conditional_gate(
                    changed, [tc.gates.i(), tc.gates.x()], n + i
                )
                c.conditional_gate(
                    bits[layer, i],
                    [
                        tc.gates.rzz(theta=p[3, layer, i]),
                        tc.gates.rzz(theta=p[4, layer, i]),
                    ],
                    n + i,
                    i,
                )
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rz(i, theta=p[5, layer, i])
            old_bits = bits[layer]

        energy = jnp.asarray(0.0, dtype=jnp.float32)
        for i in range(n - 1):
            energy -= tc.backend.real(c.expectation_ps(z=[i, i + 1]))
        for i in range(n):
            energy -= field * tc.backend.real(c.expectation_ps(x=[i]))
        return energy

    all_energies = jax.vmap(
        trajectory_energy, in_axes=(None, 0, 0)
    )

    def objective(p):
        return jnp.mean(all_energies(p, measured, pre_bits))

    value_and_grad = jax.value_and_grad(objective)
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)

    def update(carry, _):
        p, state = carry
        value, gradient = value_and_grad(p)
        updates, state = optimizer.update(gradient, state, p)
        return (optax.apply_updates(p, updates), state), value

    def optimize(p, state):
        return jax.lax.scan(update, (p, state), None, length=steps)

    (params, _), history = jax.jit(optimize)(params, opt_state)
    final_energies = jax.jit(all_energies)(params, measured, pre_bits)
    return {
        "energy_history": np.asarray(history),
        "final_trajectory_energies": np.asarray(final_energies),
    }
