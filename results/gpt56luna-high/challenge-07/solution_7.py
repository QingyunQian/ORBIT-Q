import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


tc.set_backend("jax")
_B = tc.backend


def _trajectory_energy(params, uniforms, layers, ndata, nqubits, field):
    c = tc.Circuit(nqubits)
    norm = _B.convert_to_tensor(1.0)
    for layer in range(layers):
        for i in range(ndata):
            c.ry(i, theta=params[layer, i])
        for i in range(ndata):
            c.ry(ndata + i, theta=params[layer, ndata + i])
        for i in range(ndata):
            c.rzz(ndata + i, i, theta=params[layer, 2 * ndata + i])
        for i in range(ndata - 1):
            c.cnot(ndata + i, ndata + i + 1)

        for i in range(ndata):
            z = _B.real(c.expectation_ps(z=[ndata + i]))
            p1 = (norm - z) / (2.0 * norm)
            bit = _B.cast(uniforms[layer * ndata + i] < p1, "int32")
            b = _B.cast(bit, "float32")
            zero = _B.convert_to_tensor(0.0)
            # A projective measurement, with the sampled branch selected by
            # the fixed trajectory uniform.  The norm is tracked explicitly
            # because this projector is intentionally not normalized.
            projector = _B.stack((_B.stack((1.0 - b, zero)),
                                  _B.stack((zero, b))))
            c.any(ndata + i, unitary=projector)
            norm = norm * _B.where(bit > 0, p1, 1.0 - p1)
            c.conditional_gate(
                bit,
                [tc.gates.rzz(theta=params[layer, 3 * ndata + i]),
                 tc.gates.rzz(theta=params[layer, 4 * ndata + i])],
                ndata + i,
                i,
            )
        for i in range(ndata - 1):
            c.cnot(i, i + 1)
        for i in range(ndata):
            c.rz(i, theta=params[layer, 5 * ndata + i])

    den = _B.real(_B.norm(c.state()) ** 2)
    energy = _B.convert_to_tensor(0.0)
    for i in range(ndata - 1):
        energy -= _B.real(c.expectation_ps(z=[i, i + 1]))
    for i in range(ndata):
        energy -= field * _B.real(c.expectation_ps(x=[i]))
    return energy / den


def run_solution(config):
    ndata = int(config["n_data_qubits"])
    layers = int(config["n_layers"])
    nqubits = int(config["n_qubits"])
    ntraj = int(config["n_trajectories"])
    steps = int(config["max_steps"])
    scale = float(config["initial_parameter_scale"])
    field = float(config["transverse_field"])
    rng = np.random.default_rng(int(config["seed"]))
    params = jnp.asarray(
        rng.normal(0.0, scale, (layers, 6 * ndata)), dtype=jnp.float32
    )
    uniforms = jnp.asarray(
        np.random.default_rng(int(config["seed"]) + 1).random(
            (ntraj, layers * ndata)
        ), dtype=jnp.float32
    )

    def objective(p, us):
        def scan_step(total, u):
            value = _trajectory_energy(p, u, layers, ndata, nqubits, field)
            return total + value, value

        _, values = jax.lax.scan(
            scan_step, jnp.asarray(0.0, dtype=p.dtype), us
        )
        return jnp.mean(values)

    value_grad = _B.jit(_B.value_and_grad(objective))
    history = []
    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(steps):
        value, grad = value_grad(params, uniforms)
        history.append(value)
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * grad * grad
        bias1 = 1.0 - beta1 ** (step + 1)
        bias2 = 1.0 - beta2 ** (step + 1)
        params = params - float(config["learning_rate"]) * (
            (first_moment / bias1)
            / (jnp.sqrt(second_moment / bias2) + eps)
        )

    def final_values(p, us):
        def scan_step(_, u):
            value = _trajectory_energy(p, u, layers, ndata, nqubits, field)
            return 0.0, value

        _, values = jax.lax.scan(scan_step, 0.0, us)
        return values

    final_values_jit = _B.jit(final_values)
    final_energies = final_values_jit(params, uniforms)
    return {
        "energy_history": np.asarray(history, dtype=np.float32),
        "final_trajectory_energies": np.asarray(final_energies, dtype=np.float32),
    }
