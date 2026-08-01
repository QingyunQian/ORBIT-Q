import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


tc.set_backend("jax")


def run_solution(config):
    """Optimize the fixed-uniform measurement-trajectory objective."""
    n = config["n_data_qubits"]
    nl = config["n_layers"]
    nt = config["n_trajectories"]
    h = config["transverse_field"]
    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(rng.normal(0.0, config["initial_parameter_scale"], (nl, 6, n)))
    uniforms = jnp.asarray(rng.random((nt, nl, n)))

    def trajectory(p, status):
        # A measured ancilla is a Z eigenstate.  Thus RZZ(a, d, angle) is
        # exactly RZ(d, (+angle if a=0 else -angle)); this keeps the full
        # stochastic protocol in TensorCircuit without retaining dead wires.
        data = tc.Circuit(n)
        prior = jnp.zeros(n, dtype=jnp.int32)
        for layer in range(nl):
            anc = tc.Circuit(n)
            for i in range(n):
                anc.select_gate(prior[i], [tc.gates.i(), tc.gates.x()], i)
                anc.ry(i, theta=p[layer, 1, i])
                data.ry(i, theta=p[layer, 0, i])
            for i in range(n - 1):
                anc.cnot(i, i + 1)
            measured, _ = anc.measure(*range(n), status=status[layer])
            measured = measured.astype(jnp.int32)
            pre_ladder = jnp.concatenate((measured[:1], measured[1:] ^ measured[:-1]))
            for i in range(n):
                data.rz(i, theta=(1 - 2 * pre_ladder[i]) * p[layer, 2, i])
                angle = p[layer, 3, i] * (1 - measured[i]) + p[layer, 4, i] * measured[i]
                data.rz(i, theta=(1 - 2 * measured[i]) * angle)
            for i in range(n - 1):
                data.cnot(i, i + 1)
            for i in range(n):
                data.rz(i, theta=p[layer, 5, i])
            prior = measured
        energy = jnp.array(0.0)
        for i in range(n - 1):
            energy -= jnp.real(data.expectation_ps(z=[i, i + 1]))
        for i in range(n):
            energy -= h * jnp.real(data.expectation_ps(x=[i]))
        return energy

    batched = jax.vmap(trajectory, in_axes=(None, 0))
    value_grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(batched(p, uniforms))))
    moment = jnp.zeros_like(params)
    velocity = jnp.zeros_like(params)
    history = []
    for step in range(config["max_steps"]):
        value, grad = value_grad(params)
        history.append(value)
        moment = 0.9 * moment + 0.1 * grad
        velocity = 0.999 * velocity + 0.001 * grad * grad
        mhat = moment / (1.0 - 0.9 ** (step + 1))
        vhat = velocity / (1.0 - 0.999 ** (step + 1))
        params = params - config["learning_rate"] * mhat / (jnp.sqrt(vhat) + 1e-8)
    return {
        "energy_history": np.asarray(jnp.stack(history)),
        "final_trajectory_energies": np.asarray(batched(params, uniforms)),
    }
