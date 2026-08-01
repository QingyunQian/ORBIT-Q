import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    n = int(config["n_qubits"])
    state = config["dmrg_state"]
    tensors = list(getattr(state, "tensors", state))
    target_tensors = []
    for i, tensor in enumerate(tensors):
        x = jnp.asarray(getattr(tensor, "data", tensor))
        if x.ndim == 2:
            x = x[None, :, :] if i == 0 else x[:, :, None]
        target_tensors.append(x)
    target = tc.MPSCircuit(n, tensors=target_tensors, center_position=0)

    neel = []
    for i in range(n):
        x = np.zeros((1, 2, 1), dtype=np.complex64)
        x[0, i % 2, 0] = 1.0
        neel.append(x)
    bonds = list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))
    parameter_count = 15 * len(bonds)

    def overlap(params):
        circuit = tc.MPSCircuit(n, tensors=neel, center_position=0)
        k = 0
        for i in bonds:
            circuit.apply(
                tc.gates.su4_gate(params[k : k + 15]), i, i + 1
            )
            k += 15
        return circuit.proj_with_mps(target)

    def loss(params):
        z = overlap(params)
        return 1.0 - jnp.real(z * jnp.conj(z))

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    params = jnp.asarray(
        np.random.default_rng(int(config["seed"])).normal(
            0.0, float(config["initial_parameter_scale"]), parameter_count
        ),
        dtype=jnp.float32,
    )
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    steps = int(config["max_steps"])
    rate = float(config["learning_rate"])
    losses = np.empty(steps, dtype=np.float64)
    fidelities = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        value, grad = value_and_grad(params)
        value.block_until_ready()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mhat = m / (1.0 - 0.9 ** (step + 1))
        vhat = v / (1.0 - 0.999 ** (step + 1))
        params = params - rate * mhat / (jnp.sqrt(vhat) + 1e-8)
        losses[step] = float(value)
        fidelities[step] = 1.0 - losses[step]
    z = overlap(params)
    z.block_until_ready()
    return {
        "loss_history": losses,
        "fidelity_history": fidelities,
        "final_parameters": np.asarray(params, dtype=np.float64),
        "final_overlap_phase": float(np.angle(np.asarray(z))),
    }
