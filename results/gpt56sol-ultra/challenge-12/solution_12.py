import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _mps_tensors(state, n):
    """Convert quimb's named MPS legs to TensorCircuit's (left, phys, right)."""
    tensors = []
    for i in range(n):
        inds = []
        if i:
            inds.append(state.bond(i - 1, i))
        inds.append(state.site_ind(i))
        if i + 1 < n:
            inds.append(state.bond(i, i + 1))
        a = np.asarray(state[i].transpose(*inds).data)
        if i == 0:
            a = a.reshape(1, 2, a.shape[-1])
        elif i == n - 1:
            a = a.reshape(a.shape[0], 2, 1)
        tensors.append(jnp.asarray(a, dtype=jnp.complex64))
    return tensors


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy", preprocessing=True)

    n = int(config["n_qubits"])
    steps = int(config["max_steps"])
    layers = int(config["n_layers"])
    lr = float(config["learning_rate"])
    bonds = [
        (i, i + 1)
        for layer in range(layers)
        for i in range(layer & 1, n - 1, 2)
    ]

    target_tensors = _mps_tensors(config["dmrg_state"], n)
    target_bra = tc.Circuit(n, tensors=target_tensors).get_quvector().adjoint()

    def overlap(parameters):
        circuit = tc.Circuit(n)
        circuit.x(range(1, n, 2))
        for k, (i, j) in enumerate(bonds):
            circuit.su4(i, j, theta=parameters[k])
        return (target_bra @ circuit.get_quvector()).eval()

    def objective(parameters):
        z = overlap(parameters)
        fidelity = jnp.real(z * jnp.conj(z))
        return 1.0 - fidelity, fidelity

    value_and_grad = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def optimize(parameters):
        def adam_step(carry, t):
            parameters, first_moment, second_moment = carry
            (loss, fidelity), gradient = value_and_grad(parameters)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            iteration = t.astype(jnp.float32) + 1.0
            mhat = first_moment / (1.0 - jnp.power(0.9, iteration))
            vhat = second_moment / (1.0 - jnp.power(0.999, iteration))
            parameters = parameters - lr * mhat / (jnp.sqrt(vhat) + 1e-8)
            return (parameters, first_moment, second_moment), (loss, fidelity)

        zeros = jnp.zeros_like(parameters)
        return jax.lax.scan(
            adam_step,
            (parameters, zeros, zeros),
            jnp.arange(steps, dtype=jnp.int32),
        )

    rng = np.random.default_rng(int(config["seed"]))
    initial = rng.normal(
        scale=float(config["initial_parameter_scale"]), size=(len(bonds), 15)
    ).astype(np.float32)
    (final_parameters, _, _), histories = optimize(jnp.asarray(initial))
    final_overlap = overlap(final_parameters)
    final_parameters, histories, final_overlap = jax.device_get(
        (final_parameters, histories, final_overlap)
    )
    loss_history, fidelity_history = histories

    return {
        "loss_history": np.asarray(loss_history),
        "fidelity_history": np.asarray(fidelity_history),
        "final_parameters": np.asarray(final_parameters).reshape(-1),
        "final_overlap_phase": float(np.angle(final_overlap)),
    }
