"""TensorCircuit MPS variational overlap optimization."""

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _tc_mps_tensors(mps, n):
    """Adapt quimb's named-index MPS tensors to TensorCircuit's (l, p, r)."""
    tensors = []
    for i, tensor in enumerate(mps.tensors):
        inds = list(tensor.inds)
        physical = inds.index(mps.site_ind(i))
        left = next(
            (inds.index(x) for x in inds if i and x in mps.tensors[i - 1].inds),
            None,
        )
        right = next(
            (inds.index(x) for x in inds if i + 1 < n and x in mps.tensors[i + 1].inds),
            None,
        )
        order = ([] if left is None else [left]) + [physical]
        order += [] if right is None else [right]
        array = np.transpose(np.asarray(tensor.data), order)
        if left is None:
            array = array[None, :, :]
        if right is None:
            array = array[:, :, None]
        tensors.append(jnp.asarray(array, dtype=jnp.complex64))
    return tensors


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    steps = int(config["max_steps"])
    target = tc.MPSCircuit(
        n, tensors=_tc_mps_tensors(config["dmrg_state"], n), center_position=0
    )

    zero = jnp.array([1.0, 0.0], dtype=jnp.complex64)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex64)
    neel = tuple((zero if i % 2 == 0 else one).reshape(1, 2, 1) for i in range(n))
    bonds = tuple(range(0, n - 1, 2)) + tuple(range(1, n - 1, 2))
    size = 15 * len(bonds)

    def overlap(parameters):
        circuit = tc.MPSCircuit(n, tensors=neel, center_position=0)
        for gate_number, site in enumerate(bonds):
            start = 15 * gate_number
            circuit.apply(tc.gates.su4(parameters[start : start + 15]), site, site + 1)
        return circuit.proj_with_mps(target)

    def fidelity(parameters):
        value = overlap(parameters)
        return jnp.real(value * jnp.conj(value))

    value_and_grad = jax.value_and_grad(fidelity)
    lr = float(config["learning_rate"])

    def adam_step(carry, _):
        parameters, first, second, count = carry
        value, gradient = value_and_grad(parameters)
        count = count + 1
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        parameters = parameters + lr * (first / (1.0 - 0.9**count)) / (
            jnp.sqrt(second / (1.0 - 0.999**count)) + 1e-8
        )
        return (parameters, first, second, count), (1.0 - value, value)

    def optimize(parameters):
        initial = (parameters, jnp.zeros_like(parameters), jnp.zeros_like(parameters), jnp.array(0))
        return jax.lax.scan(adam_step, initial, xs=None, length=steps)

    parameters = jax.random.normal(jax.random.PRNGKey(int(config["seed"])), (size,))
    parameters = parameters * float(config["initial_parameter_scale"])
    (parameters, _, _, _), (losses, fidelities) = jax.jit(optimize)(parameters)
    final_overlap = overlap(parameters)
    parameters, losses, fidelities, final_overlap = jax.device_get(
        (parameters, losses, fidelities, final_overlap)
    )
    return {
        "loss_history": np.asarray(losses, dtype=float),
        "fidelity_history": np.asarray(fidelities, dtype=float),
        "final_parameters": np.asarray(parameters, dtype=float),
        "final_overlap_phase": float(np.angle(final_overlap)),
    }
