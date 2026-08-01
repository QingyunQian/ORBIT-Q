"""TensorCircuit-NG solution for the controlled-Z hyperedge VQE benchmark."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _cmz_mpo(n, selected):
    """Embed TensorCircuit's native CMZ MPS factorization as a diagonal MPO."""
    gate = tc.gates.cmz_gate(len(selected))
    factors, selected = [np.asarray(edge.node1.tensor) for edge in gate.out_edges], set(selected)
    result, k = [], 0
    for q in range(n):
        if q not in selected:
            result.append(jnp.asarray(np.einsum("ab,ij->aijb", np.eye(2), np.eye(2))))
            continue
        factor, k = factors[k], k + 1
        if k == 1:
            tensor = np.zeros((1, 2, 2, 2), np.complex64)
            tensor[0, 0, 0], tensor[0, 1, 1] = factor[0], factor[1]
        elif factor.ndim == 2:
            tensor = np.zeros((2, 2, 2, 1), np.complex64)
            tensor[:, 0, 0, 0], tensor[:, 1, 1, 0] = factor[:, 0], factor[:, 1]
        else:
            tensor = np.zeros((2, 2, 2, 2), np.complex64)
            tensor[:, 0, 0, :], tensor[:, 1, 1, :] = factor[:, 0, :], factor[:, 1, :]
        result.append(jnp.asarray(tensor))
    return result


def run_solution(config):
    tc.set_backend("jax")
    n = config["n_qubits"]
    selected = tuple(config["selected_qubits"])
    layers = config["n_layers"]
    cmz = _cmz_mpo(n, selected)
    zz, x = config["zz_strength"], config["x_strength"]

    def energy(angles):
        circuit = tc.MPSCircuit(n)
        for q in config["initial_ones"]:
            circuit.x(q)
        for layer in range(layers):
            for q in range(n):
                circuit.rx(q, theta=angles[layer, q, 0])
                circuit.rz(q, theta=angles[layer, q, 1])
                circuit.ry(q, theta=angles[layer, q, 2])
            circuit.apply_MPO(cmz, 0)
        zz_value = sum(jnp.real(circuit.expectation(
            (tc.gates.z(), [q]), (tc.gates.z(), [q + 1]))) for q in range(n - 1))
        x_value = sum(jnp.real(circuit.expectation((tc.gates.x(), [q]))) for q in range(n))
        return (-zz * zz_value - x * x_value) / n

    energy_grad = jax.jit(jax.value_and_grad(energy))
    rng = np.random.default_rng(config["seed"])
    parameters = jnp.asarray(
        rng.normal(0.0, config["initial_parameter_scale"], (layers, n, 3)),
        dtype=jnp.float32,
    )
    first = jnp.zeros_like(parameters)
    second = jnp.zeros_like(parameters)
    rate, beta1, beta2, eps = config["learning_rate"], 0.9, 0.999, 1.0e-8

    def adam_step(carry, step):
        parameters, first, second = carry
        value, gradient = energy_grad(parameters)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        correction1 = 1.0 - beta1 ** (step + 1)
        correction2 = 1.0 - beta2 ** (step + 1)
        parameters = parameters - rate * (first / correction1) / (
            jnp.sqrt(second / correction2) + eps
        )
        return (parameters, first, second), value

    def optimize(parameters, first, second):
        return jax.lax.scan(adam_step, (parameters, first, second),
                            jnp.arange(config["max_steps"]))

    (parameters, _, _), history = jax.jit(optimize)(parameters, first, second)

    return {"energy_history": np.asarray(history), "final_parameters": np.asarray(parameters)}
