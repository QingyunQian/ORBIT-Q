"""
Task Suite Problem 1: DMRG-MPS input with variational circuit refinement.

The DMRG state is injected into a regular TensorCircuit Circuit. The solution
returns NumPy values only; external validation lives in evaluate_1.py.
"""

import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.templates.measurements import mpo_expectation

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def parameter_count(config):
    count = 0
    for layer in range(config["n_layers"]):
        count += 3 * config["n_qubits"]
        count += 3 * len(range(layer % 2, config["n_qubits"] - 1, 2))
    return count


def initial_parameters(config):
    rng = np.random.default_rng(1234)
    params = rng.normal(scale=1e-4, size=parameter_count(config)).astype(np.float32)
    return K.convert_to_tensor(params)


def tfim_mpo(config):
    eye = np.eye(2, dtype=np.complex64)
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex64)
    bulk[0, 0] = eye
    bulk[1, 0] = z_gate
    bulk[2, 0] = -config["field"] * x_gate
    bulk[2, 1] = -z_gate
    bulk[2, 2] = eye

    tensors = [bulk[2:3]] + [bulk] * (config["n_qubits"] - 2) + [bulk[:, 0:1]]
    nodes = [tc.quantum.Node(K.convert_to_tensor(tensor)) for tensor in tensors]
    for i in range(config["n_qubits"] - 1):
        nodes[i][1] ^ nodes[i + 1][0]

    return tc.quantum.QuOperator(
        out_edges=[node[2] for node in nodes],
        in_edges=[node[3] for node in nodes],
        ref_nodes=nodes,
        ignore_edges=[nodes[0][0], nodes[-1][1]],
    )


def apply_variational_layers(circuit, params, config):
    offset = 0
    for layer in range(config["n_layers"]):
        for i in range(config["n_qubits"]):
            circuit.rz(i, theta=params[offset])
            circuit.ry(i, theta=params[offset + 1])
            circuit.rz(i, theta=params[offset + 2])
            offset += 3

        for i in range(layer % 2, config["n_qubits"] - 1, 2):
            circuit.rxx(i, i + 1, theta=params[offset])
            circuit.ryy(i, i + 1, theta=params[offset + 1])
            circuit.rzz(i, i + 1, theta=params[offset + 2])
            offset += 3


def circuit_energy(params, mps_input, config, mpo):
    circuit = tc.Circuit(config["n_qubits"], mps_inputs=mps_input)
    apply_variational_layers(circuit, params, config)
    return mpo_expectation(circuit, mpo)


def run_solution(config):
    mps_input = tc.quantum.quimb2qop(config["dmrg_state"])
    params = initial_parameters(config)
    mpo = tfim_mpo(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    energy_fn = lambda p, m: circuit_energy(p, m, config, mpo)

    def train_step(p, state, m):
        energy, grads = K.value_and_grad(energy_fn)(p, m)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, energy

    train_step = K.jit(train_step, static_argnums=(2,))

    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, energy = train_step(params, opt_state, mps_input)
        energy_history.append(energy)

    return {
        "energy_history": K.numpy(K.stack(energy_history)),
    }
