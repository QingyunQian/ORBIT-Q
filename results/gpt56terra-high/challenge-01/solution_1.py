"""TensorCircuit MPS variational refinement for the open TFIM chain."""

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensornetwork import Node


def _tc_tensors(quimb_mps):
    """Convert quimb's (left, right, physical) arrays to TC MPS arrays."""
    arrays = [np.asarray(a) for a in quimb_mps.arrays]
    ts = [arrays[0].T[None, :, :]]
    ts += [a.transpose(0, 2, 1) for a in arrays[1:-1]]
    ts += [arrays[-1][:, :, None]]
    return ts


def _ising_mpo(n, field):
    """A TensorCircuit QuOperator MPO for -ZZ - field*X."""
    ident = jnp.eye(2, dtype=jnp.complex64)
    x, z = tc.gates.x().tensor, tc.gates.z().tensor
    w = jnp.zeros((3, 2, 2, 3), dtype=jnp.complex64)
    w = w.at[0, :, :, 0].set(ident).at[0, :, :, 1].set(z)
    w = w.at[0, :, :, 2].set(-field * x).at[1, :, :, 2].set(-z)
    w = w.at[2, :, :, 2].set(ident)
    nodes = [Node(w[0])] + [Node(w) for _ in range(n - 2)] + [Node(w[:, :, :, 2])]
    nodes[0][2] ^ nodes[1][0]
    for left, right in zip(nodes[1:-1], nodes[2:]):
        left[3] ^ right[0]
    out = [nodes[0][0]] + [a[1] for a in nodes[1:-1]] + [nodes[-1][1]]
    inn = [nodes[0][1]] + [a[2] for a in nodes[1:-1]] + [nodes[-1][2]]
    return tc.quantum.QuOperator(out, inn)


def _mps_vector(tensors):
    """Expose an MPSCircuit output as TensorCircuit's MPS QuVector."""
    nodes = [Node(tensors[0][0])] + [Node(a) for a in tensors[1:-1]]
    nodes += [Node(tensors[-1][:, :, 0])]
    nodes[0][1] ^ nodes[1][0]
    for left, right in zip(nodes[1:-1], nodes[2:]):
        left[2] ^ right[0]
    edges = [nodes[0][0]] + [a[1] for a in nodes[1:-1]] + [nodes[-1][1]]
    return tc.quantum.QuVector(edges)


def run_solution(config):
    tc.set_backend("jax")
    n, nlayers = int(config["n_qubits"]), int(config["n_layers"])
    field, steps, lr = float(config["field"]), int(config["max_steps"]), float(config["learning_rate"])

    raw = _tc_tensors(config["dmrg_state"])
    # Canonicalizing once is outside the differentiated circuit and preserves the input state.
    initial_circuit = tc.MPSCircuit(n, tensors=raw)
    initial = tuple(initial_circuit.get_tensors())
    hamiltonian = _ising_mpo(n, field)
    xx, yy, zz = tc.gates._xx_matrix, tc.gates._yy_matrix, tc.gates._zz_matrix

    def energy(parameters):
        # Keeping a modest MPS bond cap makes the shallow correction scalable
        # while retaining twice the supplied DMRG bond dimension.
        circuit = tc.MPSCircuit(
            n, tensors=list(initial), center_position=0,
            split={"max_singular_values": 16},
        )
        for layer in range(nlayers):
            for site in range(n):
                circuit.rz(site, theta=parameters[layer, site, 0])
                circuit.ry(site, theta=parameters[layer, site, 1])
                circuit.rz(site, theta=parameters[layer, site, 2])
            for site in range(layer % 2, n - 1, 2):
                # XX, YY and ZZ commute, so these are exactly the specified interaction.
                circuit.exp1(site, site + 1, unitary=xx, theta=parameters[layer, site, 3])
                circuit.exp1(site, site + 1, unitary=yy, theta=parameters[layer, site, 4])
                circuit.exp1(site, site + 1, unitary=zz, theta=parameters[layer, site, 5])
        psi = _mps_vector(circuit.get_tensors())
        return jnp.real((psi.adjoint() @ hamiltonian @ psi).eval_matrix()[0, 0])

    value_and_grad = jax.value_and_grad(energy)

    def step(carry, _):
        parameters, first, second, count = carry
        value, gradient = value_and_grad(parameters)
        count = count + 1
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**count)
        second_hat = second / (1.0 - 0.999**count)
        parameters = parameters - lr * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
        return (parameters, first, second, count), value

    parameters = jnp.zeros((nlayers, n, 6), dtype=jnp.float32)
    carry = (parameters, jnp.zeros_like(parameters), jnp.zeros_like(parameters), jnp.array(0))
    _, history = jax.jit(lambda c: jax.lax.scan(step, c, None, length=steps))(carry)
    return {"energy_history": np.asarray(history)}
