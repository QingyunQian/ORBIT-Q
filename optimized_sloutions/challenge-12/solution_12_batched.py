"""
Challenge Suite Problem 12: variational circuit to MPS overlap optimization.

Minimal-diff performance-optimized variant of the reference solution. The
protocol, circuit layout, su(4) parameterization, seed, and Adam schedule are
identical to the reference; only two computational changes are made:

1. All 31 SU4 gate matrices are built in one batched shot per step: a single
   einsum against the stacked su(4) generators (same generator order as
   tc.gates.su4_gate) followed by one batched matrix exponential using a
   fixed-order scaling-and-squaring diagonal Pade(3,3) approximant. The
   approximant is exactly unitary for anti-Hermitian arguments and accurate
   below the complex64 noise floor over this protocol's parameter range,
   while its static graph traces, compiles, and differentiates far faster
   than 31 separate norm-adaptive expm calls.
2. The 5000 Adam updates run inside a single jax.lax.scan, so the training
   step is compiled exactly once and the loop adds no per-step dispatch.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax
import jax.numpy as jnp

_PAULI = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
# 15 su(4) generators in the exact order used by tc.gates.su4_gate
_GENERATORS = np.stack(
    [
        np.einsum("ab,cd->acbd", _PAULI[a], _PAULI[b]).reshape(4, 4)
        for a in "ixyz"
        for b in "ixyz"
        if (a, b) != ("i", "i")
    ]
)


def _su4_batch(thetas, gens):
    """(gates, 15) angles -> (gates, 4, 4) SU4 matrices exp(-i sum theta G).

    Fixed 2**5 scaling-and-squaring with a diagonal Pade(3,3) core: exactly
    unitary for anti-Hermitian input, static graph, cheap to differentiate.
    """
    a = jnp.einsum("gi,iab->gab", thetas.astype(gens.dtype), gens) / 32j
    eye = jnp.eye(4, dtype=a.dtype)
    a2 = a @ a
    odd = a @ (a2 + 60.0 * eye)
    even = 12.0 * a2 + 120.0 * eye
    r = jnp.linalg.solve(even - odd, even + odd)
    for _ in range(5):
        r = r @ r
    return r


def run_solution(config):
    n_qubits = config["n_qubits"]
    layer_bonds = [
        list(range(layer % 2, n_qubits - 1, 2))
        for layer in range(config["n_layers"])
    ]
    n_gates = sum(len(bonds) for bonds in layer_bonds)

    rng = np.random.default_rng(config["seed"])
    params = rng.normal(
        scale=config["initial_parameter_scale"],
        size=(15 * n_gates,),
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    target_mps = tc.quantum.quimb2qop(config["dmrg_state"])
    target_bra = target_mps.adjoint()
    gens = jnp.asarray(_GENERATORS, dtype=jnp.complex64)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        gates = _su4_batch(p.reshape(n_gates, 15), gens)
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        k = 0
        for bonds in layer_bonds:
            for i in bonds:
                circuit.any(i, i + 1, unitary=gates[k])
                k += 1
        overlap_value = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap_value) * overlap_value)
        return 1.0 - fidelity, (fidelity, overlap_value)

    def train_step(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        return (optax.apply_updates(p, updates), state), (loss,) + aux

    @jax.jit
    def train(p, state):
        return jax.lax.scan(
            train_step, (p, state), None, length=config["max_steps"]
        )

    (params, _), (losses, fidelities, overlaps) = train(params, opt_state)

    return {
        "loss_history": K.numpy(losses),
        "fidelity_history": K.numpy(fidelities),
        "final_parameters": K.numpy(params),
        "final_overlap_phase": np.asarray(np.angle(K.numpy(overlaps)[-1])),
    }
