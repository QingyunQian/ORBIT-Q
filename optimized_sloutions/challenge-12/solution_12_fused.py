"""
Challenge Suite Problem 12: variational circuit to MPS overlap optimization.

Performance-optimized variant of the reference solution. The protocol is
unchanged: identical su(4)-generator parameterization of every SU4 gate,
identical seed/initialization, identical Adam schedule on the identical
direct tensor-network overlap loss between the DMRG-MPS bra and the circuit
ket. Only the computation is restructured:

1. Batched gate construction: all 31 SU4 exponentials are built in one shot
   from a (31, 15) parameter tensor. The matrix exponential uses a
   fixed-order scaling-and-squaring diagonal Pade(3,3) approximant, which is
   exactly unitary for anti-Hermitian arguments and accurate to below the
   complex64 noise floor over this protocol's parameter range, and whose
   static graph compiles and differentiates far faster than the generic
   norm-adaptive expm.
2. Qubit-pair fusion: the depth-2 brickwork on 32 qubits is contracted as a
   depth-1 chain on 16 four-level sites (layer-1 gates become single-site
   unitaries with the Neel preparation folded in as a constant basis
   permutation; layer-2 gates straddle fused sites). The DMRG target is
   pair-fused once outside the training loop. This halves the contraction
   network without changing any contraction result.
3. The 5000 Adam updates run inside a single jax.lax.scan, so the step is
   compiled exactly once and the loop adds no per-step dispatch overhead.
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
# constant single-site basis permutation |b> -> |b xor 1> on the second qubit
# of a fused pair; folds the Neel |01> preparation into the layer-1 gates
_NEEL_PERM = np.eye(4, dtype=complex)[:, [1, 0, 3, 2]]


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


def _pair_fused_target_bra(dmrg_state):
    """Pair-fuse the quimb DMRG MPS into a 16-site QuVector bra (chi <= 8)."""
    mps = dmrg_state.copy()
    mps.permute_arrays("lpr")
    arrays = [np.asarray(a, dtype=np.complex64) for a in mps.arrays]
    arrays[0] = arrays[0].reshape(1, *arrays[0].shape)
    arrays[-1] = arrays[-1].reshape(*arrays[-1].shape, 1)
    fused = []
    for j in range(len(arrays) // 2):
        pair = np.einsum("lpm,mqr->lpqr", arrays[2 * j], arrays[2 * j + 1])
        fused.append(pair.reshape(pair.shape[0], 4, pair.shape[3]))
    fused[0] = fused[0][0]
    fused[-1] = fused[-1][..., 0]
    nodes = [tc.quantum.Node(t) for t in fused]
    for left, right in zip(nodes[:-1], nodes[1:]):
        left[-1] ^ right[0]
    out_edges = [nodes[0][0]] + [node[1] for node in nodes[1:]]
    return tc.quantum.QuVector(out_edges).adjoint()


def run_solution(config):
    n_qubits = config["n_qubits"]
    n_sites = n_qubits // 2
    layer_sizes = [
        len(range(layer % 2, n_qubits - 1, 2))
        for layer in range(config["n_layers"])
    ]
    n_gates = sum(layer_sizes)

    rng = np.random.default_rng(config["seed"])
    params = rng.normal(
        scale=config["initial_parameter_scale"],
        size=(15 * n_gates,),
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    target_bra = _pair_fused_target_bra(config["dmrg_state"])
    gens = jnp.asarray(_GENERATORS, dtype=jnp.complex64)
    neel_perm = jnp.asarray(_NEEL_PERM, dtype=jnp.complex64)
    eye2 = jnp.eye(2, dtype=jnp.complex64)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        gates = _su4_batch(p.reshape(n_gates, 15), gens)
        # layer 1 on fused pairs, with the Neel preparation folded in
        single = jnp.einsum("gab,bc->gac", gates[: layer_sizes[0]], neel_perm)
        # layer 2 straddles fused sites j, j+1: I2 (x) SU4 (x) I2, regrouped
        double = gates[layer_sizes[0] :].reshape(-1, 2, 2, 2, 2)
        double = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", eye2, double, eye2)
        double = double.reshape(-1, 16, 16)

        circuit = tc.QuditCircuit(n_sites, dim=4)
        for j in range(n_sites):
            circuit.any(j, unitary=single[j])
        for j in range(n_sites - 1):
            circuit.any(j, j + 1, unitary=double[j])

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
