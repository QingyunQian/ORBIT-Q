"""
Task Suite Problem 12 (optimized v2): identical protocol to the reference,
three engineering changes:
1. All 31 SU(4) gate matrices are built in one batched expm (same generator
   basis and parameter order as tc.gates.su4_gate, so the optimization
   trajectory is protocol-identical), shrinking the traced graph.
2. The 5000 Adam updates run inside one jax.lax.scan (no per-step dispatch).
3. Contraction paths come from a cheap deterministic-budget cotengra greedy
   search (sub-second) instead of omeco's costlier stochastic search.
"""
import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
import cotengra as ctg
tc.set_contractor("custom",
                  optimizer=ctg.HyperOptimizer(methods=["greedy"], max_repeats=32,
                                               max_time=1.0, parallel=False, progbar=False),
                  preprocessing=True)

import jax
import jax.numpy as jnp
from tensorcircuit import gates as g

_PAULIS15 = np.stack([
    np.asarray(m, dtype=np.complex64) for m in (
        g._ix_matrix, g._iy_matrix, g._iz_matrix,
        g._xi_matrix, g._xx_matrix, g._xy_matrix, g._xz_matrix,
        g._yi_matrix, g._yx_matrix, g._yy_matrix, g._yz_matrix,
        g._zi_matrix, g._zx_matrix, g._zy_matrix, g._zz_matrix,
    )
])


def run_solution(config):
    n_qubits = config["n_qubits"]
    bonds = []
    for layer in range(config["n_layers"]):
        for i in range(layer % 2, n_qubits - 1, 2):
            bonds.append((i, i + 1))
    n_gates = len(bonds)

    rng = np.random.default_rng(config["seed"])
    params = rng.normal(scale=config["initial_parameter_scale"],
                        size=(n_gates * 15,)).astype(np.float32)
    params = K.convert_to_tensor(params)

    target_bra = tc.quantum.quimb2qop(config["dmrg_state"]).adjoint()
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    paulis = jnp.asarray(_PAULIS15)

    def objective(p):
        # batched SU(4): same generator basis/order as tc.gates.su4_gate
        gens = jnp.einsum("gk,kab->gab", p.reshape(n_gates, 15).astype(jnp.complex64), paulis)
        mats = jax.scipy.linalg.expm(-1j * gens)
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        for k, (a, b) in enumerate(bonds):
            circuit.any(a, b, unitary=mats[k])
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    def scan_step(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return (p, state), (loss, aux[0], aux[1])

    @jax.jit
    def train(p, state):
        (p, _), ys = jax.lax.scan(scan_step, (p, state), None, length=config["max_steps"])
        return p, ys

    params, (loss_h, fid_h, ov_h) = train(params, opt_state)
    return {
        "loss_history": np.asarray(jax.device_get(loss_h)),
        "fidelity_history": np.asarray(jax.device_get(fid_h)),
        "final_parameters": np.asarray(jax.device_get(params)),
        "final_overlap_phase": np.asarray(np.angle(np.asarray(jax.device_get(ov_h))[-1])),
    }
