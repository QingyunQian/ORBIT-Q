"""Diagnostics for the optimized solution:

1. Early-trajectory equivalence: per-step |loss_ref - loss_fused| for the
   first K steps (same seed/protocol; divergence should start at the c64
   noise floor and grow only through optimizer chaos).
2. Generator-norm coverage: max over steps/gates of the su(4) generator
   spectral-norm bound, to document the fixed-order expm accuracy margin.
3. Stage timing of the fused pipeline (trace / compile / loop).
"""

import sys
import time
from pathlib import Path

import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/tasks/challenge-12/solution")

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax
import jax.numpy as jnp

import solution_12_fused as fused

N = 32
STEPS_CHECK = 400


def build_dmrg_state():
    import quimb.tensor as qtn

    ham = qtn.SpinHam1D(S=0.5)
    for i in range(N - 1):
        ham[i, i + 1] += 4.0, "X", "X"
        ham[i, i + 1] += 4.0, "Y", "Y"
        ham[i, i + 1] += 4.0 * 1.4, "Z", "Z"
    for i in range(N):
        ham[i] += 2.0 * 0.2 * ((-1) ** i), "Z"
    dmrg = qtn.DMRG2(ham.build_mpo(N), bond_dims=[8], cutoffs=1e-8)
    dmrg.solve(tol=1e-7, max_sweeps=4, verbosity=0)
    dmrg.state.normalize()
    return dmrg.state


def reference_losses(dmrg_state, steps):
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()

    def objective(p):
        c = tc.Circuit(N)
        for i in range(1, N, 2):
            c.x(i)
        offset = 0
        for layer in range(2):
            for i in range(layer % 2, N - 1, 2):
                c.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15
        ov = (target_bra @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    opt = optax.adam(0.02)

    def body(carry, _):
        p, s = carry
        (loss, aux), g = K.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(g, s, p)
        return (optax.apply_updates(p, upd), s), loss

    rng = np.random.default_rng(2039)
    p = jnp.asarray(rng.normal(scale=0.02, size=(465,)).astype(np.float32))
    s = opt.init(p)
    (_, _), losses = jax.jit(
        lambda p, s: jax.lax.scan(body, (p, s), None, length=steps)
    )(p, s)
    return np.asarray(losses)


def fused_losses_and_norms(dmrg_state, steps):
    target_bra = fused._pair_fused_target_bra(dmrg_state)
    gens = jnp.asarray(fused._GENERATORS, dtype=jnp.complex64)
    neel_perm = jnp.asarray(fused._NEEL_PERM, dtype=jnp.complex64)
    eye2 = jnp.eye(2, dtype=jnp.complex64)
    opt = optax.adam(0.02)

    def objective(p):
        gates = fused._su4_batch(p.reshape(31, 15), gens)
        single = jnp.einsum("gab,bc->gac", gates[:16], neel_perm)
        double = gates[16:].reshape(-1, 2, 2, 2, 2)
        double = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", eye2, double, eye2)
        double = double.reshape(-1, 16, 16)
        c = tc.QuditCircuit(16, dim=4)
        for j in range(16):
            c.any(j, unitary=single[j])
        for j in range(15):
            c.any(j, j + 1, unitary=double[j])
        ov = (target_bra @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    def body(carry, _):
        p, s = carry
        (loss, aux), g = K.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(g, s, p)
        # spectral-norm bound per gate: sum_i |theta_i| (generators have norm 1)
        norm_bound = jnp.max(jnp.sum(jnp.abs(p.reshape(31, 15)), axis=1))
        return (optax.apply_updates(p, upd), s), (loss, norm_bound)

    rng = np.random.default_rng(2039)
    p = jnp.asarray(rng.normal(scale=0.02, size=(465,)).astype(np.float32))
    s = opt.init(p)
    (_, _), (losses, norms) = jax.jit(
        lambda p, s: jax.lax.scan(body, (p, s), None, length=steps)
    )(p, s)
    return np.asarray(losses), np.asarray(norms)


def stage_profile(dmrg_state):
    config = {
        "n_qubits": 32,
        "n_layers": 2,
        "max_steps": 5000,
        "learning_rate": 0.02,
        "initial_parameter_scale": 0.02,
        "seed": 2039,
        "dmrg_state": dmrg_state,
    }
    t0 = time.perf_counter()
    out = fused.run_solution(config)
    total = time.perf_counter() - t0
    print(f"fused run_solution end-to-end: {total:.2f}s "
          f"(final fid {out['fidelity_history'][-1]:.6f})")


def main():
    dmrg_state = build_dmrg_state()

    ref = reference_losses(dmrg_state, STEPS_CHECK)
    fus, norms = fused_losses_and_norms(dmrg_state, 5000)
    d = np.abs(ref - fus[:STEPS_CHECK])
    for k in (10, 50, 100, 200, 400):
        print(f"max |loss_ref - loss_fused| over first {k:4d} steps: "
              f"{d[:k].max():.3e}")
    print(f"max generator spectral-norm bound over 5000 steps: "
          f"{norms.max():.3f} (scaled by 2**-5 -> {norms.max()/32:.4f})")
    stage_profile(dmrg_state)


if __name__ == "__main__":
    main()
