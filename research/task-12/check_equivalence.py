#!/usr/bin/env python3
"""Trajectory-equivalence audit: candidate vs immutable Task 12 reference.

Runs both training loops from the identical seeded initialization in one
process and reports the per-step loss deviation over the first K steps, plus
the maximum su(4) generator-norm bound reached across all 5000 candidate
steps (the accuracy-margin input for the fixed-order Pade exponential).

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-12/check_equivalence.py

Writes ``research/task-12/profiles/equivalence-check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "solutions" / "task-12"))

import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax
import jax.numpy as jnp
import solution_12  # campaign candidate

N = 32
CHECK_STEPS = 400


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


def scan_losses(objective, steps, extra=None):
    opt = optax.adam(0.02)

    def body(carry, _):
        p, s = carry
        (loss, _aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(grads, s, p)
        ys = (loss,) if extra is None else (loss, extra(p))
        return (optax.apply_updates(p, upd), s), ys

    rng = np.random.default_rng(2039)
    p = jnp.asarray(rng.normal(scale=0.02, size=(465,)).astype(np.float32))
    s = opt.init(p)
    (_, _), ys = jax.jit(
        lambda p, s: jax.lax.scan(body, (p, s), None, length=steps)
    )(p, s)
    return tuple(np.asarray(y) for y in ys)


def main():
    dmrg_state = build_dmrg_state()
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()

    def reference_objective(p):
        circuit = tc.Circuit(N)
        for i in range(1, N, 2):
            circuit.x(i)
        offset = 0
        for layer in range(2):
            for i in range(layer % 2, N - 1, 2):
                circuit.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    gens = jnp.asarray(solution_12._GENERATORS, dtype=jnp.complex64)

    def candidate_objective(p):
        gates = solution_12._su4_batch(p.reshape(31, 15), gens)
        circuit = tc.Circuit(N)
        for i in range(1, N, 2):
            circuit.x(i)
        k = 0
        for layer in range(2):
            for i in range(layer % 2, N - 1, 2):
                circuit.any(i, i + 1, unitary=gates[k])
                k += 1
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    (ref_losses,) = scan_losses(reference_objective, CHECK_STEPS)
    cand_losses, norm_bounds = scan_losses(
        candidate_objective,
        5000,
        extra=lambda p: jnp.max(jnp.sum(jnp.abs(p.reshape(31, 15)), axis=1)),
    )
    delta = np.abs(ref_losses - cand_losses[:CHECK_STEPS])
    report = {
        "generated_by": "research/task-12/check_equivalence.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "max_abs_loss_delta_first_k_steps": {
            "10": float(delta[:10].max()),
            "50": float(delta[:50].max()),
            "100": float(delta[:100].max()),
            "400": float(delta[:400].max()),
        },
        "candidate_final_loss_5000": float(cand_losses[-1]),
        "max_generator_norm_bound_over_5000_steps": float(norm_bounds.max()),
        "pade_scaled_norm_at_max": float(norm_bounds.max() / 32.0),
        "note": (
            "Loss histories are compared step-by-step from the identical "
            "seeded initialization; a zero delta over the first steps means "
            "the candidate reproduces the reference trajectory bit-for-bit "
            "in float32 before ordinary round-off noise is amplified by "
            "optimizer dynamics."
        ),
    }
    out = Path(__file__).parent / "profiles" / "equivalence-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
