#!/usr/bin/env python3
"""Micro-benchmark and accuracy audit of batched 4x4 SU(4) exponentials.

Compares the generic norm-adaptive ``jax.scipy.linalg.expm`` against the
candidate's fixed-order scaling-and-squaring diagonal Pade(3,3) inside a
value_and_grad + Adam scan step shaped like the Task 12 gate build, and
measures both approximants against a float64 SciPy reference over the
generator-norm range relevant to this protocol.

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-12/profile_expm.py

Writes ``research/task-12/profiles/expm-microbench.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "solutions" / "task-12"))

import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import solution_12  # campaign candidate; provides _GENERATORS and _su4_batch
from jax.scipy.linalg import expm as jexpm

GENS = solution_12._GENERATORS
GENS_J = jnp.asarray(GENS, dtype=jnp.complex64)


def accuracy_rows():
    import scipy.linalg

    rows = []
    rng = np.random.default_rng(0)
    for scale in (0.05, 0.5, 1.0, 2.0):
        theta = rng.normal(scale=scale, size=(31, 15))
        h = np.einsum("gi,iab->gab", theta, GENS)
        exact = np.stack(
            [scipy.linalg.expm(-1j * m.astype(np.complex128)) for m in h]
        )
        got_jexpm = np.asarray(jexpm(jnp.asarray(-1j * h, dtype=jnp.complex64)))
        got_pade = np.asarray(
            solution_12._su4_batch(jnp.asarray(theta, dtype=jnp.float32), GENS_J)
        )
        norm_bound = float(np.abs(theta).sum(axis=1).max())
        rows.append(
            {
                "theta_scale": scale,
                "max_generator_norm_bound": round(norm_bound, 3),
                "jexpm_max_abs_err": float(np.abs(got_jexpm - exact).max()),
                "pade33_s5_max_abs_err": float(np.abs(got_pade - exact).max()),
                "pade33_s5_max_unitarity_defect": float(
                    np.abs(
                        np.einsum("gab,gcb->gac", got_pade, got_pade.conj())
                        - np.eye(4)
                    ).max()
                ),
            }
        )
    return rows


def bench(fn, n=2000):
    opt = optax.adam(0.02)

    def loss_fn(p):
        u = fn(p.reshape(31, 15))
        return jnp.sum(jnp.abs(u - jnp.eye(4, dtype=u.dtype)) ** 2)

    def body(carry, _):
        p, s = carry
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s, p)
        return (optax.apply_updates(p, upd), s), loss

    @jax.jit
    def train(p, s):
        return jax.lax.scan(body, (p, s), None, length=n)

    p = jnp.asarray(
        np.random.default_rng(1).normal(scale=0.02, size=(465,)).astype(np.float32)
    )
    s = opt.init(p)
    out = train(p, s)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = train(p, s)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n


def main():
    def jexpm_gates(thetas):
        h = jnp.einsum("gi,iab->gab", thetas.astype(jnp.complex64), GENS_J)
        return jexpm(-1j * h)

    report = {
        "generated_by": "research/task-12/profile_expm.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "accuracy_vs_float64_scipy": accuracy_rows(),
        "scan_step_with_grad_and_adam_ms": {
            "jax_scipy_expm_pade13_adaptive": round(bench(jexpm_gates) * 1e3, 4),
            "fixed_pade33_s5_candidate": round(
                bench(lambda t: solution_12._su4_batch(t, GENS_J)) * 1e3, 4
            ),
        },
        "note": (
            "Both rows time one scan step containing only the batched 31-gate "
            "construction, its gradient, and an Adam update; the candidate "
            "kernel is the exact function used by solution_12._su4_batch."
        ),
    }
    out = Path(__file__).parent / "profiles" / "expm-microbench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
