#!/usr/bin/env python3
"""Dense-state gate-application microbenchmark for Task 11.

Measures the cost of one two-qudit (9x9) gate contraction against the dense
3^12 state as a function of bond position, through the framework circuit
path and through a bare reshape-matmul floor estimate. The floor row is a
boundary datapoint only: hand-rolled state evolution is not a shippable
approach under the framework-fidelity rules.

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-11/profile_gate_application.py

Writes ``research/task-11/profiles/gate-application-microbench.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

DIM = 3
N = 12


def timeit(fn, *args, n=30):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n


def main():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(DIM**N,)) + 1j * rng.normal(size=(DIM**N,))
    state = jnp.asarray((state / np.linalg.norm(state)).astype(np.complex64))
    gate = jnp.asarray(
        (rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))).astype(
            np.complex64
        )
    )

    rows = []
    for left in (0, 3, 5, 8, 10):

        def apply_tc(st, left=left):
            c = tc.QuditCircuit(N, dim=DIM, inputs=st)
            c.unitary(
                left,
                left + 1,
                unitary=tc.gates.Gate(gate.reshape(3, 3, 3, 3)),
            )
            return c.state()

        def apply_floor(st, left=left):
            psi = st.reshape(3**left, 9, 3 ** (N - left - 2))
            return jnp.einsum("ab,LbR->LaR", gate, psi).reshape(-1)

        rows.append(
            {
                "bond": [left, left + 1],
                "framework_circuit_ms": round(
                    timeit(jax.jit(apply_tc), state) * 1e3, 3
                ),
                "reshape_matmul_floor_ms": round(
                    timeit(jax.jit(apply_floor), state) * 1e3, 3
                ),
            }
        )

    report = {
        "generated_by": "research/task-11/profile_gate_application.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "state_bytes": int(DIM**N * 8),
        "per_position": rows,
        "note": (
            "Both columns are one 9x9 two-qudit contraction against the "
            "dense 3^12 complex64 state. Position dependence reflects the "
            "transpose+gemm lowering of strided axis contractions; the "
            "floor column bounds what layout tuning could recover and is "
            "not a shippable evolution path under the framework rules."
        ),
    }
    out = Path(__file__).parent / "profiles" / "gate-application-microbench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
