#!/usr/bin/env python3
"""Stage-split profiling for the Task 12 reference and campaign candidate.

Splits end-to-end time into: quimb-target conversion, jit trace (including
contraction-path search), XLA compile, and the 5000-step optimizer loop, using
ahead-of-time lowering (``jit(...).lower()`` / ``.compile()``).

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-12/profile_reference.py

Writes ``research/task-12/profiles/reference-profile.json``.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references" / "task-12"))
sys.path.insert(0, str(ROOT / "src" / "solutions" / "task-12"))

CONFIG = {
    "n_qubits": 32,
    "zz_anisotropy": 1.4,
    "staggered_field": 0.2,
    "dmrg_chi": 8,
    "dmrg_sweeps": 4,
    "dmrg_tolerance": 1e-7,
    "n_layers": 2,
    "max_steps": 5000,
    "learning_rate": 0.02,
    "initial_parameter_scale": 0.02,
    "seed": 2039,
    "fidelity_threshold": 0.85,
}


def build_dmrg_state():
    import quimb.tensor as qtn

    n = CONFIG["n_qubits"]
    ham = qtn.SpinHam1D(S=0.5)
    for i in range(n - 1):
        ham[i, i + 1] += 4.0, "X", "X"
        ham[i, i + 1] += 4.0, "Y", "Y"
        ham[i, i + 1] += 4.0 * CONFIG["zz_anisotropy"], "Z", "Z"
    for i in range(n):
        ham[i] += 2.0 * CONFIG["staggered_field"] * ((-1) ** i), "Z"
    dmrg = qtn.DMRG2(
        ham.build_mpo(n),
        bond_dims=[CONFIG["dmrg_chi"]],
        cutoffs=1e-8,
    )
    dmrg.solve(tol=CONFIG["dmrg_tolerance"], max_sweeps=CONFIG["dmrg_sweeps"], verbosity=0)
    dmrg.state.normalize()
    return dmrg.state


def profile_reference(dmrg_state, n_measure=1000):
    """Replicates references/task-12/solution_12.py with stage timers."""
    import numpy as np
    import optax
    import tensorcircuit as tc

    K = tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("omeco")
    import jax

    n_qubits = CONFIG["n_qubits"]
    parameter_count = 0
    for layer in range(CONFIG["n_layers"]):
        parameter_count += 15 * len(range(layer % 2, n_qubits - 1, 2))
    rng = np.random.default_rng(CONFIG["seed"])
    params = rng.normal(
        scale=CONFIG["initial_parameter_scale"], size=(parameter_count,)
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    t0 = time.perf_counter()
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
    t_convert = time.perf_counter() - t0

    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        offset = 0
        for layer in range(CONFIG["n_layers"]):
            for i in range(layer % 2, n_qubits - 1, 2):
                circuit.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    def train_step(p, state):
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, aux

    jitted = jax.jit(train_step)
    t0 = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    t_trace = time.perf_counter() - t0
    hlo_lines = lowered.as_text().count("\n")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    p, s = params, opt_state
    for _ in range(20):
        p, s, loss, _aux = compiled(p, s)
    jax.block_until_ready(loss)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        p, s, loss, _aux = compiled(p, s)
    jax.block_until_ready(loss)
    per_step = (time.perf_counter() - t0) / n_measure

    return {
        "solution": "reference (replicated objective)",
        "quimb_conversion_sec": round(t_convert, 4),
        "jit_trace_sec": round(t_trace, 3),
        "xla_compile_sec": round(t_compile, 3),
        "stablehlo_line_count": hlo_lines,
        "steady_state_step_ms": round(per_step * 1e3, 4),
        "loop_5000_estimate_sec": round(per_step * 5000, 3),
        "total_estimate_sec": round(
            t_convert + t_trace + t_compile + per_step * 5000, 3
        ),
    }


def profile_candidate(dmrg_state):
    """End-to-end candidate timing plus its scan-level stage split."""
    import numpy as np
    import optax
    import tensorcircuit as tc

    K = tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("omeco")
    import jax
    import jax.numpy as jnp

    solution = importlib.import_module("solution_12")
    gens = jnp.asarray(solution._GENERATORS, dtype=jnp.complex64)
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
    optimizer = optax.adam(CONFIG["learning_rate"])

    n_qubits = CONFIG["n_qubits"]
    layer_bonds = [
        list(range(layer % 2, n_qubits - 1, 2))
        for layer in range(CONFIG["n_layers"])
    ]
    n_gates = sum(len(b) for b in layer_bonds)

    def objective(p):
        gates = solution._su4_batch(p.reshape(n_gates, 15), gens)
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        k = 0
        for bonds in layer_bonds:
            for i in bonds:
                circuit.any(i, i + 1, unitary=gates[k])
                k += 1
        overlap = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap) * overlap)
        return 1.0 - fidelity, (fidelity, overlap)

    def body(carry, _):
        p, s = carry
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, s = optimizer.update(grads, s, p)
        return (optax.apply_updates(p, updates), s), (loss,) + aux

    def train(p, s):
        return jax.lax.scan(body, (p, s), None, length=CONFIG["max_steps"])

    rng = np.random.default_rng(CONFIG["seed"])
    params = jnp.asarray(
        rng.normal(
            scale=CONFIG["initial_parameter_scale"], size=(15 * n_gates,)
        ).astype(np.float32)
    )
    state = optimizer.init(params)

    jitted = jax.jit(train)
    t0 = time.perf_counter()
    lowered = jitted.lower(params, state)
    t_trace = time.perf_counter() - t0
    hlo_lines = lowered.as_text().count("\n")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = compiled(params, state)
    jax.block_until_ready(out)
    t_loop = time.perf_counter() - t0

    return {
        "solution": "candidate (e01 objective in a 5000-step scan)",
        "jit_trace_sec": round(t_trace, 3),
        "xla_compile_sec": round(t_compile, 3),
        "stablehlo_line_count": hlo_lines,
        "scan_loop_5000_sec": round(t_loop, 3),
        "steady_state_step_ms": round(t_loop / CONFIG["max_steps"] * 1e3, 4),
    }


def main():
    dmrg_state = build_dmrg_state()
    report = {
        "generated_by": "research/task-12/profile_reference.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "reference": profile_reference(dmrg_state),
        "candidate": profile_candidate(dmrg_state),
    }
    out = Path(__file__).parent / "profiles" / "reference-profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
