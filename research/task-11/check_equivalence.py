#!/usr/bin/env python3
"""Trajectory-equivalence audit: Task 11 candidate vs immutable reference.

Runs both training loops from the identical seeded initialization and
reports per-step energy deviations over the first K steps, the maximum
entangler generator-norm bound across all 500 candidate steps, and the
candidate's final metrics.

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-11/check_equivalence.py

Writes ``research/task-11/profiles/equivalence-check.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "n_sites": 12,
    "n_layers": 5,
    "beta": 0.20,
    "single_ion_anisotropy": 0.15,
    "max_steps": 500,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.05,
    "seed": 2041,
}
CHECK_STEPS = 100


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scan_run(objective, params, steps, extra=None):
    opt = optax.adam(CONFIG["learning_rate"])

    def body(carry, _):
        p, s = carry
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        ys = (v,) if extra is None else (v, extra(p))
        return (optax.apply_updates(p, upd), s), ys

    @jax.jit
    def run(p):
        (p, _), ys = jax.lax.scan(body, (p, opt.init(p)), None, length=steps)
        return p, ys

    return run(params)


def main():
    reference = load_module(
        ROOT / "references" / "task-11" / "solution_11.py", "reference_11"
    )
    candidate = load_module(
        ROOT / "src" / "solutions" / "task-11" / "solution_11.py",
        "candidate_11",
    )

    params = reference.initial_parameters(CONFIG)
    energy_cand = candidate.make_energy_from_state(CONFIG)

    obj_ref = lambda p: reference.energy_density(p, CONFIG)
    obj_cand = lambda p: energy_cand(candidate.build_state(p, CONFIG))

    _, (hist_ref,) = scan_run(obj_ref, params, CHECK_STEPS)

    def gen_norm_bound(p):
        t = jnp.concatenate(
            [p["even_theta"].reshape(-1), p["odd_theta"].reshape(-1)]
        )
        f = jnp.concatenate(
            [p["even_phi"].reshape(-1), p["odd_phi"].reshape(-1)]
        )
        return (
            jnp.max(jnp.abs(t) * 2.0 + jnp.abs(f - t) * 1.0)
            + CONFIG["beta"] * 4.0
        )

    p_final, (hist_cand, norms) = scan_run(
        obj_cand, params, CONFIG["max_steps"], extra=gen_norm_bound
    )

    delta = np.abs(np.asarray(hist_ref) - np.asarray(hist_cand)[:CHECK_STEPS])

    @jax.jit
    def finalize(p):
        st = candidate.build_state(p, CONFIG)
        return energy_cand(st), candidate.string_orders_from_state(st, CONFIG)

    e_fin, strings = finalize(p_final)
    report = {
        "generated_by": "research/task-11/check_equivalence.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "max_abs_energy_delta_first_k_steps": {
            str(k): float(delta[:k].max()) for k in (5, 20, 50, 100)
        },
        "max_entangler_generator_norm_bound_500_steps": float(
            np.asarray(norms).max()
        ),
        "pade_scaled_norm_at_max": float(np.asarray(norms).max() / 32.0),
        "candidate_final_energy_density": float(e_fin),
        "candidate_final_string_orders": [
            float(x) for x in np.asarray(strings)
        ],
        "note": (
            "Exact gate fusion changes float rounding, so deltas start at "
            "the complex64 noise floor (not exactly zero) and grow only "
            "through optimizer dynamics; the layer unitaries are "
            "algebraically identical."
        ),
    }
    out = Path(__file__).parent / "profiles" / "equivalence-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
