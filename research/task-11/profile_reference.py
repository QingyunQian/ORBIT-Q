#!/usr/bin/env python3
"""Stage-split profiling for the Task 11 reference and campaign candidate.

Splits the jitted train step into trace / compile / steady-state execution
and measures the forward state-construction and energy components, using
ahead-of-time lowering (``jit(...).lower()`` / ``.compile()``).

Run from the repository root inside the pinned environment:

    PYTHONPATH=envs/tensorcircuit-py311 NUMBA_DISABLE_JIT=1 \
      python research/task-11/profile_reference.py

Writes ``research/task-11/profiles/reference-profile.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

import jax
import optax
import tensorcircuit as tc

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


def profile_module(module, objective_builder, n_meas=15):
    K = tc.backend
    params = module.initial_parameters(CONFIG)
    objective, build_state, energy_from_state = objective_builder(module)
    optimizer = optax.adam(CONFIG["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(p, state):
        value, grads = K.value_and_grad(objective)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, value

    jitted = jax.jit(train_step)
    t0 = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    t_trace = time.perf_counter() - t0
    hlo = lowered.as_text().count("\n")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    p, s = params, opt_state
    for _ in range(3):
        p, s, v = compiled(p, s)
    jax.block_until_ready(v)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        p, s, v = compiled(p, s)
    jax.block_until_ready(v)
    t_step = (time.perf_counter() - t0) / n_meas

    build_jit = jax.jit(build_state)
    state = build_jit(params)
    jax.block_until_ready(state)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        state = build_jit(params)
    jax.block_until_ready(state)
    t_build = (time.perf_counter() - t0) / n_meas

    energy_jit = jax.jit(energy_from_state)
    e = energy_jit(state)
    jax.block_until_ready(e)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        e = energy_jit(state)
    jax.block_until_ready(e)
    t_energy = (time.perf_counter() - t0) / n_meas

    return {
        "module": module.__name__,
        "jit_trace_sec": round(t_trace, 3),
        "xla_compile_sec": round(t_compile, 3),
        "stablehlo_line_count": hlo,
        "train_step_ms": round(t_step * 1e3, 2),
        "loop_500_estimate_sec": round(t_step * 500, 1),
        "forward_build_state_ms": round(t_build * 1e3, 2),
        "forward_energy_ms": round(t_energy * 1e3, 2),
    }


def reference_builder(module):
    objective = lambda p: module.energy_density(p, CONFIG)
    build = lambda p: module.build_state(p, CONFIG)
    energy = lambda st: module.energy_density_from_state(st, CONFIG)
    return objective, build, energy


def candidate_builder(module):
    energy_from_state = module.make_energy_from_state(CONFIG)
    build = lambda p: module.build_state(p, CONFIG)
    objective = lambda p: energy_from_state(build(p))
    return objective, build, energy_from_state


def main():
    reference = load_module(
        ROOT / "references" / "task-11" / "solution_11.py", "reference_11"
    )
    candidate = load_module(
        ROOT / "src" / "solutions" / "task-11" / "solution_11.py",
        "candidate_11",
    )
    report = {
        "generated_by": "research/task-11/profile_reference.py",
        "environment_lock": "envs/tensorcircuit-py311/requirements.lock",
        "reference": profile_module(reference, reference_builder),
        "candidate": profile_module(candidate, candidate_builder),
    }
    out = Path(__file__).parent / "profiles" / "reference-profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
