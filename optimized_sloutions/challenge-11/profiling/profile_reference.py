"""Stage-by-stage timing of the challenge-11 reference solution.

Imports the reference module directly and AOT-splits the jitted train step
into trace / compile / steady-state execution, plus per-component forward
costs (state construction vs energy evaluation) and the post-training
string-order block.
"""

import importlib
import sys
import time

sys.path.insert(0, "/workspace/tasks/challenge-11/solution")

import jax
import numpy as np
import optax

import tensorcircuit as tc

sol = importlib.import_module("solution_11")
K = tc.backend

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


def main():
    config = CONFIG
    params = sol.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return sol.energy_density(p, config)

    def train_step(p, state):
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, value

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
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        p, s, v = compiled(p, s)
    jax.block_until_ready(v)
    t_step = (time.perf_counter() - t0) / n

    # component forwards
    build_jit = jax.jit(lambda q: sol.build_state(q, config))
    state = build_jit(params)
    jax.block_until_ready(state)
    t0 = time.perf_counter()
    for _ in range(n):
        state = build_jit(params)
    jax.block_until_ready(state)
    t_build = (time.perf_counter() - t0) / n

    energy_jit = jax.jit(lambda st: sol.energy_density_from_state(st, config))
    e = energy_jit(state)
    jax.block_until_ready(e)
    t0 = time.perf_counter()
    for _ in range(n):
        e = energy_jit(state)
    jax.block_until_ready(e)
    t_energy = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    strings = sol.string_orders_from_state(state, config)
    t_strings = time.perf_counter() - t0

    print(f"trace: {t_trace:.2f}s  compile: {t_compile:.2f}s  hlo: {hlo}")
    print(f"train step: {t_step*1e3:.1f} ms -> 500 steps ~ {t_step*500:.1f}s")
    print(f"forward build_state: {t_build*1e3:.1f} ms")
    print(f"forward energy_from_state: {t_energy*1e3:.1f} ms")
    print(f"string orders (once, incl. trace): {t_strings:.2f}s")
    print(
        f"total estimate: {t_trace + t_compile + t_step*500 + t_strings:.1f}s"
    )


if __name__ == "__main__":
    main()
