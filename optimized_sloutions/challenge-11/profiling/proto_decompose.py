"""Decompose the fused step cost: build vs energy, forward vs backward."""

import importlib
import sys
import time

sys.path.insert(0, "/workspace/tasks/challenge-11/solution")
sys.path.insert(0, ".")

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

import proto_fused as pf

sol = importlib.import_module("solution_11")
CONFIG = pf.CONFIG


def timeit(fn, *args, n=20, label=""):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n
    print(f"{label}: {dt*1e3:.1f} ms")
    return out, dt


def main():
    config = CONFIG
    params = sol.initial_parameters(config)
    obj, build, energy = pf.make_fused_objective(config)

    state, t_build = timeit(jax.jit(build), params, label="fwd build_state (fused)")
    _, t_energy = timeit(jax.jit(energy), state, label="fwd energy (11 bond expectations + diag)")
    _, t_obj = timeit(jax.jit(obj), params, label="fwd full objective")

    # grad of build only (sum of |state|^2 as dummy scalar)
    g_build = jax.jit(jax.grad(lambda p: jnp.sum(jnp.abs(build(p)) ** 2)))
    _, t_gbuild = timeit(g_build, params, label="grad(build) dummy loss")

    # grad of energy w.r.t. state only
    g_energy = jax.jit(jax.grad(lambda st: energy(st), holomorphic=False))
    # state is complex; use value_and_grad via real cast trick: grad wrt real params only matters in full chain
    _, t_full = timeit(
        jax.jit(lambda p: jax.value_and_grad(obj)(p)[0]), params,
        label="value_and_grad(full objective)",
    )

    # energy variants: pure-jnp bond expectations (apply + vdot), for comparison
    n = config["n_sites"]
    h9 = pf.DOT9 + jnp.complex64(config["beta"]) * pf.DOT9SQ

    def energy_jnp(st):
        e = jnp.float32(0.0)
        for left in range(n - 1):
            psi = st.reshape(3**left, 9, 3 ** (n - left - 2))
            hpsi = jnp.einsum("ab,LbR->LaR", h9, psi)
            e += jnp.real(jnp.vdot(psi, hpsi))
        onsite = jnp.sum(pf.make_fused_objective(config)  # avoid rebuild: inline weight
                         if False else 0.0)
        return e / n

    _, t_ejnp = timeit(jax.jit(energy_jnp), state, label="fwd energy (jnp apply+vdot, bonds only)")


if __name__ == "__main__":
    main()
