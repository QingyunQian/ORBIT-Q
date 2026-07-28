"""Prototype: fused-gate challenge-11 objective, cross-validated vs reference.

Checks: objective equality at random params, per-step trajectory deltas,
and stage timings (trace/compile/step) of the fused train step.
"""

import importlib
import sys
import time

sys.path.insert(0, "/workspace/tasks/challenge-11/solution")

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

sol = importlib.import_module("solution_11")

DIM = 3
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

SQRT2 = np.sqrt(2.0, dtype=np.float32)
DOT9 = jnp.asarray(np.asarray(sol.DOT_BOND), dtype=jnp.complex64)
DOT9SQ = jnp.asarray(np.asarray(sol.DOT_BOND_SQUARED), dtype=jnp.complex64)
ZZ9 = jnp.asarray(np.asarray(sol.ZZ_BOND), dtype=jnp.complex64)
SZ2_DIAG = np.array([1.0, 0.0, 1.0], dtype=np.float32)


def expm_pade33_fixed(a, s=5):
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    a = a / (2**s)
    a2 = a @ a
    odd = a @ (a2 + 60.0 * eye)
    even = 12.0 * a2 + 120.0 * eye
    r = jnp.linalg.solve(even - odd, even + odd)
    for _ in range(s):
        r = r @ r
    return r


def rz_batch(theta):
    z = jnp.zeros_like(theta, dtype=jnp.complex64)
    e_m = jnp.exp(-1j * theta.astype(jnp.complex64))
    e_p = jnp.exp(1j * theta.astype(jnp.complex64))
    one = jnp.ones_like(e_m)
    rows = [
        jnp.stack([e_m, z, z], axis=-1),
        jnp.stack([z, one, z], axis=-1),
        jnp.stack([z, z, e_p], axis=-1),
    ]
    return jnp.stack(rows, axis=-2)


def ry_batch(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    rows = [
        jnp.stack([(1.0 + c) / 2.0, -s / SQRT2, (1.0 - c) / 2.0], axis=-1),
        jnp.stack([s / SQRT2, c, -s / SQRT2], axis=-1),
        jnp.stack([(1.0 - c) / 2.0, s / SQRT2, (1.0 + c) / 2.0], axis=-1),
    ]
    return jnp.stack(rows, axis=-2).astype(jnp.complex64)


def entangler_batch(theta, phi, beta):
    gen = (
        theta[:, None, None].astype(jnp.complex64) * DOT9
        + (phi - theta)[:, None, None].astype(jnp.complex64) * ZZ9
        + jnp.complex64(beta) * DOT9SQ
    )
    return expm_pade33_fixed(-1j * gen)


def make_fused_objective(config):
    n = config["n_sites"]
    beta = config["beta"]
    aniso = config["single_ion_anisotropy"]

    # diagonal onsite weight vector: sum_i diag(SZ2)_i over the product basis
    digits = np.zeros((DIM**n, n), dtype=np.int8)
    values = np.arange(DIM**n, dtype=np.int64)
    for site in range(n - 1, -1, -1):
        digits[:, site] = values % DIM
        values //= DIM
    w_onsite = jnp.asarray(
        aniso * SZ2_DIAG[digits].sum(axis=1), dtype=jnp.float32
    )

    bond_gate = tc.gates.Gate(
        K.reshape(DOT9 + jnp.complex64(beta) * DOT9SQ, (DIM,) * 4)
    )

    def apply_layer(state, lp):
        u = jnp.einsum(
            "sab,sbc,scd->sad",
            rz_batch(lp["single_rz2"]),
            ry_batch(lp["single_ry"]),
            rz_batch(lp["single_rz1"]),
        )
        pair = jnp.einsum("kac,kbd->kabcd", u[0::2], u[1::2]).reshape(-1, 9, 9)
        even = entangler_batch(lp["even_theta"], lp["even_phi"], beta) @ pair
        odd = entangler_batch(lp["odd_theta"], lp["odd_phi"], beta)
        c = tc.QuditCircuit(n, dim=DIM, inputs=state)
        for k in range(even.shape[0]):
            c.unitary(
                2 * k,
                2 * k + 1,
                unitary=tc.gates.Gate(even[k].reshape((DIM,) * 4)),
            )
        for k in range(odd.shape[0]):
            c.unitary(
                2 * k + 1,
                2 * k + 2,
                unitary=tc.gates.Gate(odd[k].reshape((DIM,) * 4)),
            )
        return c.state()

    def build_state(params):
        return K.scan(apply_layer, params, sol.initial_state(config))

    def energy_from_state(state):
        c = tc.QuditCircuit(n, dim=DIM, inputs=state)
        energy = K.cast(0.0, "complex64")
        for left in range(n - 1):
            energy += c.expectation((bond_gate, [left, left + 1]))
        onsite = jnp.sum(w_onsite * jnp.abs(state) ** 2)
        return (K.real(energy) + onsite) / n

    def objective(params):
        return energy_from_state(build_state(params))

    return objective, build_state, energy_from_state


def main():
    config = CONFIG
    params = sol.initial_parameters(config)

    obj_fused, build_fused, energy_fused = make_fused_objective(config)
    obj_ref = lambda p: sol.energy_density(p, config)

    # --- objective equality on random parameter sets
    for trial, scale in enumerate((0.05, 0.4, 1.0)):
        rng = np.random.default_rng(100 + trial)
        p = jax.tree.map(
            lambda x: jnp.asarray(
                rng.normal(scale=scale, size=x.shape).astype(np.float32)
            ),
            params,
        )
        e_ref = jax.jit(obj_ref)(p)
        e_fus = jax.jit(obj_fused)(p)
        print(
            f"scale {scale}: E_ref {float(e_ref):+.8f}  "
            f"|dE| {abs(float(e_ref) - float(e_fus)):.2e}"
        )

    # --- trajectory deltas over first K steps
    def make_scan(objective, steps):
        opt = optax.adam(config["learning_rate"])

        def body(carry, _):
            p, s = carry
            v, g = jax.value_and_grad(objective)(p)
            upd, s = opt.update(g, s, p)
            return (optax.apply_updates(p, upd), s), v

        def run(p):
            return jax.lax.scan(body, (p, opt.init(p)), None, length=steps)[1]

        return jax.jit(run)

    steps = 60
    hist_ref = np.asarray(make_scan(obj_ref, steps)(params))
    hist_fus = np.asarray(make_scan(obj_fused, steps)(params))
    d = np.abs(hist_ref - hist_fus)
    for k in (5, 20, 60):
        print(f"max |dE| first {k:3d} steps: {d[:k].max():.3e}")

    # --- stage timing of the fused step
    opt = optax.adam(config["learning_rate"])

    def train_step(p, s):
        v, g = jax.value_and_grad(obj_fused)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, v

    jitted = jax.jit(train_step)
    t0 = time.perf_counter()
    low = jitted.lower(params, opt.init(params))
    t_trace = time.perf_counter() - t0
    hlo = low.as_text().count("\n")
    t0 = time.perf_counter()
    comp = low.compile()
    t_compile = time.perf_counter() - t0
    p, s = params, opt.init(params)
    for _ in range(3):
        p, s, v = comp(p, s)
    jax.block_until_ready(v)
    n_meas = 30
    t0 = time.perf_counter()
    for _ in range(n_meas):
        p, s, v = comp(p, s)
    jax.block_until_ready(v)
    t_step = (time.perf_counter() - t0) / n_meas
    print(
        f"fused: trace {t_trace:.2f}s compile {t_compile:.2f}s hlo {hlo} "
        f"step {t_step*1e3:.1f}ms -> est total {t_trace+t_compile+t_step*500:.1f}s"
    )


if __name__ == "__main__":
    main()
