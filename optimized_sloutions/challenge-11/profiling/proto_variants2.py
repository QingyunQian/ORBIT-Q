"""Second-round variants: layer scan vs unroll; single vs split circuits."""

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
DIM = 3


def make_objective(config, layer_mode="scan", circuit_mode="joint"):
    n = config["n_sites"]
    beta = config["beta"]
    aniso = config["single_ion_anisotropy"]

    digits = np.zeros((DIM**n, n), dtype=np.int8)
    values = np.arange(DIM**n, dtype=np.int64)
    for site in range(n - 1, -1, -1):
        digits[:, site] = values % DIM
        values //= DIM
    w_onsite = jnp.asarray(
        aniso * pf.SZ2_DIAG[digits].sum(axis=1), dtype=jnp.float32
    )
    bond_gate = tc.gates.Gate(
        K.reshape(pf.DOT9 + jnp.complex64(beta) * pf.DOT9SQ, (DIM,) * 4)
    )

    def layer_gates(lp):
        u = jnp.einsum(
            "sab,sbc,scd->sad",
            pf.rz_batch(lp["single_rz2"]),
            pf.ry_batch(lp["single_ry"]),
            pf.rz_batch(lp["single_rz1"]),
        )
        pair = jnp.einsum("kac,kbd->kabcd", u[0::2], u[1::2]).reshape(-1, 9, 9)
        even = pf.entangler_batch(lp["even_theta"], lp["even_phi"], beta) @ pair
        odd = pf.entangler_batch(lp["odd_theta"], lp["odd_phi"], beta)
        return even, odd

    def apply_layer(state, lp):
        even, odd = layer_gates(lp)
        if circuit_mode == "joint":
            c = tc.QuditCircuit(n, dim=DIM, inputs=state)
            for k in range(even.shape[0]):
                c.unitary(2 * k, 2 * k + 1,
                          unitary=tc.gates.Gate(even[k].reshape((DIM,) * 4)))
            for k in range(odd.shape[0]):
                c.unitary(2 * k + 1, 2 * k + 2,
                          unitary=tc.gates.Gate(odd[k].reshape((DIM,) * 4)))
            return c.state()
        c = tc.QuditCircuit(n, dim=DIM, inputs=state)
        for k in range(even.shape[0]):
            c.unitary(2 * k, 2 * k + 1,
                      unitary=tc.gates.Gate(even[k].reshape((DIM,) * 4)))
        state = c.state()
        c = tc.QuditCircuit(n, dim=DIM, inputs=state)
        for k in range(odd.shape[0]):
            c.unitary(2 * k + 1, 2 * k + 2,
                      unitary=tc.gates.Gate(odd[k].reshape((DIM,) * 4)))
        return c.state()

    def build_state(params):
        if layer_mode == "scan":
            return K.scan(apply_layer, params, sol.initial_state(config))
        state = sol.initial_state(config)
        for layer in range(config["n_layers"]):
            lp = jax.tree.map(lambda x: x[layer], params)
            state = apply_layer(state, lp)
        return state

    def objective(params):
        state = build_state(params)
        c = tc.QuditCircuit(n, dim=DIM, inputs=state)
        energy = K.cast(0.0, "complex64")
        for left in range(n - 1):
            energy += c.expectation((bond_gate, [left, left + 1]))
        onsite = jnp.sum(w_onsite * jnp.abs(state) ** 2)
        return (K.real(energy) + onsite) / n

    return objective


def bench_step(objective, params, config, label, n_meas=20):
    opt = optax.adam(config["learning_rate"])

    def train_step(p, s):
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, v

    jitted = jax.jit(train_step)
    t0 = time.perf_counter()
    low = jitted.lower(params, opt.init(params))
    t_trace = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp = low.compile()
    t_compile = time.perf_counter() - t0
    p, s = params, opt.init(params)
    for _ in range(3):
        p, s, v = comp(p, s)
    jax.block_until_ready(v)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        p, s, v = comp(p, s)
    jax.block_until_ready(v)
    t_step = (time.perf_counter() - t0) / n_meas
    print(
        f"{label}: trace {t_trace:.2f}s compile {t_compile:.2f}s "
        f"step {t_step*1e3:.1f}ms est-total {t_trace+t_compile+t_step*500:.1f}s"
    )


def main():
    config = CONFIG
    params = sol.initial_parameters(config)
    for layer_mode in ("scan", "unroll"):
        for circuit_mode in ("joint", "split"):
            obj = make_objective(config, layer_mode, circuit_mode)
            bench_step(obj, params, config,
                       f"layers={layer_mode:6s} circuits={circuit_mode:5s}")


if __name__ == "__main__":
    main()
