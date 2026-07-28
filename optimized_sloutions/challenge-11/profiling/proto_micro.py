"""Micro-experiments: where does dense-state gate application time go?

1. TC two-qudit gate cost vs bond position (transpose sensitivity).
2. Full-layer application: one circuit vs the same via jnp reshape-matmul
   (floor estimate only; NOT a shippable approach under framework rules).
3. Backward-pass cost of the same layer in both representations.
"""

import sys
import time

sys.path.insert(0, "/workspace/tasks/challenge-11/solution")

import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

DIM = 3
N = 12


def timeit(fn, *args, n=30, label=""):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n
    print(f"{label}: {dt*1e3:.2f} ms")
    return dt


def main():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(DIM**N,)) + 1j * rng.normal(size=(DIM**N,))
    state = jnp.asarray((state / np.linalg.norm(state)).astype(np.complex64))
    gate = jnp.asarray(
        (rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))).astype(
            np.complex64
        )
    )

    # 1. TC gate application cost vs position
    for left in (0, 3, 5, 8, 10):
        def apply_tc(st, left=left):
            c = tc.QuditCircuit(N, dim=DIM, inputs=st)
            c.unitary(
                left, left + 1,
                unitary=tc.gates.Gate(gate.reshape(3, 3, 3, 3)),
            )
            return c.state()

        timeit(jax.jit(apply_tc), state, label=f"TC gate at bond ({left},{left+1})")

    # jnp reshape-matmul at same positions
    for left in (0, 3, 5, 8, 10):
        def apply_jnp(st, left=left):
            psi = st.reshape(3**left, 9, 3 ** (N - left - 2))
            out = jnp.einsum("ab,LbR->LaR", gate, psi)
            return out.reshape(-1)

        timeit(jax.jit(apply_jnp), state, label=f"jnp gate at bond ({left},{left+1})")

    # 2. full layer: 6 even + 5 odd gates
    gates_e = jnp.asarray(
        (rng.normal(size=(6, 9, 9)) + 1j * rng.normal(size=(6, 9, 9))).astype(
            np.complex64
        )
    )
    gates_o = jnp.asarray(
        (rng.normal(size=(5, 9, 9)) + 1j * rng.normal(size=(5, 9, 9))).astype(
            np.complex64
        )
    )

    def layer_tc(st):
        c = tc.QuditCircuit(N, dim=DIM, inputs=st)
        for k in range(6):
            c.unitary(2 * k, 2 * k + 1,
                      unitary=tc.gates.Gate(gates_e[k].reshape(3, 3, 3, 3)))
        for k in range(5):
            c.unitary(2 * k + 1, 2 * k + 2,
                      unitary=tc.gates.Gate(gates_o[k].reshape(3, 3, 3, 3)))
        return c.state()

    t_tc = timeit(jax.jit(layer_tc), state, label="TC full layer (11 gates)")

    def layer_jnp(st):
        psi = st.reshape((9,) * 6)
        # evens: contract axis k with gate k, keep output axis order
        psi = jnp.einsum("Aa,Bb,Cc,Dd,Ee,Ff,abcdef->ABCDEF",
                         *[gates_e[k] for k in range(6)], psi)
        st = psi.reshape(-1)
        for k in range(5):
            left = 2 * k + 1
            p = st.reshape(3**left, 9, 3 ** (N - left - 2))
            st = jnp.einsum("ab,LbR->LaR", gates_o[k], p).reshape(-1)
        return st

    t_jnp = timeit(jax.jit(layer_jnp), state, label="jnp full layer (6-in-1 + 5)")

    # equivalence
    a = jax.jit(layer_tc)(state)
    b = jax.jit(layer_jnp)(state)
    print("layer outputs max|diff|:", float(jnp.max(jnp.abs(a - b))))

    # 3. backward cost of the layer (dummy loss), params = gates
    def loss_tc(ge, go, st):
        c = tc.QuditCircuit(N, dim=DIM, inputs=st)
        for k in range(6):
            c.unitary(2 * k, 2 * k + 1,
                      unitary=tc.gates.Gate(ge[k].reshape(3, 3, 3, 3)))
        for k in range(5):
            c.unitary(2 * k + 1, 2 * k + 2,
                      unitary=tc.gates.Gate(go[k].reshape(3, 3, 3, 3)))
        out = c.state()
        return jnp.sum(jnp.abs(out) ** 2)

    timeit(jax.jit(jax.grad(loss_tc, argnums=(0, 1))), gates_e, gates_o, state,
           n=10, label="grad TC layer wrt gates")


if __name__ == "__main__":
    main()
