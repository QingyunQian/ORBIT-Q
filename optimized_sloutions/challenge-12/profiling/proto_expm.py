"""Micro-benchmark: batched 4x4 expm alternatives (forward + grad).

Compares jax.scipy.linalg.expm against fixed-order scaling-and-squaring
Pade variants for accuracy (vs float64 reference) and speed inside a
value_and_grad + adam step, mimicking the challenge-12 gate build.
"""

import time

import numpy as np
import optax

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm

_G_NAMES = ["ix", "iy", "iz", "xi", "xx", "xy", "xz", "yi", "yx", "yy", "yz",
            "zi", "zx", "zy", "zz"]
GENS = np.stack([getattr(tc.gates, f"_{n}_matrix") for n in _G_NAMES])
GENS_J = jnp.asarray(GENS, dtype=jnp.complex64)


def expm_pade33_fixed(a, s=4):
    """Fixed scaling-and-squaring with Pade(3,3); a: (..., 4, 4)."""
    a = a / (2**s)
    eye = jnp.eye(4, dtype=a.dtype)
    a2 = a @ a
    u = a @ (a2 + 60.0 * eye)
    v = 12.0 * a2 + 120.0 * eye
    r = jnp.linalg.solve(v - u, v + u)
    for _ in range(s):
        r = r @ r
    return r


def expm_taylor(a, s=4, order=6):
    a = a / (2**s)
    eye = jnp.broadcast_to(jnp.eye(4, dtype=a.dtype), a.shape)
    term = eye
    out = eye
    for k in range(1, order + 1):
        term = term @ a / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def accuracy_check():
    rng = np.random.default_rng(0)
    for scale in (0.05, 0.5, 1.0, 2.0, 4.0):
        theta = rng.normal(scale=scale, size=(31, 15))
        h = np.einsum("gi,iab->gab", theta, GENS)
        ref = np.stack([_expm64(-1j * m) for m in h])
        for name, fn in (
            ("jexpm", lambda x: jexpm(x)),
            ("pade33-s4", lambda x: expm_pade33_fixed(x, 4)),
            ("pade33-s5", lambda x: expm_pade33_fixed(x, 5)),
            ("taylor6-s4", lambda x: expm_taylor(x, 4, 6)),
        ):
            got = np.asarray(fn(jnp.asarray(-1j * h, dtype=jnp.complex64)))
            err = np.max(np.abs(got - ref))
            print(f"scale {scale:4.2f} |H|max {np.abs(h).max():5.2f} "
                  f"{name:10s} maxerr {err:.2e}")
        print()


def _expm64(m):
    import scipy.linalg

    return scipy.linalg.expm(np.asarray(m, dtype=np.complex128))


def bench(fn, label, n=3000):
    opt = optax.adam(0.02)

    def loss_fn(p):
        h = jnp.einsum("gi,iab->gab", p.reshape(31, 15).astype(jnp.complex64),
                       GENS_J)
        u = fn(-1j * h)
        return jnp.sum(jnp.abs(u - jnp.eye(4, dtype=u.dtype)) ** 2)

    def body(carry, _):
        p, s = carry
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s, p)
        return (optax.apply_updates(p, upd), s), loss

    @jax.jit
    def train(p, s):
        return jax.lax.scan(body, (p, s), None, length=n)

    p = jnp.asarray(np.random.default_rng(1).normal(
        scale=0.02, size=(465,)).astype(np.float32))
    s = opt.init(p)
    out = train(p, s)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = train(p, s)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n
    print(f"{label}: {dt*1e3:.3f} ms/step (incl. value_and_grad + adam)")


def bench_adam_only(n=3000):
    opt = optax.adam(0.02)

    def body(carry, _):
        p, s = carry
        loss, g = jax.value_and_grad(lambda q: jnp.sum(q * q))(p)
        upd, s = opt.update(g, s, p)
        return (optax.apply_updates(p, upd), s), loss

    @jax.jit
    def train(p, s):
        return jax.lax.scan(body, (p, s), None, length=n)

    p = jnp.asarray(np.random.default_rng(1).normal(
        scale=0.02, size=(465,)).astype(np.float32))
    s = opt.init(p)
    out = train(p, s)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = train(p, s)
    jax.block_until_ready(out)
    print(f"adam-only floor: {(time.perf_counter()-t0)/n*1e3:.3f} ms/step")


if __name__ == "__main__":
    accuracy_check()
    bench_adam_only()
    bench(jexpm, "jexpm     ")
    bench(lambda a: expm_pade33_fixed(a, 4), "pade33-s4 ")
    bench(lambda a: expm_pade33_fixed(a, 5), "pade33-s5 ")
    bench(lambda a: expm_taylor(a, 4, 6), "taylor6-s4")
