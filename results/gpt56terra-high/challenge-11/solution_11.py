import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")


def run_solution(config):
    n, layers, beta0 = config["n_sites"], config["n_layers"], config["beta"]
    dion, steps, lr = config["single_ion_anisotropy"], config["max_steps"], config["learning_rate"]
    rt2 = np.sqrt(2.0)
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex) / rt2
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex) / rt2
    sz = np.diag([1., 0., -1.]).astype(complex)
    zz = np.kron(sz, sz)
    xy = np.kron(sx, sx) + np.kron(sy, sy)
    dot = xy + zz
    h2 = dot + beta0 * dot @ dot
    z2 = sz @ sz
    q = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], complex)
    h2, z2, q = map(jnp.asarray, (h2, z2, q))
    parity = jnp.diag(jnp.array([-1., 1., -1.]))
    # The only nontrivial three-dimensional magnetization block of the entangler.
    a0 = jnp.asarray(xy[np.ix_([2, 4, 6], [2, 4, 6])])
    z0 = jnp.asarray(zz[np.ix_([2, 4, 6], [2, 4, 6])])
    c0 = (a0 + z0) @ (a0 + z0)
    ids0 = jnp.asarray([2, 4, 6])
    ids1 = jnp.asarray([1, 3, 5, 7])

    def one(a, b, g):
        co, si = jnp.cos(b), jnp.sin(b)
        ry = jnp.array([[(1 + co) / 2, -si / rt2, (1 - co) / 2],
                        [si / rt2, co, -si / rt2],
                        [(1 - co) / 2, si / rt2, (1 + co) / 2]], dtype=jnp.complex64)
        return jnp.diag(jnp.array([jnp.exp(-1j * a), 1, jnp.exp(1j * a)])) @ ry @ jnp.diag(jnp.array([jnp.exp(-1j * g), 1, jnp.exp(1j * g)]))

    def two(x):
        th, ph = x
        ev, vv = jnp.linalg.eigh(th * a0 + ph * z0 + beta0 * c0)
        u0 = (vv * jnp.exp(-1j * ev)) @ vv.conj().T
        u = jnp.zeros((9, 9), dtype=jnp.complex64)
        u = u.at[0, 0].set(jnp.exp(-1j * (ph + beta0)))
        u = u.at[8, 8].set(jnp.exp(-1j * (ph + beta0)))
        u = u.at[ids0[:, None], ids0].set(u0)
        e = jnp.exp(-1j * beta0)
        for p, r in ((1, 3), (5, 7)):
            u = u.at[p, p].set(e * jnp.cos(th)).at[r, r].set(e * jnp.cos(th))
            u = u.at[p, r].set(-1j * e * jnp.sin(th)).at[r, p].set(-1j * e * jnp.sin(th))
        return u

    nb = n - 1
    def circuit(p):
        c = tc.QuditCircuit(n, dim=3)
        for i in range(1, n, 2): c.any(i, unitary=q)
        k = 0
        for _ in range(layers):
            for i in range(n):
                c.any(i, unitary=one(p[k], p[k + 1], p[k + 2])); k += 3
            gs = jax.vmap(two)(p[k:k + 2 * nb].reshape(nb, 2)); k += 2 * nb
            for i in range(0, n - 1, 2): c.any(i, i + 1, unitary=gs[i])
            for i in range(1, n - 1, 2): c.any(i, i + 1, unitary=gs[i])
        return c

    def energy(p):
        c = circuit(p)
        e = sum(c.expectation((h2, [i, i + 1])) for i in range(n - 1))
        e += dion * sum(c.expectation((z2, [i])) for i in range(n))
        return jnp.real(e) / n

    def string_orders(p):
        c = circuit(p)
        return jnp.real(jnp.array([c.expectation((sz, [i]), *((parity, [k]) for k in range(i + 1, j)), (sz, [j])) for i, j in ((0, n - 1), (1, n - 2), (2, n - 3))]))

    def update(carry, t):
        p, m, v = carry
        val, grad = jax.value_and_grad(energy)(p)
        m, v = .9 * m + .1 * grad, .999 * v + .001 * grad * grad
        p = p - lr * (m / (1 - .9 ** (t + 1))) / (jnp.sqrt(v / (1 - .999 ** (t + 1))) + 1e-8)
        return (p, m, v), val

    optimize = jax.jit(lambda x: jax.lax.scan(update, (x, jnp.zeros_like(x), jnp.zeros_like(x)), jnp.arange(steps)))
    rng = np.random.default_rng(config["seed"])
    p = jnp.asarray(rng.normal(0, config["initial_parameter_scale"], layers * (3 * n + 2 * nb)), dtype=jnp.float32)
    (p, _, _), hist = optimize(p)
    return {"energy_density_history": np.asarray(hist), "final_energy_density": float(energy(p)), "final_string_orders": np.asarray(string_orders(p))}
