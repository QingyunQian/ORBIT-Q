"""Task-11 optimized variant: Python-unrolled layers (fast per-step) + energy
as a single precomputed sparse Hamiltonian matvec instead of 23 separate
framework expectation contractions.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax
import scipy.sparse as sp
from jax.experimental import sparse as jsparse

_SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
_SY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
_SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
_I3 = np.eye(3, dtype=complex)
_A_XY = np.kron(_SX, _SX) + np.kron(_SY, _SY)
_B_ZZ = np.kron(_SZ, _SZ)
_DOT = np.kron(_SX, _SX) + np.kron(_SY, _SY) + np.kron(_SZ, _SZ)
_C_BIQ = _DOT @ _DOT


def _sparse_H(n, beta, d):
    I = sp.identity(3, format="csr", dtype=complex)
    bond = sp.csr_matrix(_DOT + beta * _C_BIQ)
    sz2 = sp.csr_matrix(_SZ @ _SZ)
    dim = 3 ** n
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(n - 1):
        left = sp.identity(3 ** i, format="csr", dtype=complex)
        right = sp.identity(3 ** (n - i - 2), format="csr", dtype=complex)
        H = H + sp.kron(sp.kron(left, bond), right, format="csr")
    for i in range(n):
        left = sp.identity(3 ** i, format="csr", dtype=complex)
        right = sp.identity(3 ** (n - i - 1), format="csr", dtype=complex)
        H = H + d * sp.kron(sp.kron(left, sz2), right, format="csr")
    return H.tocoo()


def _two_q(theta, phi, beta):
    gen = theta * jnp.asarray(_A_XY, jnp.complex64) + phi * jnp.asarray(_B_ZZ, jnp.complex64) + beta * jnp.asarray(_C_BIQ, jnp.complex64)
    return jax.scipy.linalg.expm(-1j * gen)


def _rz(phi):
    return jnp.diag(jnp.stack([jnp.exp(-1j * phi), jnp.ones((), jnp.complex64), jnp.exp(1j * phi)]))


def _ry(theta):
    sy = jnp.asarray(_SY, jnp.complex64); sy2 = jnp.asarray(_SY @ _SY, jnp.complex64)
    return jnp.eye(3, dtype=jnp.complex64) - 1j * jnp.sin(theta) * sy + (jnp.cos(theta) - 1.0) * sy2


def run_solution(config):
    n = int(config["n_sites"]); n_layers = int(config["n_layers"])
    beta = float(config["beta"]); d = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"]); lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"]); seed = int(config["seed"])

    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    Hc = _sparse_H(n, beta, d)
    Hj = jsparse.BCOO((jnp.asarray(Hc.data, jnp.complex64),
                       jnp.stack([jnp.asarray(Hc.row), jnp.asarray(Hc.col)], axis=1)),
                      shape=Hc.shape)
    mid_m = np.asarray(np.diag([-1.0, 1.0, -1.0]), np.complex64)
    sz_m = np.asarray(_SZ, np.complex64)
    gate = tc.gates.Gate

    def state(params):
        c = tc.QuditCircuit(n, dim=3)
        for q in range(1, n, 2):
            c.any(q, unitary=np.asarray([[0,0,1],[0,1,0],[1,0,0]], np.complex64))
        for l in range(n_layers):
            for q in range(n):
                c.any(q, unitary=_rz(params["rz1"][l, q]) @ _ry(params["ry"][l, q]) @ _rz(params["rz2"][l, q]))
            for k, (a, b) in enumerate(even):
                c.any(a, b, unitary=_two_q(params["et"][l, k], params["ep"][l, k], beta))
            for k, (a, b) in enumerate(odd):
                c.any(a, b, unitary=_two_q(params["ot"][l, k], params["op"][l, k], beta))
        return c.state()

    def energy_density(params):
        psi = state(params)
        return jnp.real(jnp.vdot(psi, Hj @ psi)) / n

    opt = optax.adam(lr)

    @jax.jit
    def step(params, os):
        e, g = jax.value_and_grad(energy_density)(params)
        u, os = opt.update(g, os); return optax.apply_updates(params, u), os, e

    rng = np.random.default_rng(seed)
    mk = lambda *s: scale * jnp.asarray(rng.normal(size=s).astype(np.float32))
    params = {"rz1": mk(n_layers, n), "ry": mk(n_layers, n), "rz2": mk(n_layers, n),
              "et": mk(n_layers, len(even)), "ep": mk(n_layers, len(even)),
              "ot": mk(n_layers, len(odd)), "op": mk(n_layers, len(odd))}
    os = opt.init(params)
    es = []
    for _ in range(max_steps):
        params, os, e = step(params, os); es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), np.float64)

    psi = state(params)
    c = tc.QuditCircuit(n, dim=3, inputs=psi)
    strings = []
    for (i, j) in ((0, 11), (1, 10), (2, 9)):
        ops = [(gate(sz_m), [i])] + [(gate(mid_m), [k]) for k in range(i + 1, j)] + [(gate(sz_m), [j])]
        strings.append(float(jnp.real(c.expectation(*ops))))
    return {"energy_density_history": e_hist,
            "final_energy_density": np.float64(e_hist[-1]),
            "final_string_orders": np.asarray(strings, np.float64)}
