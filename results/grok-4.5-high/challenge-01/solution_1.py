"""DMRG-MPS + shallow brickwork VQE refinement in TensorCircuit-NG."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def _quimb_to_tensors(psi):
    tensors = []
    for i in range(psi.L):
        t = np.asarray(psi.arrays[i], dtype=np.complex64)
        if t.ndim == 2:
            t = t.reshape(1, t.shape[0], t.shape[1]) if i == 0 else t.reshape(t.shape[0], t.shape[1], 1)
        tensors.append(jnp.asarray(t))
    return tensors


def _nparams(n, n_layers):
    c = 0
    for layer in range(n_layers):
        c += n * 3
        bonds = range(0, n - 1, 2) if layer % 2 == 0 else range(1, n - 1, 2)
        c += len(list(bonds)) * 3
    return c


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["field"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    tensors = _quimb_to_tensors(config["dmrg_state"])
    # modest truncation keeps two-site SVD differentiable / JIT-friendly
    max_bond = 16
    nparams = _nparams(n, n_layers)

    def energy(params):
        c = tc.MPSCircuit(n, tensors=tensors, split={"max_singular_values": max_bond})
        idx = 0
        for layer in range(n_layers):
            for q in range(n):
                c.rz(q, theta=params[idx])
                c.ry(q, theta=params[idx + 1])
                c.rz(q, theta=params[idx + 2])
                idx += 3
            bonds = list(range(0, n - 1, 2)) if layer % 2 == 0 else list(range(1, n - 1, 2))
            for b in bonds:
                # exp[-i(θxx XX + θyy YY + θzz ZZ)] via commuting RXX/RYY/RZZ
                c.rxx(b, b + 1, theta=2.0 * params[idx])
                c.ryy(b, b + 1, theta=2.0 * params[idx + 1])
                c.rzz(b, b + 1, theta=2.0 * params[idx + 2])
                idx += 3
        e = jnp.float32(0.0)
        for i in range(n - 1):
            e = e - jnp.real(c.expectation_ps(z=[i, i + 1])).astype(jnp.float32)
        f = jnp.float32(field)
        for i in range(n):
            e = e - f * jnp.real(c.expectation_ps(x=[i])).astype(jnp.float32)
        return e

    params0 = jnp.zeros((nparams,), dtype=jnp.float32)
    opt = optax.adam(lr)
    opt_state0 = opt.init(params0)

    def step(carry, _):
        params, opt_state = carry
        e, g = jax.value_and_grad(energy)(params)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), e

    # JIT the scanned optimization (compile once)
    @jax.jit
    def run_all(params, opt_state):
        (params, opt_state), hist = jax.lax.scan(step, (params, opt_state), None, length=max_steps)
        return hist

    hist = run_all(params0, opt_state0)
    hist = np.asarray(hist, dtype=np.float64)
    return {"energy_history": hist}
