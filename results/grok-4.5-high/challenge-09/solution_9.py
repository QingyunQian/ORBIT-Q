"""Light-cone VQE for local Paulis on a large irregular ladder circuit."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp


tc.set_backend("jax")
tc.set_dtype("complex64")


def _extract_cone(gate_tape, support_qubits):
    relevant = set(support_qubits)
    gates = []
    for g in reversed(gate_tape):
        if len(g) == 3:
            name, q, pidx = g
            if q in relevant:
                gates.append(g)
        else:
            name, a, b, pidx = g
            if a in relevant or b in relevant:
                relevant.add(a)
                relevant.add(b)
                gates.append(g)
    gates.reverse()
    qubits = sorted(relevant)
    qmap = {q: i for i, q in enumerate(qubits)}
    return qubits, qmap, gates


def _build_cone_data(gate_tape, pauli_terms):
    raw = []
    all_pidx = []
    for coeff, ops in pauli_terms:
        support = [q for _, q in ops]
        qubits, qmap, gates = _extract_cone(gate_tape, support)
        mapped_ops = tuple((p, qmap[q]) for p, q in ops)
        mgates = []
        for g in gates:
            if len(g) == 3:
                name, q, pidx = g
                mgates.append((name, qmap[q], pidx))
                all_pidx.append(pidx)
            else:
                name, a, b, pidx = g
                mgates.append((name, qmap[a], qmap[b], pidx))
                all_pidx.append(pidx)
        raw.append((float(coeff), len(qubits), mgates, mapped_ops))

    unique_p = sorted(set(all_pidx))
    p_to_local = {p: i for i, p in enumerate(unique_p)}
    cone_data = []
    for coeff, nq, mgates, mapped_ops in raw:
        local_gates = []
        for g in mgates:
            if len(g) == 3:
                local_gates.append((g[0], g[1], p_to_local[g[2]]))
            else:
                local_gates.append((g[0], g[1], g[2], p_to_local[g[3]]))
        cone_data.append((coeff, nq, tuple(local_gates), mapped_ops))
    return cone_data, np.asarray(unique_p, dtype=np.int64)


def run_solution(config):
    gate_tape = config["gate_tape"]
    parameter_count = int(config["parameter_count"])
    pauli_terms = config["pauli_terms"]
    n_restarts = int(config["n_restarts"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    cone_data, unique_p = _build_cone_data(gate_tape, pauli_terms)
    n_local = int(unique_p.shape[0])

    def objective(theta_local):
        total = 0.0
        for coeff, nq, local_gates, mapped_ops in cone_data:
            c = tc.Circuit(nq)
            for q in range(nq):
                c.h(q)
            for g in local_gates:
                if len(g) == 3:
                    getattr(c, g[0])(g[1], theta=theta_local[g[2]])
                else:
                    getattr(c, g[0])(g[1], g[2], theta=theta_local[g[3]])
            kw = {}
            for p, q in mapped_ops:
                kw.setdefault(p, []).append(q)
            exp = c.expectation_ps(**kw)
            total = total + coeff * jnp.real(exp)
        return total

    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def optimize_one(theta0):
        def step(carry, _):
            theta, m, v, t = carry
            obj, g = jax.value_and_grad(objective)(theta)
            t = t + 1.0
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * jnp.square(g)
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)
            # Minimize loss = -obj  <=>  ascend on obj with Adam.
            theta = theta + lr * mhat / (jnp.sqrt(vhat) + eps)
            return (theta, m, v, t), obj

        init = (
            theta0,
            jnp.zeros_like(theta0),
            jnp.zeros_like(theta0),
            jnp.asarray(0.0, dtype=theta0.dtype),
        )
        _, objs = jax.lax.scan(step, init, xs=None, length=max_steps)
        return objs

    histories = np.empty((n_restarts, max_steps), dtype=np.float64)
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + 100000 + r)
        full = rng.normal(0.0, scale, size=parameter_count).astype(np.float32)
        theta0 = jnp.asarray(full[unique_p])
        objs = optimize_one(theta0)
        histories[r] = np.asarray(objs, dtype=np.float64)

    return {"observable_history": histories}
