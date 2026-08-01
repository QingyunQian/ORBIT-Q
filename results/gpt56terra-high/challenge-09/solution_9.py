import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _cone(tape, term):
    support = {q for _, q in term}
    chosen = []
    for gate in reversed(tape):
        qs = gate[1:-1]
        if support.intersection(qs):
            chosen.append(gate)
            support.update(qs)
    chosen.reverse()
    order = sorted(support)
    local = {q: i for i, q in enumerate(order)}
    gates = []
    for gate in chosen:
        if len(gate) == 3:
            name, q, p = gate
            gates.append((name, local[q], p))
        else:
            name, q0, q1, p = gate
            gates.append((name, local[q0], local[q1], p))
    axes = {"x": [], "y": [], "z": []}
    for axis, q in term:
        axes[axis].append(local[q])
    return len(order), tuple(gates), axes


def run_solution(config):
    tc.set_backend("jax")
    tape = config["gate_tape"]
    cones = [_cone(tape, term) for _, term in config["pauli_terms"]]
    used = sorted({g[-1] for _, gates, _ in cones for g in gates})
    pmap = {p: i for i, p in enumerate(used)}

    def make_term(data):
        nq, gates, axes = data
        def value(theta):
            c = tc.Circuit(nq)
            for q in range(nq):
                c.h(q)
            for gate in gates:
                if len(gate) == 3:
                    name, q, p = gate
                    getattr(c, name)(q, theta=theta[pmap[p]])
                else:
                    name, q0, q1, p = gate
                    getattr(c, name)(q0, q1, theta=theta[pmap[p]])
            return jnp.real(c.expectation_ps(x=axes["x"], y=axes["y"], z=axes["z"]))
        return value

    values = [make_term(c) for c in cones]
    weights = jnp.asarray([term[0] for term in config["pauli_terms"]])
    def objective(theta):
        return sum(w * f(theta) for w, f in zip(weights, values))

    restarts, steps = config["n_restarts"], config["max_steps"]
    starts = np.empty((restarts, len(used)), dtype=np.float32)
    for r in range(restarts):
        full = np.random.default_rng(config["seed"] + 100000 + r).normal(
            0.0, config["initial_parameter_scale"], config["parameter_count"]
        )
        starts[r] = full[used]
    history = np.empty((restarts, steps), dtype=np.float32)
    lr = config["learning_rate"]
    batch_vg = jax.vmap(jax.value_and_grad(objective))
    def evolve(theta):
        def update(carry, step):
            theta, moment, variance = carry
            val, grad = batch_vg(theta)
            moment = 0.9 * moment + 0.1 * grad
            variance = 0.999 * variance + 0.001 * grad * grad
            t = step + 1
            theta = theta + lr * (moment / (1.0 - 0.9**t)) / (
                jnp.sqrt(variance / (1.0 - 0.999**t)) + 1.0e-8
            )
            return (theta, moment, variance), val
        return jax.lax.scan(update, (theta, jnp.zeros_like(theta), jnp.zeros_like(theta)),
                            jnp.arange(steps))[1]
    evolve = jax.jit(evolve)
    for first in range(0, restarts, 8):
        last = min(first + 8, restarts)
        theta = jnp.asarray(starts[first:last])
        history[first:last] = np.asarray(evolve(theta)).T
    return {"observable_history": history}
