"""Train an asymmetric Kraus bit-flip channel with TensorCircuit-NG."""

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    bonds += [(i, i + 1) for i in range(1, n - 1, 2)]

    def kraus(p01, p10):
        # K0, K1 (1 -> 0), K2 (0 -> 1), expressed as channel tensors.
        return [
            jnp.array([[jnp.sqrt(1.0 - p01), 0], [0, jnp.sqrt(1.0 - p10)]], jnp.complex64),
            jnp.array([[0, jnp.sqrt(p10)], [0, 0]], jnp.complex64),
            jnp.array([[0, 0], [jnp.sqrt(p01), 0]], jnp.complex64),
        ]

    def probe(params, kind):
        c = tc.DMCircuit(n)
        if kind == 0:  # GHZ
            c.h(0)
            for q in range(1, n):
                c.cnot(0, q)
        elif kind == 1:  # tensor product of (|01> + |10>)/sqrt(2) pairs
            for q in range(0, n, 2):
                c.x(q + 1)
                c.h(q)
                c.cnot(q, q + 1)
        elif kind == 3:  # |+>^n
            for q in range(n):
                c.h(q)
        for a, b in bonds:
            c.rxx(a, b, theta=theta)
            c.general_kraus(kraus(params[0], params[1]), a)
            c.general_kraus(kraus(params[0], params[1]), b)
        z = [tc.backend.real(c.expectation((tc.gates.z(), [q]), reuse=False)) for q in range(n)]
        parity = tc.backend.real(c.expectation(*[(tc.gates.z(), [q]) for q in range(n)], reuse=False))
        return jnp.stack(z + [parity])

    probes = [jax.jit(lambda params, kind=kind: probe(params, kind)) for kind in range(4)]
    [REDACTED]_params = jnp.array([config["[REDACTED]_p01"], config["[REDACTED]_p10"]], dtype=jnp.float32)
    target = [f([REDACTED]_params) for f in probes]

    def make_value_and_grad(circuit):
        def loss(raw, observed):
            return jnp.mean((circuit(jax.nn.sigmoid(raw)) - observed) ** 2)
        return jax.jit(jax.value_and_grad(loss))

    value_and_grads = [make_value_and_grad(f) for f in probes]
    initial = jnp.array([config["initial_p01"], config["initial_p10"]], dtype=jnp.float32)
    raw = jnp.log(initial / (1.0 - initial))
    m = jnp.zeros(2, dtype=jnp.float32)
    v = jnp.zeros(2, dtype=jnp.float32)
    history = []
    rate, beta1, beta2, eps = float(config["learning_rate"]), 0.9, 0.999, 1e-8
    for step in range(1, int(config["max_steps"]) + 1):
        parts = [f(raw, observed) for f, observed in zip(value_and_grads, target)]
        loss = sum(item[0] for item in parts) / 4.0
        grad = sum(item[1] for item in parts) / 4.0
        history.append(loss)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        raw = raw - rate * (m / (1.0 - beta1**step)) / (jnp.sqrt(v / (1.0 - beta2**step)) + eps)
    fitted = jnp.stack([f(jax.nn.sigmoid(raw)) for f in probes])
    return {
        "loss_history": np.asarray(jnp.stack(history)),
        # Return probabilities in NumPy double precision so an external Kraus
        # completeness check is not limited by JAX's default float32 rounding.
        "final_probabilities": np.asarray(jax.nn.sigmoid(raw), dtype=np.float64),
        "fitted_expectations": np.asarray(fitted),
    }
