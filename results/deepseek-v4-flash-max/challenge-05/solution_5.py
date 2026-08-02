import numpy as np
import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO

import jax
import jax.numpy as jnp


def run_solution(config):
    n = int(config["n_qubits"])
    h = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    initial = float(config["initial_filter_strength"])

    old_backend = tc.backend.name
    tc.set_backend("jax")
    try:
        paulis = []
        weights = []
        for i in range(n - 1):
            s = [0] * n
            s[i] = 3
            s[i + 1] = 3
            paulis.append(s)
            weights.append(-1.0)
        for i in range(n):
            s = [0] * n
            s[i] = 1
            paulis.append(s)
            weights.append(-h)
        hamiltonian = PauliStringSum2COO(paulis, weights, numpy=False)

        init = np.ones(2**n, dtype=np.complex64) / np.sqrt(2**n)

        def single_filter(a):
            ca = jnp.cosh(a)
            sa = jnp.sinh(a)
            return jnp.array([[ca, sa], [sa, ca]], dtype=jnp.complex64)

        def pair_filter(b):
            eb = jnp.exp(b)
            em = jnp.exp(-b)
            return jnp.diag(
                jnp.array([eb, em, em, eb], dtype=jnp.complex64)
            )

        def loss_fn(p):
            a = p[:n_layers].reshape(n_layers // 2, 2)
            b = p[n_layers:].reshape(n_layers // 2, 2)
            c = tc.Circuit(n, inputs=init)
            for l in range(n_layers):
                u = single_filter(a[l // 2, l % 2])
                v = pair_filter(b[l // 2, l % 2])
                for i in range(n):
                    c.any(i, unitary=u)
                bonds = (
                    range(0, n - 1, 2)
                    if l % 2 == 0
                    else range(1, n - 1, 2)
                )
                for i in bonds:
                    c.any(i, i + 1, unitary=v)
                psi = c.wavefunction()
                psi = psi / jnp.linalg.norm(psi)
                if l + 1 < n_layers:
                    c = tc.Circuit(n, inputs=psi)
            hv = hamiltonian @ psi
            return jnp.real(jnp.dot(jnp.conj(psi), hv)) / n

        def train(p0, steps):
            b1 = jnp.float32(0.9)
            b2 = jnp.float32(0.999)
            eps = jnp.float32(1e-8)
            lrt = jnp.float32(lr)

            def body(carry, _):
                p, m, v, t = carry
                loss, grad = jax.value_and_grad(loss_fn)(p)
                t = t + 1
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad * grad
                mhat = m / (1 - b1**t)
                vhat = v / (1 - b2**t)
                p = p - lrt * mhat / (jnp.sqrt(vhat) + eps)
                return (p, m, v, t), loss

            init_state = (
                p0,
                jnp.zeros_like(p0),
                jnp.zeros_like(p0),
                jnp.array(0, dtype=jnp.int32),
            )
            (p, _, _, _), history = jax.lax.scan(
                body, init_state, None, length=steps
            )
            return p, history

        train_jit = jax.jit(train, static_argnums=(1,))
        p0 = jnp.full(2 * n_layers, initial, dtype=jnp.float32)
        p, history = train_jit(p0, max_steps)
        p = np.asarray(p)
        history = np.asarray(history)
        final_a = p[:n_layers].reshape(n_layers // 2, 2)
        final_b = p[n_layers:].reshape(n_layers // 2, 2)
    finally:
        tc.set_backend(old_backend)

    return {
        "final_a": final_a,
        "final_b": final_b,
        "energy_density_history": history,
    }
