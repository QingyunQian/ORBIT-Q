import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    blocks = int(config["n_layers"]) // 2
    sub = int(config["subsystem_size"])
    steps = int(config["max_steps"])
    zz_anisotropy = float(config["zz_anisotropy"])
    field = float(config["staggered_field"])
    target = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    init = np.zeros(2**n, dtype=np.complex64)
    init[int("".join(str(i % 2) for i in range(n)), 2)] = 1.0
    x, y, z = (tc.gates.x().tensor, tc.gates.y().tensor, tc.gates.z().tensor)
    xx, yy, zz = jnp.kron(x, x), jnp.kron(y, y), jnp.kron(z, z)

    def rotation(a, b):
        ca, sa = jnp.cos(a / 2), jnp.sin(a / 2)
        eb = jnp.exp(0.5j * b)
        return jnp.array([[ca / eb, -sa / eb], [sa * eb, ca * eb]])

    def rdm(state, sites):
        sites = tuple(sites)
        axes = sites + tuple(i for i in range(n) if i not in sites)
        q = jnp.transpose(jnp.reshape(state, (2,) * n), axes)
        q = jnp.reshape(q, (2 ** len(sites), -1))
        return q @ jnp.conj(q.T)

    def observable(state, op, sites):
        return jnp.real(jnp.sum(rdm(state, sites) * op.T))

    def evaluate(params):
        c = tc.Circuit(n, inputs=init)
        states, q = [], 0
        for _ in range(blocks):
            for parity in (0, 1):
                for i in range(n):
                    c.any(i, unitary=rotation(params[q], params[q + 1]))
                    q += 2
                for i in range(parity, n - 1, 2):
                    h = params[q] * xx + params[q + 1] * yy + params[q + 2] * zz
                    c.any(i, i + 1, unitary=tc.backend.expm(-tc.backend.i() * h))
                    q += 3
            states.append(c.state())
        state = states[-1]
        energy = sum(
            observable(state, xx, (i, i + 1))
            + observable(state, yy, (i, i + 1))
            + zz_anisotropy * observable(state, zz, (i, i + 1))
            for i in range(n - 1)
        )
        energy += sum(field * (-1) ** i * observable(state, z, (i,)) for i in range(n))
        entropies = jnp.stack(
            [-jnp.log(jnp.real(jnp.sum(jnp.abs(rdm(s, range(sub))) ** 2))) for s in states]
        )
        energy_density = energy / n
        loss = energy_density + float(config["entropy_weight"]) * jnp.mean(
            (entropies - target) ** 2
        )
        return loss, (energy_density, entropies)

    value_grad = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
    rng = np.random.default_rng(0)
    params = jnp.asarray(rng.normal(0.0, 0.02, 81 * blocks), dtype=jnp.float32)
    m, v = jnp.zeros_like(params), jnp.zeros_like(params)
    energy_history = np.empty(steps, dtype=np.float64)
    loss_history = np.empty(steps, dtype=np.float64)
    entropy_mse_history = np.empty(steps, dtype=np.float64)
    entropy_history = np.empty((steps, len(config["target_entropies"])), dtype=np.float64)
    lr = float(config["learning_rate"])
    for k in range(steps):
        (loss, (energy, entropies)), grad = value_grad(params)
        ent = np.asarray(entropies)
        loss_history[k] = float(loss)
        energy_history[k] = float(energy)
        entropy_history[k] = ent
        entropy_mse_history[k] = np.mean((ent - np.asarray(config["target_entropies"])) ** 2)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mhat = m / (1.0 - 0.9 ** (k + 1))
        vhat = v / (1.0 - 0.999 ** (k + 1))
        params = params - lr * mhat / (jnp.sqrt(vhat) + 1e-8)
    return {
        "energy_density_history": energy_history,
        "loss_history": loss_history,
        "entropy_mse_history": entropy_mse_history,
        "entropy_history": entropy_history,
    }
