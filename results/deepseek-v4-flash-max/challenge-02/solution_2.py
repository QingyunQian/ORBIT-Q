import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.quantum import reduced_density_matrix, renyi_entropy

tc.set_backend("jax")

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex64)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
_XX = np.kron(_X, _X)
_YY = np.kron(_Y, _Y)
_ZZ = np.kron(_Z, _Z)


def run_solution(config):
    n = int(config["n_qubits"])
    sub = int(config["subsystem_size"])
    blocks = int(config["n_layers"]) // 2
    zz = float(config["zz_anisotropy"])
    h = float(config["staggered_field"])
    weight = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    target = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    bk = tc.cons.backend

    def build_circuit(params):
        c = tc.Circuit(n)
        for i in range(1, n, 2):
            c.x(i)
        s2s = []
        idx = 0
        for _ in range(blocks):
            for i in range(n):
                c.ry(i, theta=params[idx])
                idx += 1
            for i in range(n):
                c.rz(i, theta=params[idx])
                idx += 1
            for k in range(n // 2):
                i, j = 2 * k, 2 * k + 1
                u = params[idx] * _XX + params[idx + 1] * _YY + params[idx + 2] * _ZZ
                idx += 3
                c.exp(i, j, unitary=u, theta=1.0)
            for i in range(n):
                c.ry(i, theta=params[idx])
                idx += 1
            for i in range(n):
                c.rz(i, theta=params[idx])
                idx += 1
            for k in range(n // 2 - 1):
                i, j = 2 * k + 1, 2 * k + 2
                u = params[idx] * _XX + params[idx + 1] * _YY + params[idx + 2] * _ZZ
                idx += 3
                c.exp(i, j, unitary=u, theta=1.0)
            s = c.wavefunction()
            rho = reduced_density_matrix(s, subsystem_to_keep=range(sub))
            s2s.append(bk.real(renyi_entropy(rho, k=2)))
        return c, jnp.stack(s2s)

    def loss_fn(params):
        c, s2s = build_circuit(params)
        s = c.wavefunction()
        e = 0.0
        for i in range(n - 1):
            e += bk.real(tc.expectation((tc.gates.x(), [i]), (tc.gates.x(), [i + 1]), ket=s))
            e += bk.real(tc.expectation((tc.gates.y(), [i]), (tc.gates.y(), [i + 1]), ket=s))
            e += zz * bk.real(tc.expectation((tc.gates.z(), [i]), (tc.gates.z(), [i + 1]), ket=s))
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            e += sign * h * bk.real(tc.expectation((tc.gates.z(), [i]), ket=s))
        energy_density = e / n
        mse = jnp.mean((s2s - target) ** 2)
        loss = energy_density + weight * mse
        return loss, (energy_density, s2s, mse)

    size = blocks * (7 * n - 3)
    params = jnp.asarray(np.random.default_rng(0).normal(0.0, 0.02, size=size), dtype=jnp.float32)
    vg = jax.jit(bk.value_and_grad(loss_fn, has_aux=True))
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    energy_hist = np.empty(max_steps, dtype=np.float32)
    loss_hist = np.empty(max_steps, dtype=np.float32)
    mse_hist = np.empty(max_steps, dtype=np.float32)
    ent_hist = np.empty((max_steps, 3), dtype=np.float32)

    for step in range(1, max_steps + 1):
        (loss, (energy_density, s2s, mse)), grad = vg(params)
        energy_hist[step - 1] = np.asarray(energy_density)
        loss_hist[step - 1] = np.asarray(loss)
        mse_hist[step - 1] = np.asarray(mse)
        ent_hist[step - 1] = np.asarray(s2s)

        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        m_hat = m / (1.0 - 0.9**step)
        v_hat = v / (1.0 - 0.999**step)
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

    return {
        "energy_density_history": energy_hist,
        "loss_history": loss_hist,
        "entropy_mse_history": mse_hist,
        "entropy_history": ent_hist,
    }
