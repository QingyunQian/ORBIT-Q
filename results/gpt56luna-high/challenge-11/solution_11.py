import numpy as np
import jax
import jax.numpy as jnp
import tensornetwork as tn
import tensorcircuit as tc

tc.set_backend("jax")
K = tc.backend
D = 3
SX = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.complex64) / jnp.sqrt(2.0)
SY = jnp.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=jnp.complex64) / jnp.sqrt(2.0)
SZ = jnp.diag(jnp.array([1, 0, -1], dtype=jnp.complex64))
XXYY = jnp.kron(SX, SX) + jnp.kron(SY, SY)
ZZ = jnp.kron(SZ, SZ)
DOT = XXYY + ZZ
M = jnp.diag(jnp.array([-1, 1, -1], dtype=jnp.complex64))
SZ2 = SZ @ SZ
SY2 = SY @ SY


def _gate(op):
    return tc.gates.any_gate(op, dim=3)


def _rz(angle):
    return jnp.diag(jnp.exp(-1j * angle * jnp.array([1, 0, -1], dtype=jnp.complex64)))


def _ry(angle):
    return jnp.eye(3, dtype=jnp.complex64) + (jnp.cos(angle) - 1.0) * SY2 - 1j * jnp.sin(angle) * SY


def _entangler(theta, phi, beta):
    return K.expm(-1j * (theta * XXYY + phi * ZZ + beta * (DOT @ DOT)))


def _make_mpo(n, bond_h, onsite):
    mat = np.asarray(bond_h).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    rank = len(s)
    left = jnp.asarray(u.reshape(3, 3, rank))
    right = jnp.asarray((s[:, None] * vh).reshape(rank, 3, 3))
    eye = jnp.eye(3, dtype=jnp.complex64)
    width = rank + 2
    tensors = []
    w = jnp.zeros((1, width, 3, 3), dtype=jnp.complex64).at[0, 0].set(eye).at[0, 1].set(onsite)
    for r in range(rank):
        w = w.at[0, 2 + r].set(left[:, :, r])
    tensors.append(w)
    for _ in range(n - 2):
        w = jnp.zeros((width, width, 3, 3), dtype=jnp.complex64).at[0, 0].set(eye).at[0, 1].set(onsite).at[1, 1].set(eye)
        for r in range(rank):
            w = w.at[0, 2 + r].set(left[:, :, r]).at[2 + r, 1].set(right[r])
        tensors.append(w)
    w = jnp.zeros((width, 1, 3, 3), dtype=jnp.complex64).at[0, 0].set(onsite).at[1, 0].set(eye)
    for r in range(rank):
        w = w.at[2 + r, 0].set(right[r])
    tensors.append(w)
    nodes = [tn.Node(w) for w in tensors]
    for i in range(n - 1):
        nodes[i][1] ^ nodes[i + 1][0]
    return tc.QuOperator([node[2] for node in nodes], [node[3] for node in nodes], nodes,
                         [nodes[0][0], nodes[-1][1]])


def run_solution(config):
    n = int(config["n_sites"])
    nl = int(config["n_layers"])
    beta = float(config["beta"])
    steps = int(config["max_steps"])
    scale = float(config["initial_parameter_scale"])
    lr = float(config["learning_rate"])
    rng = np.random.default_rng(int(config["seed"]))
    per = 3 * n + 2 * (n - 1)
    p0 = rng.normal(0.0, scale, size=(nl, per)).astype(np.float32)
    init = np.zeros((D,) * n, dtype=np.complex64)
    init[tuple(0 if i % 2 == 0 else 2 for i in range(n))] = 1.0
    init = jnp.asarray(init)
    sz_op, m_op = _gate(SZ), _gate(M)
    h_local = DOT + beta * (DOT @ DOT)
    ham = _make_mpo(n, h_local, float(config["single_ion_anisotropy"]) * (SZ @ SZ))

    def energy(params):
        c = tc.QuditCircuit(n, dim=D, inputs=init)
        for l in range(nl):
            q = 0
            for i in range(n):
                c.any(i, unitary=_rz(params[l, q]), name="Rz")
                c.any(i, unitary=_ry(params[l, q + 1]), name="Ry")
                c.any(i, unitary=_rz(params[l, q + 2]), name="Rz")
                q += 3
            for i in range(0, n - 1, 2):
                t, ph = params[l, q], params[l, q + 1]
                u = _entangler(t, ph, beta)
                c.any(i, i + 1, unitary=u, name="U_even")
                q += 2
            for i in range(1, n - 1, 2):
                t, ph = params[l, q], params[l, q + 1]
                u = _entangler(t, ph, beta)
                c.any(i, i + 1, unitary=u, name="U_odd")
                q += 2
        psi = c.quvector()
        return K.real((psi.adjoint() @ ham @ psi).eval()).sum() / n

    params = jnp.asarray(p0)
    value_grad = jax.jit(jax.value_and_grad(energy))
    first_moment, second_moment = jnp.zeros_like(params), jnp.zeros_like(params)
    history = np.empty(steps, dtype=np.float32)
    for step in range(steps):
        value, grad = value_grad(params)
        history[step] = float(value)
        first_moment = 0.9 * first_moment + 0.1 * grad
        second_moment = 0.999 * second_moment + 0.001 * grad * grad
        bc1, bc2 = 1.0 - 0.9 ** (step + 1), 1.0 - 0.999 ** (step + 1)
        params = params - lr * (first_moment / bc1) / (jnp.sqrt(second_moment / bc2) + 1e-8)
    final_energy = float(energy(params))

    def strings(params):
        c = tc.QuditCircuit(n, dim=D, inputs=init)
        for l in range(nl):
            q = 0
            for i in range(n):
                c.any(i, unitary=_rz(params[l, q]), name="Rz")
                c.any(i, unitary=_ry(params[l, q + 1]), name="Ry")
                c.any(i, unitary=_rz(params[l, q + 2]), name="Rz")
                q += 3
            for i in range(0, n - 1, 2):
                u = _entangler(params[l, q], params[l, q + 1], beta)
                c.any(i, i + 1, unitary=u, name="U_even")
                q += 2
            for i in range(1, n - 1, 2):
                u = _entangler(params[l, q], params[l, q + 1], beta)
                c.any(i, i + 1, unitary=u, name="U_odd")
                q += 2
        out = []
        for i, j in ((0, n - 1), (1, n - 2), (2, n - 3)):
            ops = [(_gate(SZ), [i])] + [(_gate(M), [k]) for k in range(i + 1, j)] + [(_gate(SZ), [j])]
            out.append(K.real(c.expectation(*ops)))
        return jnp.stack(out)

    final_strings = np.asarray(strings(params), dtype=np.float64)
    return {"energy_density_history": history, "final_energy_density": final_energy, "final_string_orders": final_strings}
