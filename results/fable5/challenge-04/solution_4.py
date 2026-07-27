"""Challenge 4: trainable Kraus noise calibration from multi-circuit data.

The asymmetric bit-flip channel is expressed explicitly as Kraus tensor
algebra (a (3, 2, 2) Kraus stack) and inserted after each fixed RXX
entangler. Every noisy observable Tr[O K(|psi><psi|)] is evaluated exactly as
a ket/bra Kraus-ladder tensor network built from TensorCircuit gate tensors
and contracted by the framework's tensornetwork engine, so the whole model
is differentiable in the two channel probabilities. The synthetic target
table is generated first from the true probabilities with the same probe
circuits, then the probabilities are fitted by Adam on the table MSE.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax
import tensornetwork as tn
from tensorcircuit import gates as g


def _kraus_stack(p01, p10):
    # K0 = sqrt(1-p01)|0><0| + sqrt(1-p10)|1><1|; K1 = sqrt(p10)|0><1|;
    # K2 = sqrt(p01)|1><0|. Index order (a, out, in).
    k = jnp.zeros((3, 2, 2), dtype=jnp.complex128)
    k = k.at[0, 0, 0].set(jnp.sqrt(1.0 - p01))
    k = k.at[0, 1, 1].set(jnp.sqrt(1.0 - p10))
    k = k.at[1, 0, 1].set(jnp.sqrt(p10))
    k = k.at[2, 1, 0].set(jnp.sqrt(p01))
    return k


def _probe_mps(which, n):
    # (l, p, r) site tensors for the four fixed probe inputs.
    if which == "ghz":
        first = np.zeros((1, 2, 2)); first[0, 0, 0] = first[0, 1, 1] = 1 / np.sqrt(2)
        mid = np.zeros((2, 2, 2)); mid[0, 0, 0] = mid[1, 1, 1] = 1.0
        last = np.zeros((2, 2, 1)); last[0, 0, 0] = last[1, 1, 0] = 1.0
        ts = [first] + [mid] * (n - 2) + [last]
    elif which == "bell":
        a = np.zeros((1, 2, 2)); a[0, 0, 0] = a[0, 1, 1] = 1 / np.sqrt(2)
        b = np.zeros((2, 2, 1)); b[0, 1, 0] = b[1, 0, 0] = 1.0
        ts = [a, b] * (n // 2)
    elif which == "zero":
        t = np.zeros((1, 2, 1)); t[0, 0, 0] = 1.0
        ts = [t] * n
    else:  # plus
        t = np.ones((1, 2, 1)) / np.sqrt(2)
        ts = [t] * n
    return [jnp.asarray(t, dtype=jnp.complex128) for t in ts]


def _noisy_expectation(mps, kmat, gate_t, bonds_seq, z_sites, n):
    # Tr[(prod Z_{z_sites}) K(|psi><psi|)] as a ket/bra ladder with the Kraus
    # index of each channel contracted between the two layers.
    ket = [tn.Node(t) for t in mps]
    bra = [tn.Node(jnp.conj(t)) for t in mps]
    for a, b in zip(ket[:-1], ket[1:]):
        a[2] ^ b[0]
    for a, b in zip(bra[:-1], bra[1:]):
        a[2] ^ b[0]
    ones = lambda: tn.Node(jnp.ones(1, jnp.complex128))
    caps = [ones() for _ in range(4)]
    caps[0][0] ^ ket[0][0]; caps[1][0] ^ ket[-1][2]
    caps[2][0] ^ bra[0][0]; caps[3][0] ^ bra[-1][2]
    nodes = ket + bra + caps
    ket_w = [nd[1] for nd in ket]
    bra_w = [nd[1] for nd in bra]

    def channel(q):
        kk, kb = tn.Node(kmat), tn.Node(jnp.conj(kmat))
        kk[2] ^ ket_w[q]
        kb[2] ^ bra_w[q]
        kk[0] ^ kb[0]
        ket_w[q], bra_w[q] = kk[1], kb[1]
        nodes.extend([kk, kb])

    for bonds in bonds_seq:
        for (a, b) in bonds:
            gk, gb = tn.Node(gate_t), tn.Node(jnp.conj(gate_t))
            gk[2] ^ ket_w[a]; gk[3] ^ ket_w[b]
            gb[2] ^ bra_w[a]; gb[3] ^ bra_w[b]
            ket_w[a], ket_w[b] = gk[0], gk[1]
            bra_w[a], bra_w[b] = gb[0], gb[1]
            nodes.extend([gk, gb])
            channel(a)
            channel(b)

    zt = jnp.asarray(np.diag([1.0, -1.0]), dtype=jnp.complex128)
    for q in range(n):
        if q in z_sites:
            zn = tn.Node(zt)
            zn[1] ^ ket_w[q]
            zn[0] ^ bra_w[q]
            nodes.append(zn)
        else:
            ket_w[q] ^ bra_w[q]
    return tc.backend.real(tn.contractors.auto(nodes).tensor)


def run_solution(config):
    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    gate_t = g.rxx(theta=theta).tensor  # rxx(theta) = exp(-i theta X X / 2)
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    mpses = [_probe_mps(w, n) for w in ("ghz", "bell", "zero", "plus")]

    def table(p01, p10):
        kmat = _kraus_stack(p01, p10)
        rows = []
        for mps in mpses:
            obs = [_noisy_expectation(mps, kmat, gate_t, (even, odd), {i}, n) for i in range(n)]
            obs.append(_noisy_expectation(mps, kmat, gate_t, (even, odd), set(range(n)), n))
            rows.append(jnp.stack(obs))
        return jnp.stack(rows)

    table_jit = jax.jit(table)
    # 1) generate the synthetic target table at the true probabilities
    target = table_jit(float(config["true_p01"]), float(config["true_p10"]))

    # 2) fit sigmoid-parameterized probabilities against the target table
    def loss_fn(r):
        p = jax.nn.sigmoid(r)
        return jnp.mean((table(p[0], p[1]) - target) ** 2)

    vg = jax.jit(jax.value_and_grad(loss_fn))
    p_init = np.array([float(config["initial_p01"]), float(config["initial_p10"])])
    r = jnp.asarray(np.log(p_init / (1.0 - p_init)))
    opt = optax.adam(lr)
    opt_state = opt.init(r)

    loss_hist = np.empty(max_steps, dtype=np.float64)
    for k in range(max_steps):
        loss, grad = vg(r)
        loss_hist[k] = float(loss)
        updates, opt_state = opt.update(grad, opt_state)
        r = optax.apply_updates(r, updates)

    p_final = jax.nn.sigmoid(r)
    fitted = table_jit(p_final[0], p_final[1])
    return {
        "loss_history": loss_hist,
        "final_probabilities": np.asarray(p_final, dtype=np.float64),
        "fitted_expectations": np.asarray(fitted, dtype=np.float64),
    }
