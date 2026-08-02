import numpy as np

import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")

    n = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    state = config["dmrg_state"]

    tensors = []
    for i, t in enumerate(state.tensors):
        a = np.asarray(t.data)
        pos = t.inds.index(state.site_ind(i))
        if i == 0:
            b = a[None, :, :] if pos == 1 else a.T[None, :, :]
        elif i == n - 1:
            b = a[:, :, None] if pos == 0 else a.T[:, :, None]
        elif pos == 1:
            b = a
        elif pos == 2:
            b = a.transpose(0, 2, 1)
        else:
            b = a.transpose(1, 0, 2)
        tensors.append(jnp.asarray(b.astype(np.complex64)))

    def make_bra():
        return tc.MPSCircuit(n, tensors=tensors, center_position=0)

    norm2 = float(jnp.real(jax.jit(lambda: make_bra().proj_with_mps(make_bra()))()))
    norm = float(np.sqrt(norm2))
    tensors[:] = [t / norm for t in tensors]

    def make_ket(par):
        c = tc.MPSCircuit(n)
        for i in range(1, n, 2):
            c.X(i)
        for layer in range(n_layers):
            start = layer % 2
            n_gates = (n - start) // 2
            base = layer * (n // 2)
            for gi in range(n_gates):
                i = start + 2 * gi
                off = (base + gi) * 15
                c.apply(tc.gates.su4_gate(par[off : off + 15]), i, i + 1)
        return c

    def overlap_fn(par):
        return make_ket(par).proj_with_mps(make_bra())

    def loss_fn(par):
        ov = overlap_fn(par)
        fid = jnp.clip(jnp.real(ov * jnp.conj(ov)), 0.0, 1.0)
        return 1.0 - fid

    rng = np.random.RandomState(seed)
    p0 = jnp.asarray((scale * rng.randn((n - 1) * 15)).astype(np.float32))

    def adam_step(carry, _):
        par, m, v, b1t, b2t = carry
        loss, g = jax.value_and_grad(loss_fn)(par)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        b1t = 0.9 * b1t
        b2t = 0.999 * b2t
        mh = m / (1.0 - b1t)
        vh = v / (1.0 - b2t)
        par = par - lr * mh / (jnp.sqrt(vh) + 1e-8)
        return (par, m, v, b1t, b2t), (loss, 1.0 - loss)

    carry0 = (
        p0,
        jnp.zeros_like(p0),
        jnp.zeros_like(p0),
        jnp.ones(()),
        jnp.ones(()),
    )
    carry, hist = jax.lax.scan(adam_step, carry0, None, length=max_steps)
    p_final = np.asarray(carry[0])
    ov_final = overlap_fn(jnp.asarray(p_final))

    return {
        "loss_history": np.asarray(hist[0]),
        "fidelity_history": np.asarray(hist[1]),
        "final_parameters": p_final,
        "final_overlap_phase": float(np.angle(ov_final)),
    }
