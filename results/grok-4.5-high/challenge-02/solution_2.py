import numpy as np
import jax
import jax.numpy as jnp
import optax
import scipy.sparse as sp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = int(config["n_qubits"])
    zz_a = float(config["zz_anisotropy"])
    hz = float(config["staggered_field"])
    n_layers = int(config["n_layers"])
    n_blocks = n_layers // 2
    ss = int(config["subsystem_size"])
    target = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    w = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    # even: 2n + 3*(n/2); odd: 2n + 3*((n/2)-1) if n even
    n_even_bonds = n // 2
    n_odd_bonds = n // 2 - 1
    ppb = 2 * n + 3 * n_even_bonds + 2 * n + 3 * n_odd_bonds
    n_params = n_blocks * ppb

    # Sparse -> dense Hamiltonian for <psi|H|psi>
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
    Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))
    I = sp.eye(2, dtype=np.complex64)

    def kron_n(ops):
        out = ops[0]
        for op in ops[1:]:
            out = sp.kron(out, op, format="csr")
        return out

    dim = 2 ** n
    Hsp = sp.csr_matrix((dim, dim), dtype=np.complex64)
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = X
        ops[i + 1] = X
        Hsp = Hsp + kron_n(ops)
        ops = [I] * n
        ops[i] = Y
        ops[i + 1] = Y
        Hsp = Hsp + kron_n(ops)
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        Hsp = Hsp + zz_a * kron_n(ops)
    for i in range(n):
        ops = [I] * n
        ops[i] = Z
        Hsp = Hsp + hz * ((-1) ** i) * kron_n(ops)
    H = jnp.asarray(Hsp.toarray())

    # Initial product state |0101...>
    c0 = tc.Circuit(n)
    for i in range(1, n, 2):
        c0.x(i)
    init = c0.state().astype(jnp.complex64)

    xxm = jnp.asarray(tc.gates._xx_matrix)
    yym = jnp.asarray(tc.gates._yy_matrix)
    zzm = jnp.asarray(tc.gates._zz_matrix)
    even = [(i, i + 1) for i in range(0, n, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    trace_out = list(range(ss, n))

    def apply_block(state, pb):
        c = tc.Circuit(n, inputs=state)
        idx = 0
        ery = pb[idx : idx + n]
        idx += n
        erz = pb[idx : idx + n]
        idx += n
        eint = pb[idx : idx + 3 * n_even_bonds].reshape(n_even_bonds, 3)
        idx += 3 * n_even_bonds
        ory = pb[idx : idx + n]
        idx += n
        orz = pb[idx : idx + n]
        idx += n
        oint = pb[idx : idx + 3 * n_odd_bonds].reshape(n_odd_bonds, 3)

        for q in range(n):
            c.ry(q, theta=ery[q])
            c.rz(q, theta=erz[q])
        for bi, (i, j) in enumerate(even):
            c.exp1(i, j, theta=eint[bi, 0], unitary=xxm)
            c.exp1(i, j, theta=eint[bi, 1], unitary=yym)
            c.exp1(i, j, theta=eint[bi, 2], unitary=zzm)
        for q in range(n):
            c.ry(q, theta=ory[q])
            c.rz(q, theta=orz[q])
        for bi, (i, j) in enumerate(odd):
            c.exp1(i, j, theta=oint[bi, 0], unitary=xxm)
            c.exp1(i, j, theta=oint[bi, 1], unitary=yym)
            c.exp1(i, j, theta=oint[bi, 2], unitary=zzm)

        st = c.state()
        rho = tc.quantum.reduced_density_matrix(st, trace_out)
        ent = jnp.real(tc.quantum.renyi_entropy(rho, 2)).astype(jnp.float32)
        return st, ent

    def loss_fn(params):
        p = params.reshape(n_blocks, ppb)

        def body(state, pb):
            return apply_block(state, pb)

        final_state, ents = jax.lax.scan(body, init, p)
        e = jnp.real(jnp.vdot(final_state, H @ final_state))
        ed = e / n
        mse = jnp.mean((ents - target) ** 2)
        loss = ed + w * mse
        return loss, (ed, mse, ents)

    params = 0.02 * jax.random.normal(
        jax.random.PRNGKey(0), (n_params,), dtype=jnp.float32
    )
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step_fn(carry, _):
        params, opt_state = carry
        (loss, (ed, mse, ents)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (ed, loss, mse, ents)

    @jax.jit
    def train(params, opt_state):
        return jax.lax.scan(step_fn, (params, opt_state), None, length=max_steps)

    (_, _), (ed_h, loss_h, mse_h, ent_h) = train(params, opt_state)

    energy_density_history = np.asarray(ed_h, dtype=np.float64)
    loss_history = np.asarray(loss_h, dtype=np.float64)
    entropy_history = np.asarray(ent_h, dtype=np.float64)
    target_np = np.asarray(config["target_entropies"], dtype=np.float64)
    entropy_mse_history = np.mean((entropy_history - target_np) ** 2, axis=1)

    return {
        "energy_density_history": energy_density_history,
        "loss_history": loss_history,
        "entropy_mse_history": entropy_mse_history,
        "entropy_history": entropy_history,
    }
