import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def run_solution(config):
    n = int(config["n_qubits"])
    g = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    wlog = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    K = tc.backend

    # Sparse TFIM Hamiltonian: -sum ZZ - g sum X
    ps, ws = [], []
    for i in range(n - 1):
        p = [0] * n
        p[i] = 3
        p[i + 1] = 3
        ps.append(p)
        ws.append(-1.0)
    for i in range(n):
        p = [0] * n
        p[i] = 1
        ps.append(p)
        ws.append(-float(g))
    H = tc.quantum.PauliStringSum2COO(ps, ws)

    def init_plus():
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
        return c.state()

    def apply_even(state, bond_params, rx_params):
        c = tc.Circuit(n, inputs=state)
        n_bonds = n // 2
        for i in range(n_bonds):
            a, b = 2 * i, 2 * i + 1
            c.rxx(a, b, theta=2.0 * bond_params[i, 0])
            c.rzz(a, b, theta=2.0 * bond_params[i, 1])
        for q in range(n):
            c.rx(q, theta=rx_params[q])
        return c.state()

    def apply_odd(state, bond_params, rx_params):
        c = tc.Circuit(n, inputs=state)
        n_bonds = n // 2 - 1
        for i in range(n_bonds):
            a, b = 2 * i + 1, 2 * i + 2
            c.rxx(a, b, theta=2.0 * bond_params[i, 0])
            c.rzz(a, b, theta=2.0 * bond_params[i, 1])
        for q in range(n):
            c.rx(q, theta=rx_params[q])
        return c.state()

    def postselect_all(state):
        logs = []
        for q in range(0, n, 2):
            c = tc.Circuit(n, inputs=state)
            c.post_select(q, keep=0)
            st = c.state()
            p = jnp.real(jnp.vdot(st, st))
            logs.append(jnp.log(p + 1e-12))
            state = st / jnp.sqrt(p + 1e-30)
        return state, jnp.stack(logs)

    def pair_step(state, params):
        be, bo, rxe, rxo = params
        state = apply_even(state, be, rxe)
        state, l1 = postselect_all(state)
        state = apply_odd(state, bo, rxo)
        state, l2 = postselect_all(state)
        return state, jnp.concatenate([l1, l2])

    def evaluate(params):
        bond_e, bond_o, rx = params
        state = init_plus()
        rxe = rx[0::2]
        rxo = rx[1::2]
        state, logs = jax.lax.scan(pair_step, state, (bond_e, bond_o, rxe, rxo))
        log_arr = logs.reshape((-1,))
        ket = state.reshape((2 ** n, 1))
        tmp = K.sparse_dense_matmul(H, ket)
        e = jnp.real(jnp.matmul(jnp.conjugate(ket).T, tmp)[0, 0])
        ed = e / n
        mean_log = jnp.mean(log_arr)
        success = jnp.exp(jnp.sum(log_arr))
        loss = ed - wlog * mean_log
        return loss, (ed, success, mean_log)

    # n_steps assumed even (10): 5 even/odd pairs
    n_pairs = n_steps // 2
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.05
    bond_e = scale * jax.random.normal(k1, (n_pairs, n // 2, 2), dtype=jnp.float32)
    bond_o = scale * jax.random.normal(k2, (n_pairs, n // 2 - 1, 2), dtype=jnp.float32)
    rx = scale * jax.random.normal(k3, (n_steps, n), dtype=jnp.float32)
    params = (bond_e, bond_o, rx)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def train_step(carry, _):
        params, opt_state = carry
        (loss, aux), grads = jax.value_and_grad(evaluate, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ed, success, mean_log = aux
        return (params, opt_state), (loss, ed, success, mean_log)

    @jax.jit
    def run_training(params, opt_state):
        (params, opt_state), hist = jax.lax.scan(
            train_step, (params, opt_state), None, length=max_steps
        )
        return hist

    loss_h, ed_h, suc_h, mlog_h = run_training(params, opt_state)

    return {
        "energy_density_history": np.asarray(ed_h, dtype=np.float64),
        "success_probability_history": np.asarray(suc_h, dtype=np.float64),
        "mean_log_probability_history": np.asarray(mlog_h, dtype=np.float64),
        "loss_history": np.asarray(loss_h, dtype=np.float64),
    }
