import numpy as np

import tensorcircuit as tc
import jax
import jax.numpy as jnp


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    n_steps = int(config["max_steps"])

    target_ls = []
    target_ws = []
    for i in range(n - 1):
        for pauli, weight in ((1, 0.7), (2, 0.7), (3, 1.1)):
            row = [0] * n
            row[i] = row[i + 1] = pauli
            target_ls.append(row)
            target_ws.append(weight)
    for i in range(n):
        row = [0] * n
        row[i] = 3
        target_ls.append(row)
        target_ws.append(0.25 * (-1.0) ** i)
    h_target = tc.quantum.PauliStringSum2COO(target_ls, target_ws, numpy=False)

    xy_rows = []
    z_rows = []
    for i in range(n - 1):
        row = [0] * n
        row[i] = row[i + 1] = 1
        xy_rows.append(row)
        row = [0] * n
        row[i] = row[i + 1] = 2
        xy_rows.append(row)
    for i in range(n):
        row = [0] * n
        row[i] = 3
        z_rows.append(row)
    xy_mvp = tc.quantum.PauliStringSum2MVP(xy_rows, [1.0] * len(xy_rows))
    z_mvp = tc.quantum.PauliStringSum2MVP(z_rows, [1.0] * len(z_rows))

    init = tc.Circuit(n)
    for i in range(1, n, 2):
        init.x(i)
    psi0 = init.state()

    def analog(psi, tt, jj, dd):
        def rhs(y, t, a, b):
            return -1j * (jnp.tanh(a) * xy_mvp(y) + jnp.tanh(b) * z_mvp(y))

        out = tc.timeevol.ode_evol_global(
            rhs,
            psi,
            [0.0, tt],
            None,
            jj,
            dd,
            mode="raw",
            rtol=ode_rtol,
            atol=ode_atol,
            max_steps=ode_max_steps,
        )
        return out[-1]

    def digital(psi, rot):
        c = tc.Circuit(n, inputs=psi)
        for k in range(n):
            c.rz(k, theta=rot[k, 0])
            c.ry(k, theta=rot[k, 1])
            c.rz(k, theta=rot[k, 2])
        return c.state()

    def block(psi, rot, ss, jj, dd):
        tt = t_min + (t_max - t_min) * jax.nn.sigmoid(ss)
        return digital(analog(psi, tt, jj, dd), rot)

    block_jit = jax.jit(block)

    def loss(params):
        rot, ss, jj, dd = params
        psi = psi0
        for l in range(n_blocks):
            psi = block_jit(psi, rot[l], ss[l], jj[l], dd[l])
        c = tc.Circuit(n, inputs=psi)
        return jnp.real(tc.templates.measurements.operator_expectation(c, h_target)) / n

    rng = np.random.default_rng(0)
    rot = rng.normal(0.0, 0.1, (n_blocks, n, 3))
    ss = np.zeros(n_blocks)
    jj = np.full(n_blocks, 0.1)
    dd = np.full(n_blocks, 0.1)
    params = [jnp.asarray(x) for x in (rot, ss, jj, dd)]

    loss_grad = jax.value_and_grad(loss)
    moments_m = [jnp.zeros_like(p) for p in params]
    moments_v = [jnp.zeros_like(p) for p in params]
    history = np.empty(n_steps)

    lr = float(config["learning_rate"])
    for step in range(1, n_steps + 1):
        energy, grad = loss_grad(params)
        history[step - 1] = float(energy)

        new_m = []
        new_v = []
        for i, p in enumerate(params):
            m = 0.9 * moments_m[i] + 0.1 * grad[i]
            v = 0.999 * moments_v[i] + 0.001 * grad[i] * grad[i]
            new_m.append(m)
            new_v.append(v)
            m_hat = m / (1.0 - 0.9**step)
            v_hat = v / (1.0 - 0.999**step)
            params[i] = p - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        moments_m, moments_v = new_m, new_v

    rot, ss, jj, dd = [np.asarray(p) for p in params]
    return {
        "final_analog_times": t_min + (t_max - t_min) / (1.0 + np.exp(-ss)),
        "final_analog_couplings": np.tanh(jj),
        "final_analog_detunings": np.tanh(dd),
        "energy_density_history": history,
    }
