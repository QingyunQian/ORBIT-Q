import numpy as np
import jax
import tensorcircuit as tc
from jax.experimental import sparse as jsparse
from tensorcircuit.quantum import PauliStringSum2COO
from tensorcircuit.templates.measurements import sparse_expectation


def run_solution(config):
    n = int(config["n_qubits"])
    nb = int(config["n_blocks"])
    tc.set_backend("numpy")
    xy = []
    zz = []
    for i in range(n - 1):
        for p in (1, 2):
            a = [0] * n
            a[i] = a[i + 1] = p
            xy.append(a)
        a = [0] * n
        a[i] = a[i + 1] = 3
        zz.append(a)
    field = [[3 if j == i else 0 for j in range(n)] for i in range(n)]
    target = xy + zz + field
    target_w = ([0.7] * len(xy) + [1.1] * len(zz) +
                 [0.25 * (-1.0) ** i for i in range(n)])
    hxy = PauliStringSum2COO(xy, numpy=True)
    hz = PauliStringSum2COO(field, weight=[(-1.0) ** i for i in range(n)], numpy=True)
    ht = PauliStringSum2COO(target, weight=target_w, numpy=True)
    jax.config.update("jax_enable_x64", False)
    tc.set_backend("jax")
    hxy = jsparse.BCSR.from_scipy_sparse(hxy)
    hz = jsparse.BCSR.from_scipy_sparse(hz)
    ht = tc.backend.coo_sparse_matrix_from_numpy(ht)

    initial = np.zeros(2**n, dtype=np.complex64)
    initial[int("01010101010101", 2)] = 1.0
    rng = np.random.default_rng(0)
    angles = rng.normal(0.0, 0.1, (nb, n, 3)).astype(np.float32)
    params = np.concatenate((np.zeros(nb, np.float32),
                             np.full(2 * nb, 0.1, np.float32),
                             angles.reshape(-1)))
    tmin, tmax = config["t_min"], config["t_max"]
    rtol, atol = config["ode_rtol"], config["ode_atol"]
    ode_steps = int(config["ode_max_steps"])

    def block(psi, s, j, d, a):
        coupling = tc.backend.tanh(j)
        detuning = tc.backend.tanh(d)
        time = tmin + (tmax - tmin) / (1.0 + tc.backend.exp(-s))
        ac = tc.AnalogCircuit(n, inputs=psi)
        ac.add_analog_block(
            lambda t: (lambda v: coupling * (hxy @ v) + detuning * (hz @ v)),
            time, rtol=rtol, atol=atol, max_steps=ode_steps)
        for k in range(n):
            ac.rz(k, theta=a[k, 0])
            ac.ry(k, theta=a[k, 1])
            ac.rz(k, theta=a[k, 2])
        return ac.state()

    block = tc.backend.jit(block)

    def energy(p):
        psi = tc.backend.convert_to_tensor(initial)
        ss, jj, dd = p[:nb], p[nb:2 * nb], p[2 * nb:3 * nb]
        aa = p[3 * nb:].reshape(nb, n, 3)
        for l in range(nb):
            psi = block(psi, ss[l], jj[l], dd[l], aa[l])
        return tc.backend.real(sparse_expectation(tc.Circuit(n, inputs=psi), ht)) / n

    value_grad = tc.backend.value_and_grad(energy)
    history = np.empty(int(config["max_steps"]), dtype=np.float64)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    lr = float(config["learning_rate"])
    for step in range(len(history)):
        value, grad = value_grad(tc.backend.convert_to_tensor(params))
        history[step] = float(value)
        g = np.asarray(grad)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mhat = m / (1.0 - 0.9 ** (step + 1))
        vhat = v / (1.0 - 0.999 ** (step + 1))
        params = params - lr * mhat / (np.sqrt(vhat) + 1e-8)

    return {
        "final_analog_times": np.asarray(tmin + (tmax - tmin) /
                                           (1.0 + np.exp(-params[:nb]))),
        "final_analog_couplings": np.asarray(np.tanh(params[nb:2 * nb])),
        "final_analog_detunings": np.asarray(np.tanh(params[2 * nb:3 * nb])),
        "energy_density_history": history,
    }
