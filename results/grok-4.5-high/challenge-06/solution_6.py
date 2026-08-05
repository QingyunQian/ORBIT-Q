import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO
from tensorcircuit.timeevol import ode_evol_global

tc.set_backend("jax")
tc.set_dtype("complex64")


def _pauli_xy_detuning(n_qubits):
    ls_xy, w_xy = [], []
    for i in range(n_qubits - 1):
        sx = [0] * n_qubits
        sy = [0] * n_qubits
        sx[i], sx[i + 1] = 1, 1
        sy[i], sy[i + 1] = 2, 2
        ls_xy.extend([sx, sy])
        w_xy.extend([1.0, 1.0])
    ls_d, w_d = [], []
    for i in range(n_qubits):
        sz = [0] * n_qubits
        sz[i] = 3
        ls_d.append(sz)
        w_d.append(float((-1) ** i))
    h_xy = PauliStringSum2COO(ls_xy, weight=w_xy, numpy=False)
    h_det = PauliStringSum2COO(ls_d, weight=w_d, numpy=False)
    return h_xy, h_det


def _target_hamiltonian(n_qubits):
    ls, weights = [], []
    for i in range(n_qubits - 1):
        for code, coeff in ((1, 0.7), (2, 0.7), (3, 1.1)):
            s = [0] * n_qubits
            s[i] = code
            s[i + 1] = code
            ls.append(s)
            weights.append(coeff)
    for i in range(n_qubits):
        s = [0] * n_qubits
        s[i] = 3
        ls.append(s)
        weights.append(0.25 * ((-1) ** i))
    return PauliStringSum2COO(ls, weight=weights, numpy=False)


def _neel_state(n_qubits):
    circuit = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        if i % 2 == 1:
            circuit.x(i)
    return circuit.state()


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    h_xy, h_det = _pauli_xy_detuning(n_qubits)
    h_target = _target_hamiltonian(n_qubits)
    psi0 = _neel_state(n_qubits)

    def hamiltonian(t, coupling, detuning):
        return coupling * h_xy + detuning * h_det

    def one_block(psi, block_params):
        s_l, j_l, d_l, angles = block_params
        t_l = t_min + (t_max - t_min) * jax.nn.sigmoid(s_l)
        j_val = jnp.tanh(j_l)
        d_val = jnp.tanh(d_l)
        times = jnp.asarray([0.0, t_l], dtype=jnp.float32)
        psi = ode_evol_global(
            hamiltonian,
            psi,
            times,
            None,
            j_val,
            d_val,
            rtol=ode_rtol,
            atol=ode_atol,
            max_steps=ode_max_steps,
            ode_backend="jaxode",
        )[-1]
        circuit = tc.Circuit(n_qubits, inputs=psi)
        for k in range(n_qubits):
            circuit.rz(k, theta=angles[k, 0])
            circuit.ry(k, theta=angles[k, 1])
            circuit.rz(k, theta=angles[k, 2])
        return circuit.state(), None

    def energy_density(params):
        s_params, j_params, d_params, angles = params
        psi, _ = jax.lax.scan(
            one_block, psi0, (s_params, j_params, d_params, angles)
        )
        return jnp.real(jnp.vdot(psi, h_target @ psi)) / n_qubits

    key = jax.random.PRNGKey(0)
    params = (
        jnp.zeros((n_blocks,), dtype=jnp.float32),
        jnp.full((n_blocks,), 0.1, dtype=jnp.float32),
        jnp.full((n_blocks,), 0.1, dtype=jnp.float32),
        0.1
        * jax.random.normal(key, (n_blocks, n_qubits, 3), dtype=jnp.float32),
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(energy_density))

    # Trigger compilation once before the timed history loop work.
    energy0, grads0 = value_and_grad(params)
    energy0.block_until_ready()
    del energy0, grads0

    history = np.empty(max_steps, dtype=np.float64)
    for step in range(max_steps):
        energy, grads = value_and_grad(params)
        history[step] = float(energy)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    s_params, j_params, d_params, _ = params
    final_times = t_min + (t_max - t_min) * jax.nn.sigmoid(s_params)
    final_couplings = jnp.tanh(j_params)
    final_detunings = jnp.tanh(d_params)

    return {
        "final_analog_times": np.asarray(final_times, dtype=np.float64),
        "final_analog_couplings": np.asarray(final_couplings, dtype=np.float64),
        "final_analog_detunings": np.asarray(final_detunings, dtype=np.float64),
        "energy_density_history": history,
    }
