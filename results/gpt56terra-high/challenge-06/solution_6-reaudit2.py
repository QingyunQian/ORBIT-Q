"""Digital-analog VQE using TensorCircuit's differentiable ODE backend."""

import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import tensorcircuit as tc
    from tensorcircuit.analogcircuit import AnalogCircuit

    tc.set_backend("jax")
    n, blocks = config["n_qubits"], config["n_blocks"]
    pauli = tc.gates.PAULI_CHAR_TO_INDEX

    xy, staggered, target, target_weights = [], [], [], []
    for i in range(n - 1):
        for name in ("X", "Y"):
            term = [0] * n
            term[i] = term[i + 1] = pauli[name]
            xy.append(term)
        for name, weight in (("X", 0.7), ("Y", 0.7), ("Z", 1.1)):
            term = [0] * n
            term[i] = term[i + 1] = pauli[name]
            target.append(term)
            target_weights.append(weight)
    for i in range(n):
        term = [0] * n
        term[i] = pauli["Z"]
        staggered.append(term)
        target.append(term)
        target_weights.append(0.25 * (-1.0) ** i)

    hxy = tc.quantum.PauliStringSum2COO(xy, [1.0] * len(xy), numpy=False)
    hz = tc.quantum.PauliStringSum2COO(
        staggered, [(-1.0) ** i for i in range(n)], numpy=False
    )
    htarget = tc.quantum.PauliStringSum2COO(target, target_weights, numpy=False)
    tmin, tmax = config["t_min"], config["t_max"]

    def energy(params):
        s, j, d, alpha, beta, gamma = params
        circuit = AnalogCircuit(n)
        for k in range(1, n, 2):
            circuit.x(k)
        for layer in range(blocks):
            time = tmin + (tmax - tmin) * jax.nn.sigmoid(s[layer])
            coupling, detuning = jnp.tanh(j[layer]), jnp.tanh(d[layer])
            circuit.add_analog_block(
                lambda _, jj=coupling, dd=detuning: jj * hxy + dd * hz,
                time,
                ode_backend="diffrax",
                rtol=config["ode_rtol"],
                atol=config["ode_atol"],
                max_steps=config["ode_max_steps"],
            )
            for k in range(n):
                circuit.rz(k, theta=alpha[layer, k])
                circuit.ry(k, theta=beta[layer, k])
                circuit.rz(k, theta=gamma[layer, k])
        state = circuit.state()
        return jnp.real(jnp.vdot(state, htarget @ state)) / n

    rng = np.random.default_rng(2026)
    angles = rng.normal(0.0, 0.1, size=(3, blocks, n)).astype(np.float32)
    params = (
        jnp.zeros(blocks),
        jnp.full(blocks, 0.1),
        jnp.full(blocks, 0.1),
        jnp.asarray(angles[0]),
        jnp.asarray(angles[1]),
        jnp.asarray(angles[2]),
    )
    value_and_grad = jax.jit(jax.value_and_grad(energy))
    moments = jax.tree.map(jnp.zeros_like, params)
    variances = jax.tree.map(jnp.zeros_like, params)
    history = []
    for step in range(1, config["max_steps"] + 1):
        value, grad = value_and_grad(params)
        history.append(float(value))
        moments = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, moments, grad)
        variances = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, variances, grad)
        params = jax.tree.map(
            lambda p, m, v: p - config["learning_rate"] * (m / (1 - 0.9**step))
            / (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8),
            params,
            moments,
            variances,
        )
    s, j, d, *_ = params
    return {
        "final_analog_times": np.asarray(tmin + (tmax - tmin) * jax.nn.sigmoid(s)),
        "final_analog_couplings": np.asarray(jnp.tanh(j)),
        "final_analog_detunings": np.asarray(jnp.tanh(d)),
        "energy_density_history": np.asarray(history),
    }
