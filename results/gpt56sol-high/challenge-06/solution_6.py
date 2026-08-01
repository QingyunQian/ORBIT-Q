import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    n = int(config["n_qubits"])
    blocks = int(config["n_blocks"])

    def pauli_terms(specifications):
        terms = []
        for sites, pauli in specifications:
            term = [0] * n
            for site in sites:
                term[site] = pauli
            terms.append(term)
        return terms

    xy_terms = pauli_terms(
        [((i, i + 1), p) for i in range(n - 1) for p in (1, 2)]
    )
    z_terms = pauli_terms([((i,), 3) for i in range(n)])
    target_terms = pauli_terms(
        [((i, i + 1), p) for i in range(n - 1) for p in (1, 2, 3)]
    ) + z_terms
    def sparse_operator(terms, weights):
        operator = tc.quantum.PauliStringSum2COO(
            terms, weights, numpy=True
        ).tocsr()
        operator.eliminate_zeros()
        return operator.tocoo()

    xy_matrix = sparse_operator(xy_terms, [1.0] * len(xy_terms))
    z_matrix = sparse_operator(z_terms, [(-1.0) ** i for i in range(n)])
    target_matrix = sparse_operator(
        target_terms,
        [0.7, 0.7, 1.1] * (n - 1)
        + [0.25 * (-1.0) ** i for i in range(n)],
    )
    tc.set_backend("jax")
    xy_matrix = tc.backend.coo_sparse_matrix_from_numpy(xy_matrix)
    z_matrix = tc.backend.coo_sparse_matrix_from_numpy(z_matrix)
    target_matrix = tc.backend.coo_sparse_matrix_from_numpy(target_matrix)

    initial_circuit = tc.Circuit(n)
    for i in range(1, n, 2):
        initial_circuit.x(i)
    initial_state = initial_circuit.state()
    rtol = float(config["ode_rtol"])
    atol = float(config["ode_atol"])
    ode_steps = int(config["ode_max_steps"])
    t_min = float(config["t_min"])
    t_span = float(config["t_max"]) - t_min

    def schrodinger(state, time, coupling, detuning):
        del time
        return -1j * (
            coupling * (xy_matrix @ state) + detuning * (z_matrix @ state)
        )

    def loss(params):
        s, j, d, angles = params
        def hybrid_block(state, layer_params):
            sl, jl, dl, layer_angles = layer_params
            duration = t_min + t_span * jax.nn.sigmoid(sl)
            times = jnp.stack((jnp.asarray(0.0), duration))
            state = tc.timeevol.ode_evol_global(
                schrodinger,
                state,
                times,
                None,
                jnp.tanh(jl),
                jnp.tanh(dl),
                mode="raw",
                rtol=rtol,
                atol=atol,
                max_steps=ode_steps,
            )[-1]
            circuit = tc.Circuit(n, inputs=state)
            for qubit in range(n):
                circuit.rz(qubit, theta=layer_angles[qubit, 0])
                circuit.ry(qubit, theta=layer_angles[qubit, 1])
                circuit.rz(qubit, theta=layer_angles[qubit, 2])
            return circuit.state(), None

        state, _ = jax.lax.scan(hybrid_block, initial_state, (s, j, d, angles))
        norm = jnp.vdot(state, state)
        return jnp.real(jnp.vdot(state, target_matrix @ state) / norm) / n

    rng = np.random.default_rng(42)
    params = (
        jnp.zeros(blocks),
        jnp.full(blocks, 0.1),
        jnp.full(blocks, 0.1),
        jnp.asarray(rng.normal(0.0, 0.1, (blocks, n, 3))),
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    history = []
    for _ in range(int(config["max_steps"])):
        value, gradients = value_and_grad(params)
        history.append(value)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)

    s, j, d, _ = params
    return {
        "final_analog_times": np.asarray(t_min + t_span * jax.nn.sigmoid(s)),
        "final_analog_couplings": np.asarray(jnp.tanh(j)),
        "final_analog_detunings": np.asarray(jnp.tanh(d)),
        "energy_density_history": np.asarray(jnp.stack(history)),
    }
