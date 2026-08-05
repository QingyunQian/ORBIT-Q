import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def run_solution(config):
    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    scale = float(config["initial_parameter_scale"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    hfield = jnp.float32(config["transverse_field"])

    # Per layer: data RY, anc RY, ent RZZ, fb0 RZZ, fb1 RZZ, post RZ
    n_blocks = 6
    n_params = n_layers * n_blocks * n_data

    rng = np.random.default_rng(seed)
    params0 = jnp.asarray(rng.normal(0.0, scale, size=(n_params,)), dtype=jnp.float32)
    # Fixed uniforms keep mid-circuit measurement outcomes reproducible across steps.
    status = jnp.asarray(
        rng.uniform(size=(n_traj, n_layers, n_anc)), dtype=jnp.float32
    )

    # Local bit helpers for data-qubit Hamiltonian expectations on the final state.
    bit_ids = jnp.arange(2**n_qubits, dtype=jnp.int32)

    def apply_z(psi, q):
        bit = (bit_ids >> (n_qubits - 1 - q)) & 1
        signs = (1 - 2 * bit).astype(psi.dtype)
        return psi * signs

    def apply_x(psi, q):
        flip = bit_ids ^ (1 << (n_qubits - 1 - q))
        return psi[flip]

    def energy_from_state(psi):
        hpsi = jnp.zeros_like(psi)
        for i in range(n_data - 1):
            hpsi = hpsi - apply_z(apply_z(psi, i), i + 1)
        for i in range(n_data):
            hpsi = hpsi - hfield * apply_x(psi, i)
        return jnp.real(jnp.vdot(psi, hpsi))

    def energy_one(params, status_one):
        p = params.reshape((n_layers, n_blocks, n_data))
        c = tc.Circuit(n_qubits)
        for layer in range(n_layers):
            for i in range(n_data):
                c.ry(i, theta=p[layer, 0, i])
            for i in range(n_anc):
                c.ry(n_data + i, theta=p[layer, 1, i])
            for i in range(n_data):
                c.rzz(n_data + i, i, theta=p[layer, 2, i])
            for i in range(n_anc - 1):
                c.cnot(n_data + i, n_data + i + 1)
            for i in range(n_anc):
                m = c.cond_measure(n_data + i, status=status_one[layer, i])
                g0 = tc.gates.rzz(theta=p[layer, 3, i])
                g1 = tc.gates.rzz(theta=p[layer, 4, i])
                c.conditional_gate(m, [g0, g1], n_data + i, i)
            for i in range(n_data - 1):
                c.cnot(i, i + 1)
            for i in range(n_data):
                c.rz(i, theta=p[layer, 5, i])
        return energy_from_state(c.state())

    def batch_energy(params):
        return jax.vmap(lambda s: energy_one(params, s))(status)

    optimizer = optax.adam(lr)
    opt_state0 = optimizer.init(params0)

    def train(params, opt_state):
        def step(carry, _):
            params, opt_state = carry

            def loss_fn(p):
                return jnp.mean(batch_energy(p))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            step, (params, opt_state), xs=None, length=max_steps
        )
        return params, losses

    params_f, energy_history = jax.jit(train)(params0, opt_state0)
    final_trajectory_energies = jax.jit(batch_energy)(params_f)

    return {
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "final_trajectory_energies": np.asarray(
            final_trajectory_energies, dtype=np.float64
        ),
    }
