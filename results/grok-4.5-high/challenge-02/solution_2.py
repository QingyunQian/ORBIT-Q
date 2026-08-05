import numpy as np
import jax
import jax.numpy as jnp
import optax
import scipy.sparse as sp
import tensorcircuit as tc


def _build_hamiltonian(n, zz_anisotropy, staggered_field):
    eye = sp.eye(2, format="csr", dtype=np.complex64)
    x = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
    y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))
    z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))

    def kron_all(ops):
        out = ops[0]
        for op in ops[1:]:
            out = sp.kron(out, op, format="csr")
        return out

    ham = sp.csr_matrix((2**n, 2**n), dtype=np.complex64)
    for i in range(n - 1):
        for left, right, weight in ((x, x, 1.0), (y, y, 1.0), (z, z, zz_anisotropy)):
            ops = [eye] * n
            ops[i] = left
            ops[i + 1] = right
            ham = ham + weight * kron_all(ops)
    for i in range(n):
        ops = [eye] * n
        ops[i] = z
        ham = ham + staggered_field * ((-1) ** i) * kron_all(ops)
    return jnp.asarray(ham.toarray())


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = int(config["n_qubits"])
    zz_anisotropy = float(config["zz_anisotropy"])
    staggered_field = float(config["staggered_field"])
    n_layers = int(config["n_layers"])
    subsystem_size = int(config["subsystem_size"])
    target_entropies = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    entropy_weight = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    n_blocks = n_layers // 2
    even_bonds = [(i, i + 1) for i in range(0, n, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]
    even_size = 2 * n + 3 * len(even_bonds)
    odd_size = 2 * n + 3 * len(odd_bonds)
    params_per_block = even_size + odd_size

    xx = tc.gates._xx_matrix
    yy = tc.gates._yy_matrix
    zz = tc.gates._zz_matrix
    ham = _build_hamiltonian(n, zz_anisotropy, staggered_field)

    bitstring = "".join("1" if (q % 2 == 1) else "0" for q in range(n))
    psi0 = jnp.zeros((2**n,), dtype=jnp.complex64).at[int(bitstring, 2)].set(1.0)
    traced = list(range(subsystem_size, n))

    def _sublayer(state, params, bonds):
        circuit = tc.Circuit(n, inputs=state)
        idx = 0
        for q in range(n):
            circuit.ry(q, theta=params[idx])
            idx += 1
            circuit.rz(q, theta=params[idx])
            idx += 1
        for i, j in bonds:
            circuit.exp1(i, j, theta=params[idx], unitary=xx)
            idx += 1
            circuit.exp1(i, j, theta=params[idx], unitary=yy)
            idx += 1
            circuit.exp1(i, j, theta=params[idx], unitary=zz)
            idx += 1
        return circuit.state()

    def _block(state, block_params):
        state = _sublayer(state, block_params[:even_size], even_bonds)
        state = _sublayer(state, block_params[even_size:], odd_bonds)
        rho = tc.quantum.reduced_density_matrix(state, traced)
        s2 = jnp.real(tc.quantum.renyi_entropy(rho, 2))
        return state, s2

    def loss_and_metrics(params):
        final_state, entropies = jax.lax.scan(_block, psi0, params)
        energy = jnp.real(jnp.vdot(final_state, ham @ final_state))
        energy_density = energy / n
        entropy_mse = jnp.mean((entropies - target_entropies) ** 2)
        loss = energy_density + entropy_weight * entropy_mse
        return loss, (energy_density, entropy_mse, entropies)

    key = jax.random.PRNGKey(0)
    params = 0.02 * jax.random.normal(key, (n_blocks, params_per_block), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    # Warmup compile.
    params, opt_state, _, _ = train_step(params, opt_state)
    params = 0.02 * jax.random.normal(key, (n_blocks, params_per_block), dtype=jnp.float32)
    opt_state = optimizer.init(params)

    energy_density_history = np.empty(max_steps, dtype=np.float64)
    loss_history = np.empty(max_steps, dtype=np.float64)
    entropy_mse_history = np.empty(max_steps, dtype=np.float64)
    entropy_history = np.empty((max_steps, target_entropies.shape[0]), dtype=np.float64)

    for step in range(max_steps):
        params, opt_state, loss, metrics = train_step(params, opt_state)
        energy_density, entropy_mse, entropies = metrics
        energy_density_history[step] = float(energy_density)
        loss_history[step] = float(loss)
        entropy_mse_history[step] = float(entropy_mse)
        entropy_history[step] = np.asarray(entropies, dtype=np.float64)

    return {
        "energy_density_history": energy_density_history,
        "loss_history": loss_history,
        "entropy_mse_history": entropy_mse_history,
        "entropy_history": entropy_history,
    }
