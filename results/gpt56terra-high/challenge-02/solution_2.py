"""TensorCircuit-JAX implementation of the entanglement-profile VQE."""
import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit import quantum


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    blocks = layers // 2
    targets = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    anisotropy = float(config["zz_anisotropy"])
    field = float(config["staggered_field"])
    keep = tuple(range(int(config["subsystem_size"])))
    even = tuple(range(0, n - 1, 2))
    odd = tuple(range(1, n - 1, 2))
    per_block = 4 * n + 3 * (len(even) + len(odd))
    structures, weights = [], []
    for q in range(n - 1):
        for pauli, coefficient in ((1, 1.0), (2, 1.0), (3, anisotropy)):
            term = [0] * n
            term[q] = term[q + 1] = pauli
            structures.append(term)
            weights.append(coefficient)
    for q in range(n):
        term = [0] * n
        term[q] = 3
        structures.append(term)
        weights.append(field * (-1.0 if q % 2 else 1.0))
    hamiltonian_mvp = quantum.PauliStringSum2MVP(structures, weights)

    def sublayer(c, p, bonds):
        for q in range(n):
            c.ry(q, theta=p[2 * q])
            c.rz(q, theta=p[2 * q + 1])
        k = 2 * n
        for edge in bonds:
            # TensorCircuit's rXX/rYY/rZZ use exp(-i theta P / 2).
            c.rxx(edge, edge + 1, theta=2.0 * p[k])
            c.ryy(edge, edge + 1, theta=2.0 * p[k + 1])
            c.rzz(edge, edge + 1, theta=2.0 * p[k + 2])
            k += 3
        return c

    def evolve_sublayer(state, p, bonds):
        c = tc.Circuit(n, inputs=state)
        c = sublayer(c, p, bonds)
        return c.state()

    evolve_even = jax.jit(lambda state, p: evolve_sublayer(state, p, even))
    evolve_odd = jax.jit(lambda state, p: evolve_sublayer(state, p, odd))
    initial_circuit = tc.Circuit(n)
    for q in range(1, n, 2):
        initial_circuit.x(q)
    initial_state = initial_circuit.state()

    def objective(p):
        entropies = []
        final_state = initial_state
        for b in range(blocks):
            base = b * per_block
            final_state = evolve_even(final_state, p[base : base + 2 * n + 3 * len(even)])
            start = base + 2 * n + 3 * len(even)
            final_state = evolve_odd(final_state, p[start : (b + 1) * per_block])
            rho = quantum.reduced_density_matrix(final_state, subsystem_to_keep=keep)
            entropies.append(quantum.renyi_entropy(rho, k=2))
        energy = jnp.real(jnp.vdot(final_state, hamiltonian_mvp(final_state)))
        entropies = jnp.stack(entropies)
        mse = jnp.mean((entropies - targets) ** 2)
        return energy / n + float(config["entropy_weight"]) * mse, (energy / n, mse, entropies)

    value_grad = jax.value_and_grad(objective, has_aux=True)
    params = jnp.asarray(np.random.default_rng(1234).normal(0.0, 0.02, blocks * per_block), dtype=jnp.float32)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    steps = int(config["max_steps"])
    energy_history = np.empty(steps, dtype=np.float32)
    loss_history = np.empty(steps, dtype=np.float32)
    mse_history = np.empty(steps, dtype=np.float32)
    entropy_history = np.empty((steps, blocks), dtype=np.float32)
    lr = float(config["learning_rate"])
    for step in range(steps):
        (loss, (energy, mse, entropies)), grad = value_grad(params)
        loss_history[step] = float(loss)
        energy_history[step] = float(energy)
        mse_history[step] = float(mse)
        entropy_history[step] = np.asarray(entropies)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        params = params - lr * (m / (1.0 - 0.9 ** (step + 1))) / (jnp.sqrt(v / (1.0 - 0.999 ** (step + 1))) + 1e-8)
    return {"energy_density_history": energy_history, "loss_history": loss_history,
            "entropy_mse_history": mse_history, "entropy_history": entropy_history}
