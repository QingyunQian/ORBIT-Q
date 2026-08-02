import numpy as np

import tensorcircuit as tc
import jax
import jax.numpy as jnp
from tensorcircuit.quantum import PauliStringSum2MVP


def _make_energy(n_qubits, initial_ones, selected_qubits, n_layers, mvp):
    def energy(params):
        c = tc.Circuit(n_qubits)
        for q in initial_ones:
            c.x(q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                c.rx(q, theta=params[layer, q, 0])
                c.rz(q, theta=params[layer, q, 1])
                c.ry(q, theta=params[layer, q, 2])
            c.cmz(*selected_qubits)

        psi = c.state()
        hpsi = mvp(psi)
        return jnp.real(jnp.vdot(psi, hpsi)) / n_qubits

    return energy


def _make_hamiltonian_mvp(n_qubits, zz_strength, x_strength):
    structures = []
    weights = []
    for q in range(n_qubits - 1):
        row = [0] * n_qubits
        row[q] = 3
        row[q + 1] = 3
        structures.append(row)
        weights.append(-zz_strength)
    for q in range(n_qubits):
        row = [0] * n_qubits
        row[q] = 1
        structures.append(row)
        weights.append(-x_strength)
    return PauliStringSum2MVP(structures, weights)


def run_solution(config):
    tc.set_backend("jax")

    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    max_steps = config["max_steps"]
    lr = config["learning_rate"]
    scale = config["initial_parameter_scale"]

    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(
        rng.normal(0.0, scale, size=(n_layers, n_qubits, 3))
    )

    mvp = _make_hamiltonian_mvp(
        n_qubits, config["zz_strength"], config["x_strength"]
    )
    energy = _make_energy(
        n_qubits,
        config["initial_ones"],
        config["selected_qubits"],
        n_layers,
        mvp,
    )
    energy_and_grad = jax.jit(jax.value_and_grad(energy))

    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    history = np.empty(max_steps)

    for step in range(max_steps):
        value, grads = energy_and_grad(params)
        history[step] = float(np.asarray(value))

        m = b1 * m + (1.0 - b1) * grads
        v = b2 * v + (1.0 - b2) * grads * grads
        mhat = m / (1.0 - b1 ** (step + 1))
        vhat = v / (1.0 - b2 ** (step + 1))
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

    return {
        "energy_history": history,
        "final_parameters": np.asarray(params),
    }
