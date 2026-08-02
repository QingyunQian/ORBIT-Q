import numpy as np

import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def _extract_cone(gate_tape, pauli_string):
    marked = {qubit for _, qubit in pauli_string}
    kept_reversed = []
    for gate in reversed(gate_tape):
        if len(gate) == 3:
            if gate[1] in marked:
                kept_reversed.append(gate)
        else:
            qubit_a, qubit_b = gate[1], gate[2]
            if qubit_a in marked or qubit_b in marked:
                marked.update((qubit_a, qubit_b))
                kept_reversed.append(gate)
    qubits = sorted(marked)
    mapping = {qubit: i for i, qubit in enumerate(qubits)}
    kept = []
    for gate in reversed(kept_reversed):
        if len(gate) == 3:
            kept.append((gate[0], mapping[gate[1]], gate[2]))
        else:
            kept.append((gate[0], mapping[gate[1]], mapping[gate[2]], gate[3]))
    return qubits, kept


def run_solution(config):
    gate_tape = config["gate_tape"]
    pauli_terms = config["pauli_terms"]
    n_restarts = config["n_restarts"]
    max_steps = config["max_steps"]
    learning_rate = config["learning_rate"]
    initial_scale = config["initial_parameter_scale"]
    seed = config["seed"]

    cones = []
    needed = set()
    for coefficient, pauli_string in pauli_terms:
        qubits, cone_gates = _extract_cone(gate_tape, pauli_string)
        labels = [label for label, _ in pauli_string]
        measured = [qubits.index(qubit) for _, qubit in pauli_string]
        cones.append((coefficient, labels, measured, len(qubits), cone_gates))
        for gate in cone_gates:
            needed.add(gate[3] if len(gate) == 4 else gate[2])
    needed = sorted(needed)
    param_index = {parameter: i for i, parameter in enumerate(needed)}

    def objective(theta):
        total = 0.0
        for coefficient, labels, measured, qubit_count, cone_gates in cones:
            circuit = tc.Circuit(qubit_count)
            for qubit in range(qubit_count):
                circuit.H(qubit)
            for gate in cone_gates:
                if len(gate) == 3:
                    getattr(circuit, gate[0])(
                        gate[1], theta=theta[param_index[gate[2]]]
                    )
                else:
                    getattr(circuit, gate[0])(
                        gate[1], gate[2], theta=theta[param_index[gate[3]]]
                    )
            operators = tuple(
                (getattr(tc.gates, label)(), [qubit])
                for label, qubit in zip(labels, measured)
            )
            total = total + coefficient * jnp.real(
                circuit.expectation(*operators, reuse=False)
            )
        return -total

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    parameter_count = config.get("parameter_count", len(gate_tape))
    history = np.empty((n_restarts, max_steps), dtype=np.float64)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + 100000 + restart)
        full_params = rng.normal(0.0, initial_scale, parameter_count).astype(
            np.float32
        )
        # All other parameters have exactly zero gradient, so Adam leaves
        # them at their sampled values; this is the same full-vector update.
        theta = jnp.asarray(full_params[needed])
        moment1 = jnp.zeros_like(theta)
        moment2 = jnp.zeros_like(theta)

        for step in range(1, max_steps + 1):
            loss, gradient = value_and_grad(theta)
            history[restart, step - 1] = float(-loss)
            moment1 = beta1 * moment1 + (1.0 - beta1) * gradient
            moment2 = beta2 * moment2 + (1.0 - beta2) * gradient * gradient
            corrected1 = moment1 / (1.0 - beta1**step)
            corrected2 = moment2 / (1.0 - beta2**step)
            theta = theta - learning_rate * corrected1 / (
                jnp.sqrt(corrected2) + epsilon
            )

    return {"observable_history": history}
