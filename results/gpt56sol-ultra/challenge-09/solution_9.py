import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import optax
import tensorcircuit as tc


def _causal_cone(gate_tape, pauli_ops):
    """Return the exact backwards unitary cone of one Pauli string."""
    active = {qubit for _, qubit in pauli_ops}
    kept = []
    for gate in reversed(gate_tape):
        wires = gate[1:-1]
        if active.intersection(wires):
            kept.append(gate)
            active.update(wires)
    kept.reverse()
    qubit_map = {qubit: i for i, qubit in enumerate(sorted(active))}
    return tuple(kept), qubit_map


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy", preprocessing=True)

    tape = tuple(config["gate_tape"])
    terms = tuple(config["pauli_terms"])
    cones = []
    used_parameters = set()
    for coefficient, pauli_ops in terms:
        gates, qubit_map = _causal_cone(tape, pauli_ops)
        used_parameters.update(gate[-1] for gate in gates)
        cones.append((coefficient, tuple(pauli_ops), gates, qubit_map))

    parameter_ids = np.asarray(sorted(used_parameters), dtype=np.int32)
    parameter_map = {p: i for i, p in enumerate(parameter_ids.tolist())}

    def local_objective(theta):
        value = 0.0
        for coefficient, pauli_ops, gates, qubit_map in cones:
            circuit = tc.Circuit(len(qubit_map))
            for qubit in range(len(qubit_map)):
                circuit.h(qubit)
            for gate in gates:
                name, wires, parameter = gate[0], gate[1:-1], gate[-1]
                getattr(circuit, name)(
                    *(qubit_map[q] for q in wires),
                    theta=theta[parameter_map[parameter]],
                )
            operators = tuple(
                (getattr(tc.gates, axis)(), [qubit_map[q]])
                for axis, q in pauli_ops
            )
            expectation = circuit.expectation(*operators, reuse=False)
            value = value + coefficient * tc.backend.real(expectation)
        return value

    value_and_grad = jax.vmap(jax.value_and_grad(lambda x: -local_objective(x)))
    optimizer = optax.adam(float(config["learning_rate"]))
    steps = int(config["max_steps"])

    def optimize(initial_parameters):
        optimizer_state = optimizer.init(initial_parameters)

        def update(carry, _):
            parameters, state = carry
            losses, gradients = value_and_grad(parameters)
            updates, state = optimizer.update(gradients, state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, state), -losses

        carry, history = jax.lax.scan(
            update, (initial_parameters, optimizer_state), None, length=steps
        )
        return history, carry[0]

    seed = int(config["seed"]) + 100000
    scale = float(config["initial_parameter_scale"])
    parameter_count = int(config["parameter_count"])
    initial = np.stack(
        [
            np.random.default_rng(seed + restart)
            .normal(0.0, scale, size=parameter_count)[parameter_ids]
            for restart in range(int(config["n_restarts"]))
        ]
    ).astype(np.float32)

    history, final_parameters = jax.jit(optimize)(jax.numpy.asarray(initial))
    jax.block_until_ready(final_parameters)
    return {"observable_history": np.asarray(history).T}
