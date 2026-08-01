import gc
import os
import pickle
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc


def _cone(tape, term):
    active = {q for _, q in term[1]}
    kept = []
    for gate in reversed(tape):
        if len(gate) == 3:
            if gate[1] in active:
                kept.append(gate)
        elif gate[1] in active or gate[2] in active:
            kept.append(gate)
            active.update((gate[1], gate[2]))
    qubits = sorted(active)
    qmap = {q: i for i, q in enumerate(qubits)}
    selected = sorted(g[-1] for g in kept)
    pmap = {p: i for i, p in enumerate(selected)}
    local = []
    for gate in reversed(kept):
        if len(gate) == 3:
            local.append((gate[0], qmap[gate[1]], pmap[gate[2]]))
        else:
            local.append(
                (gate[0], qmap[gate[1]], qmap[gate[2]], pmap[gate[3]])
            )
    return qubits, qmap, local, selected


def _objective(spec, term):
    qubits, qmap, local, _ = spec
    op0 = getattr(tc.gates, term[1][0][0])()
    op1 = getattr(tc.gates, term[1][1][0])()
    i0, i1 = qmap[term[1][0][1]], qmap[term[1][1][1]]

    def objective(params):
        circuit = tc.MPSCircuit(len(qubits))
        for q in range(len(qubits)):
            circuit.h(q)
        for gate in local:
            if len(gate) == 3:
                getattr(circuit, gate[0])(gate[1], theta=params[gate[2]])
            else:
                getattr(circuit, gate[0])(
                    gate[1], gate[2], theta=params[gate[3]]
                )
        value = circuit.expectation((op0, [i0]), (op1, [i1]))
        return jnp.real(value)

    return objective


def _adam(objective, initial, steps, rate, objective_weight):
    update = jax.jit(jax.value_and_grad(objective))
    history = np.empty((initial.shape[0], steps), dtype=np.float32)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for restart in range(initial.shape[0]):
        params = jnp.asarray(initial[restart])
        first = jnp.zeros_like(params)
        second = jnp.zeros_like(params)
        for step in range(steps):
            value, grad = update(params)
            history[restart, step] = float(value)
            grad = objective_weight * grad
            first = b1 * first + (1.0 - b1) * grad
            second = b2 * second + (1.0 - b2) * grad * grad
            bias1 = 1.0 - b1 ** (step + 1)
            bias2 = 1.0 - b2 ** (step + 1)
            params = params + rate * (first / bias1) / (
                jnp.sqrt(second / bias2) + eps
            )
    del update
    jax.clear_caches()
    gc.collect()
    return history


def _stdio_worker():
    spec, term, initial, steps, rate = pickle.load(sys.stdin.buffer)
    tc.set_backend("jax")
    history = _adam(_objective(spec, term), initial, steps, rate, term[0])
    pickle.dump(history, sys.stdout.buffer)


def _isolated_adam(spec, term, initial, steps, rate):
    process = subprocess.Popen(
        [sys.executable, "-c", "import solution_9; solution_9._stdio_worker()"],
        cwd=os.path.dirname(__file__),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = pickle.dumps((spec, term, initial, steps, rate), protocol=5)
    output, error = process.communicate(payload)
    if process.returncode:
        raise RuntimeError(error.decode(errors="replace"))
    return pickle.loads(output)


def run_solution(config):
    tc.set_backend("jax")
    tape = config["gate_tape"]
    terms = config["pauli_terms"]
    specs = [_cone(tape, term) for term in terms]
    parameter_count = config.get(
        "parameter_count", max(g[-1] for g in tape) + 1
    )
    restarts, steps = config["n_restarts"], config["max_steps"]
    initial = [
        np.empty((restarts, len(spec[3])), dtype=np.float32) for spec in specs
    ]
    for restart in range(restarts):
        rng = np.random.default_rng(config["seed"] + 100000 + restart)
        full = rng.normal(
            0.0, config["initial_parameter_scale"], size=parameter_count
        )
        for values, spec in zip(initial, specs):
            values[restart] = full[spec[3]]
    histories = []
    for spec, term, values in zip(specs, terms, initial):
        histories.append(
            _isolated_adam(
                spec,
                term,
                values,
                steps,
                config["learning_rate"],
            )
        )
    return {"observable_history": sum(term[0] * h for term, h in zip(terms, histories))}
