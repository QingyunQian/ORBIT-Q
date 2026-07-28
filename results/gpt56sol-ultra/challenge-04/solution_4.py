import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    """Calibrate an asymmetric bit-flip Kraus channel from four noisy probes."""
    backend = tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    # TensorCircuit prepares all four pure probe inputs.
    probes = []
    circuit = tc.Circuit(n)
    circuit.h(0)
    for q in range(1, n):
        circuit.cnot(0, q)
    probes.append(circuit.state())

    circuit = tc.Circuit(n)
    for q in range(0, n, 2):
        circuit.x(q + 1)
        circuit.h(q)
        circuit.cnot(q, q + 1)
    probes.append(circuit.state())

    circuit = tc.Circuit(n)
    probes.append(circuit.state())

    circuit = tc.Circuit(n)
    for q in range(n):
        circuit.h(q)
    probes.append(circuit.state())
    probes = backend.stack(probes)

    def kraus_operators(probabilities):
        p01, p10 = probabilities
        zero = jnp.zeros_like(p01)
        k0 = jnp.stack(
            [
                jnp.stack([jnp.sqrt(1.0 - p01), zero]),
                jnp.stack([zero, jnp.sqrt(1.0 - p10)]),
            ]
        )
        k1 = jnp.stack(
            [jnp.stack([zero, jnp.sqrt(p10)]), jnp.stack([zero, zero])]
        )
        k2 = jnp.stack(
            [jnp.stack([zero, zero]), jnp.stack([jnp.sqrt(p01), zero])]
        )
        return [k0, k1, k2]

    def one_probe(probabilities, initial_state):
        circuit = tc.DMCircuit(n, inputs=initial_state)
        kraus = kraus_operators(probabilities)
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                circuit.rxx(q, q + 1, theta=theta)
                circuit.general_kraus(kraus, q)
                circuit.general_kraus(kraus, q + 1)

        values = [
            backend.real(circuit.expectation_ps(z=[q], reuse=False))
            for q in range(n)
        ]
        values.append(
            backend.real(circuit.expectation_ps(z=list(range(n)), reuse=False))
        )
        return backend.stack(values)

    expectation_table = jax.jit(jax.vmap(one_probe, in_axes=(None, 0)))
    true_probabilities = jnp.asarray(
        [config["true_p01"], config["true_p10"]], dtype=jnp.float32
    )
    target = expectation_table(true_probabilities, probes)

    initial_probabilities = jnp.asarray(
        [config["initial_p01"], config["initial_p10"]], dtype=jnp.float32
    )
    raw_initial = jnp.log(initial_probabilities / (1.0 - initial_probabilities))

    def loss_fn(raw):
        fitted = expectation_table(jax.nn.sigmoid(raw), probes)
        return jnp.mean((fitted - target) ** 2)

    value_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(learning_rate)

    def train(raw):
        optimizer_state = optimizer.init(raw)

        def update(carry, _):
            raw, optimizer_state = carry
            loss, gradient = value_and_grad(raw)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, raw
            )
            raw = optax.apply_updates(raw, updates)
            return (raw, optimizer_state), loss

        return jax.lax.scan(
            update, (raw, optimizer_state), xs=None, length=steps
        )

    (raw_final, _), loss_history = jax.jit(train)(raw_initial)
    final_probabilities = jax.nn.sigmoid(raw_final)
    fitted_expectations = expectation_table(final_probabilities, probes)

    return {
        "loss_history": np.asarray(loss_history),
        "final_probabilities": np.asarray(final_probabilities),
        "fitted_expectations": np.asarray(fitted_expectations),
    }
