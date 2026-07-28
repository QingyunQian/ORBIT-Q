import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy", preprocessing=True)

    n = int(config["n_qubits"])
    angle = float(config["entangler_angle"])

    def asymmetric_flip_kraus(probabilities):
        p01, p10 = probabilities
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        k0 = jnp.diag(jnp.stack((jnp.sqrt(1.0 - p01), jnp.sqrt(1.0 - p10))))
        k1 = jnp.stack(
            (jnp.stack((zero, jnp.sqrt(p10))), jnp.stack((zero, zero)))
        )
        k2 = jnp.stack(
            (jnp.stack((zero, zero)), jnp.stack((jnp.sqrt(p01), zero)))
        )
        return (k0, k1, k2)

    def probe(kind, probabilities):
        circuit = tc.DMCircuit(n)
        if kind == 0:  # GHZ
            circuit.h(0)
            for q in range(1, n):
                circuit.cnot(0, q)
        elif kind == 1:  # (|01> + |10>)/sqrt(2) on every pair
            for q in range(0, n, 2):
                circuit.h(q)
                circuit.x(q + 1)
                circuit.cnot(q, q + 1)
        elif kind == 3:  # |+>^n
            for q in range(n):
                circuit.h(q)

        kraus = asymmetric_flip_kraus(probabilities)
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                circuit.rxx(q, q + 1, theta=angle)
                circuit.general_kraus(kraus, q)
                circuit.general_kraus(kraus, q + 1)

        values = [
            jnp.real(circuit.expectation_ps(z=[q], reuse=False)) for q in range(n)
        ]
        values.append(
            jnp.real(circuit.expectation_ps(z=list(range(n)), reuse=False))
        )
        return jnp.stack(values)

    def observable_table(probabilities):
        return jnp.stack([probe(kind, probabilities) for kind in range(4)])

    compiled_table = jax.jit(observable_table)
    true_probabilities = jnp.asarray(
        [config["true_p01"], config["true_p10"]], dtype=jnp.float64
    )
    target = compiled_table(true_probabilities)
    jax.block_until_ready(target)

    initial_probabilities = jnp.asarray(
        [config["initial_p01"], config["initial_p10"]], dtype=jnp.float64
    )
    raw_initial = jnp.log(initial_probabilities / (1.0 - initial_probabilities))
    optimizer = optax.adam(float(config["learning_rate"]))

    def loss(raw):
        residual = observable_table(jax.nn.sigmoid(raw)) - target
        return jnp.mean(residual * residual)

    def train(raw):
        state = optimizer.init(raw)

        def update(carry, _):
            parameters, opt_state = carry
            value, gradient = jax.value_and_grad(loss)(parameters)
            changes, opt_state = optimizer.update(gradient, opt_state, parameters)
            parameters = optax.apply_updates(parameters, changes)
            return (parameters, opt_state), value

        return jax.lax.scan(
            update, (raw, state), xs=None, length=int(config["max_steps"])
        )

    (raw_final, _), loss_history = jax.jit(train)(raw_initial)
    final_probabilities = jax.nn.sigmoid(raw_final)
    fitted_expectations = compiled_table(final_probabilities)
    jax.block_until_ready(fitted_expectations)

    return {
        "loss_history": np.asarray(loss_history),
        "final_probabilities": np.asarray(final_probabilities),
        "fitted_expectations": np.asarray(fitted_expectations),
    }
