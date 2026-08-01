import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tape = tuple(config["gate_tape"])

    # Work backwards from each measured Pauli string.  A two-qubit gate enters
    # the cone exactly when either of its wires is already in the support.
    cones = []
    for _, term in config["pauli_terms"]:
        live = {int(q) for _, q in term}
        selected = []
        for gate in reversed(tape):
            wires = tuple(int(q) for q in gate[1:-1])
            if any(q in live for q in wires):
                selected.append(gate)
                live.update(wires)
        qubits = tuple(sorted(live))
        qmap = {q: i for i, q in enumerate(qubits)}
        cones.append((tuple(reversed(selected)), qmap, term))

    active = sorted({int(g[-1]) for cone, _, _ in cones for g in cone})
    pmap = {p: i for i, p in enumerate(active)}

    def objective(theta):
        value = jnp.asarray(0.0, dtype=jnp.float32)
        for (weight, _), (cone, qmap, term) in zip(
            config["pauli_terms"], cones
        ):
            circuit = tc.Circuit(len(qmap))
            for q in range(len(qmap)):
                circuit.h(q)
            for gate in cone:
                name, parameter = gate[0], theta[pmap[int(gate[-1])]]
                wires = [qmap[int(q)] for q in gate[1:-1]]
                getattr(circuit, name)(*wires, theta=parameter)
            ops = tuple((getattr(tc.gates, axis)(), [qmap[int(q)]]) for axis, q in term)
            value = value + jnp.asarray(weight, jnp.float32) * jnp.real(
                circuit.expectation(*ops)
            )
        return value

    count = int(config["parameter_count"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"]) + 100000
    initial = np.empty((int(config["n_restarts"]), len(active)), np.float32)
    for restart in range(initial.shape[0]):
        full = np.random.default_rng(seed + restart).normal(0.0, scale, count)
        initial[restart] = full[active]
    theta0 = jnp.asarray(initial)

    optimizer = optax.adam(float(config["learning_rate"]))
    state0 = optimizer.init(theta0)
    batched_loss_grad = jax.vmap(jax.value_and_grad(lambda x: -objective(x)))

    def update(carry, _):
        theta, state = carry
        losses, gradients = batched_loss_grad(theta)
        updates, state = optimizer.update(gradients, state, theta)
        theta = optax.apply_updates(theta, updates)
        return (theta, state), -losses

    @jax.jit
    def optimize(theta, state):
        return jax.lax.scan(update, (theta, state), None, length=int(config["max_steps"]))

    (_, _), history = optimize(theta0, state0)
    history = np.asarray(jax.device_get(history)).T
    return {"observable_history": history}
