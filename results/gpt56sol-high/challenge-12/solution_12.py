def run_solution(config):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    steps = int(config["max_steps"])
    target = config["dmrg_state"]

    def target_tensors():
        arrays = []
        for i in range(n):
            tensor = target[i]
            inds = list(tensor.inds)
            wanted = []
            if i:
                wanted.append(target.bond(i - 1, i))
            wanted.append(target.site_ind(i))
            if i + 1 < n:
                wanted.append(target.bond(i, i + 1))
            a = np.asarray(tensor.data).transpose([inds.index(x) for x in wanted])
            if i == 0:
                a = a[None, :, :]
            elif i == n - 1:
                a = a[:, :, None]
            arrays.append(jnp.asarray(a, dtype=jnp.complex64))
        return arrays

    tensors = target_tensors()
    bonds = []
    for layer in range(layers):
        bs = list(range(layer & 1, n - 1, 2))
        if layer & 1:
            bs.reverse()
        bonds.extend(bs)
    count = 15 * len(bonds)

    def overlap(params):
        ket = tc.MPSCircuit(n)
        for i in range(1, n, 2):
            ket.x(i)
        p = params.reshape((-1, 15))
        for k, i in enumerate(bonds):
            center = i if (k and i < bonds[k - 1]) else i + 1
            gate = tc.gates.su4_gate(p[k])
            ket.apply_adjacent_double_gate(gate, i, i + 1, center_position=center)
        bra = tc.MPSCircuit(n, tensors=tensors, center_position=0)
        return ket.proj_with_mps(bra)

    def loss_fn(params):
        z = overlap(params)
        fidelity = jnp.real(z * jnp.conj(z))
        return 1.0 - fidelity, fidelity

    key = jax.random.PRNGKey(int(config["seed"]))
    params = (
        float(config["initial_parameter_scale"])
        * jax.random.normal(key, (count,), dtype=jnp.float32)
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    state = optimizer.init(params)
    value_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def update(carry, unused):
        p, s = carry
        (loss, fidelity), grad = value_grad(p)
        updates, s = optimizer.update(grad, s, p)
        return (optax.apply_updates(p, updates), s), (loss, fidelity)

    train = jax.jit(lambda p, s: jax.lax.scan(update, (p, s), None, steps))
    (params, state), (losses, fidelities) = train(params, state)
    final_z = jax.jit(overlap)(params)
    losses, fidelities, params, final_z = jax.device_get(
        (losses, fidelities, params, final_z)
    )
    losses = np.asarray(losses, dtype=float)
    return {
        "loss_history": losses,
        "fidelity_history": np.asarray(fidelities, dtype=float),
        "final_parameters": np.asarray(params, dtype=float),
        "final_overlap_phase": float(np.angle(final_z)),
    }
