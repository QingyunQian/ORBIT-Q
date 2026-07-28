import numpy as np


def _mps_arrays(psi, n):
    arrays = []
    for i in range(n):
        tensor = psi[i]
        order = []
        if i:
            order.append(tensor.inds.index(psi.bond(i - 1, i)))
        order.append(tensor.inds.index(psi.site_ind(i)))
        if i + 1 < n:
            order.append(tensor.inds.index(psi.bond(i, i + 1)))
        array = np.asarray(tensor.data).transpose(order)
        if i == 0:
            array = array[None, :, :]
        if i == n - 1:
            array = array[:, :, None]
        arrays.append(array)
    return arrays


def _tfim_mpo(n, field):
    import tensornetwork as tn
    import tensorcircuit as tc

    eye = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    tensors = []
    w = np.zeros((1, 3, 2, 2))
    w[0, 0], w[0, 1], w[0, 2] = -field * x, -z, eye
    tensors.append(w)
    for _ in range(1, n - 1):
        w = np.zeros((3, 3, 2, 2))
        w[0, 0], w[1, 0], w[2, 0] = eye, z, -field * x
        w[2, 1], w[2, 2] = -z, eye
        tensors.append(w)
    w = np.zeros((3, 1, 2, 2))
    w[0, 0], w[1, 0], w[2, 0] = eye, z, -field * x
    tensors.append(w)
    return tc.quantum.tn2qop(tn.FiniteMPO(tensors, backend="jax"))


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    chi = int(config["dmrg_chi"])
    steps = int(config["max_steps"])

    tc.set_backend("numpy")
    tc.set_dtype("complex64")
    initial = tc.MPSCircuit(n, tensors=_mps_arrays(config["dmrg_state"], n))
    tensors0 = tuple(np.asarray(t) for t in initial.get_tensors())

    tc.set_backend("jax")
    tc.set_contractor("greedy", preprocessing=True)
    mpo = _tfim_mpo(n, float(config["field"]))
    split = {"max_singular_values": chi}

    def circuit_layer(tensors, one, two, parity, center):
        circuit = tc.MPSCircuit(n, tensors=tensors, center_position=center)
        for q in range(n):
            rz1 = tc.backend.reshape2(tc.gates.rz(theta=one[q, 0]).tensor)
            ry = tc.backend.reshape2(tc.gates.ry(theta=one[q, 1]).tensor)
            rz2 = tc.backend.reshape2(tc.gates.rz(theta=one[q, 2]).tensor)
            circuit._mps.apply_one_site_gate(rz2 @ ry @ rz1, q)
        bonds = list(enumerate(range(parity, n - 1, 2)))
        if parity:
            bonds.reverse()
        for b, q in bonds:
            xx = tc.backend.reshape(
                tc.gates.rxx(theta=2 * two[b, 0]).tensor, (4, 4)
            )
            yy = tc.backend.reshape(
                tc.gates.ryy(theta=2 * two[b, 1]).tensor, (4, 4)
            )
            zz = tc.backend.reshape(
                tc.gates.rzz(theta=2 * two[b, 2]).tensor, (4, 4)
            )
            unitary = tc.backend.reshape(zz @ yy @ xx, (2, 2, 2, 2))
            gate = tc.gates.any_gate(unitary)
            new_center = q if parity else q + 1
            circuit.apply_adjacent_double_gate(
                gate, q, q + 1, center_position=new_center, split=split
            )
        return tuple(circuit.get_tensors())

    layer_functions = []
    center = 0
    for layer in range(layers):
        parity, input_center = layer % 2, center

        def layer_function(tensors, one, two, p=parity, c=input_center):
            return circuit_layer(tensors, one, two, p, c)

        layer_functions.append(jax.jit(layer_function))
        bonds = range(parity, n - 1, 2)
        center = min(bonds) if parity else max(bonds) + 1

    @jax.jit
    def state_energy(tensors):
        state = tc.MPSCircuit(n, tensors=tensors, center_position=center)
        state.normalize()
        circuit = tc.Circuit(n, tensors=state.get_tensors())
        return tc.templates.measurements.operator_expectation(circuit, mpo)

    def energy(params):
        tensors = tensors0
        for layer, apply_layer in enumerate(layer_functions):
            tensors = apply_layer(tensors, params[0][layer], params[1][layer])
        return state_energy(tensors)

    rng = np.random.default_rng(0)
    params = (
        jnp.asarray(rng.normal(0.0, 1e-3, (layers, n, 3)), dtype=jnp.float32),
        jnp.asarray(
            rng.normal(0.0, 1e-3, (layers, (n + 1) // 2, 3)),
            dtype=jnp.float32,
        ),
    )
    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(energy)

    @jax.jit
    def update(params, grads, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    history = np.empty(steps, dtype=np.float32)
    for step in range(steps):
        value, grads = value_and_grad(params)
        history[step] = np.asarray(value)
        params, opt_state = update(params, grads, opt_state)
    return {"energy_history": history}
