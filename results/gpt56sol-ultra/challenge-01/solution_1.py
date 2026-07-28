import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
import tensornetwork as tn


def run_solution(config):
    n = int(config["n_qubits"])
    nlayers = int(config["n_layers"])
    field = float(config["field"])

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy", preprocessing=True)
    K = tc.backend

    # Convert quimb's labelled tensors to TensorCircuit's (left, physical, right)
    # convention. This is only input interoperability; all evolution is below.
    qmps = config["dmrg_state"].copy()
    qmps.permute_arrays("lpr")
    arrays = [np.asarray(a) for a in qmps.arrays]
    if arrays[0].ndim == 2:
        arrays[0] = arrays[0][None, :, :]
    if arrays[-1].ndim == 2:
        arrays[-1] = arrays[-1][:, :, None]
    arrays = [K.cast(K.convert_to_tensor(a), "complex64") for a in arrays]

    initial = tc.MPSCircuit(n, tensors=arrays)
    initial.normalize()
    base_tensors = tuple(initial.get_tensors())

    # Bond-dimension-three TensorCircuit MPO for
    # -sum_i Z_i Z_{i+1} - field * sum_i X_i.
    eye = np.eye(2, dtype=np.complex64)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    z = np.diag([1, -1]).astype(np.complex64)
    zero = np.zeros((2, 2), dtype=np.complex64)
    bulk = np.array(
        [[eye, zero, zero], [z, zero, zero], [-field * x, -z, eye]]
    )
    mpo_arrays = [
        np.array([[-field * x, -z, eye]]),
        *([bulk] * (n - 2)),
        np.array([[eye], [z], [-field * x]]),
    ]
    mpo_arrays = [K.convert_to_tensor(a) for a in mpo_arrays]
    hamiltonian = tc.quantum.tn2qop(tn.FiniteMPO(mpo_arrays))

    def fused(gates, dim):
        matrix = K.reshape(gates[0].tensor, (dim, dim))
        for gate in gates[1:]:
            matrix = K.matmul(K.reshape(gate.tensor, (dim, dim)), matrix)
        rank = 2 if dim == 2 else 4
        return tc.Gate(K.reshape(matrix, [2] * rank))

    # A small native MPS cap keeps each compiled layer compact. Since the
    # refinement starts near identity, retaining the input DMRG bond dimension
    # captures the correction while avoiding a depth-induced rank explosion.
    max_bond = max(2, int(config["dmrg_chi"]))

    def layer_fn(tensors, rotations, interactions, parity):
        circuit = tc.MPSCircuit(
            n,
            tensors=tensors,
            center_position=0,
            split={"max_singular_values": max_bond},
        )
        for i in range(n):
            gate = fused(
                (
                    tc.gates.rz_gate(theta=rotations[i, 0]),
                    tc.gates.ry_gate(theta=rotations[i, 1]),
                    tc.gates.rz_gate(theta=rotations[i, 2]),
                ),
                2,
            )
            circuit.apply(gate, i)
        bonds = list(range(parity, n - 1, 2))
        for b, i in reversed(list(enumerate(bonds))):
            gate = fused(
                (
                    tc.gates.rxx_gate(theta=2 * interactions[b, 0]),
                    tc.gates.ryy_gate(theta=2 * interactions[b, 1]),
                    tc.gates.rzz_gate(theta=2 * interactions[b, 2]),
                ),
                4,
            )
            circuit.apply(gate, i, i + 1)
        circuit.position(0)
        circuit.normalize()
        return tuple(circuit.get_tensors())

    compiled_layer = jax.jit(layer_fn, static_argnums=3)

    @jax.jit
    def measure_energy(tensors):
        circuit = tc.Circuit(n, tensors=tensors)
        return tc.templates.measurements.operator_expectation(circuit, hamiltonian)

    def objective(params):
        tensors = base_tensors
        for layer in range(nlayers):
            tensors = compiled_layer(
                tensors, params[0][layer], params[layer + 1], layer & 1
            )
        return measure_energy(tensors)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    params = [
        1.0e-3
        * jax.random.normal(subkey, (nlayers, n, 3), dtype=jnp.float32)
    ]
    for layer in range(nlayers):
        key, subkey = jax.random.split(key)
        nbonds = len(range(layer & 1, n - 1, 2))
        params.append(
            1.0e-3
            * jax.random.normal(subkey, (nbonds, 3), dtype=jnp.float32)
        )
    params = tuple(params)

    optimizer = optax.adam(float(config["learning_rate"]))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, state, grads):
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    value_and_grad = jax.value_and_grad(objective)
    history = np.empty(int(config["max_steps"]), dtype=np.float64)
    for step in range(history.size):
        energy, grads = value_and_grad(params)
        history[step] = float(energy)
        params, opt_state = update(params, opt_state, grads)

    return {"energy_history": history}
