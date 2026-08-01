import numpy as np
import tensornetwork as tn
import tensorcircuit as tc


def run_solution(config):
    import jax
    import jax.numpy as jnp

    tc.set_backend("jax")
    tc.set_contractor("auto")
    n = int(config["n_qubits"])
    selected = tuple(config["selected_qubits"])
    initial = tuple(config["initial_ones"])
    layers = int(config["n_layers"])
    steps = int(config["max_steps"])
    zz = np.float32(config["zz_strength"])
    xstrength = np.float32(config["x_strength"])

    eye = np.eye(2, dtype=np.complex64)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    pauli_z = np.diag([1, -1]).astype(np.complex64)

    def hamiltonian_mpo():
        onsite = -xstrength * pauli_x
        start = -zz * pauli_z
        nodes = [tn.Node(jnp.asarray(np.stack([onsite, start, eye], axis=0)))]
        for _ in range(1, n - 1):
            a = np.zeros((3, 3, 2, 2), dtype=np.complex64)
            a[0, 0] = eye
            a[1, 0] = pauli_z
            a[2, 0] = onsite
            a[2, 1] = start
            a[2, 2] = eye
            nodes.append(tn.Node(jnp.asarray(a)))
        nodes.append(tn.Node(jnp.asarray(np.stack([eye, pauli_z, onsite], axis=0))))
        nodes[0][0] ^ nodes[1][0]
        for i in range(1, n - 1):
            nodes[i][1] ^ nodes[i + 1][0]
        return nodes

    def energy(params):
        circuit = tc.Circuit(n)
        for q in initial:
            circuit.x(q)
        for layer in range(layers):
            for q in range(n):
                circuit.rx(q, theta=params[layer, q, 0])
                circuit.rz(q, theta=params[layer, q, 1])
                circuit.ry(q, theta=params[layer, q, 2])
            circuit.cmz(*selected)

        ket, ket_edges = circuit._copy_state_tensor(reuse=True)
        bra, bra_edges = circuit._copy_state_tensor(conj=True, reuse=True)
        hnodes = hamiltonian_mpo()
        for q in range(n):
            out_axis = 1 if q in (0, n - 1) else 2
            in_axis = 2 if q in (0, n - 1) else 3
            hnodes[q][out_axis] ^ bra_edges[q]
            hnodes[q][in_axis] ^ ket_edges[q]
        return tc.backend.real(tc.cons.contractor(ket + bra + hnodes).tensor)

    rng = np.random.default_rng(int(config["seed"]))
    params = jnp.asarray(
        rng.normal(0.0, float(config["initial_parameter_scale"]), (layers, n, 3)),
        dtype=jnp.float32,
    )
    value_and_grad = jax.value_and_grad(energy)
    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    learning_rate = np.float32(config["learning_rate"])
    history = np.empty(steps, dtype=np.float32)
    for step in range(steps):
        value, gradient = value_and_grad(params)
        history[step] = float(value) / n
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        correction1 = 1.0 - beta1 ** (step + 1)
        correction2 = 1.0 - beta2 ** (step + 1)
        direction = (first_moment / correction1) / (
            jnp.sqrt(second_moment / correction2) + eps
        )
        params = params - learning_rate * direction
    return {"energy_history": history, "final_parameters": np.asarray(params)}
