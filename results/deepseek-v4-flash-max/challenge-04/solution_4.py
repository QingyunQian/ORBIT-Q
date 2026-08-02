import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import jax
import jax.numpy as jnp


def _make_label_tensor(n):
    size = 1 << n
    labels = np.zeros((n + 1, size), dtype=np.complex64)
    for i in range(n):
        for b in range(size):
            labels[i, b] = 1.0 if ((b >> i) & 1) == 0 else -1.0
    for b in range(size):
        labels[n, b] = 1.0 if b.bit_count() % 2 == 0 else -1.0
    return labels.reshape((n + 1,) + (2,) * n)


def run_solution(config):
    tc.set_backend("jax")
    n = config["n_qubits"]
    angle = config["entangler_angle"]
    max_steps = config["max_steps"]
    label_np = _make_label_tensor(n)

    def kraus(p01, p10):
        z = tc.num_to_tensor(0.0)
        k0 = tc.gates.Gate(tc.backend.stack([
            tc.backend.stack([tc.backend.sqrt(1.0 - p01), z]),
            tc.backend.stack([z, tc.backend.sqrt(1.0 - p10)]),
        ]))
        k1 = tc.gates.Gate(tc.backend.stack([
            tc.backend.stack([z, tc.backend.sqrt(p10)]),
            tc.backend.stack([z, z]),
        ]))
        k2 = tc.gates.Gate(tc.backend.stack([
            tc.backend.stack([z, z]),
            tc.backend.stack([tc.backend.sqrt(p01), z]),
        ]))
        return [k0, k1, k2]

    def prep(c, kind):
        if kind == 0:
            c.h(0)
            for i in range(1, n):
                c.cnot(0, i)
        elif kind == 1:
            for i in range(0, n - 1, 2):
                c.h(i)
                c.cnot(i, i + 1)
                c.x(i)
        elif kind == 2:
            pass
        else:
            for i in range(n):
                c.h(i)

    def noisy_circuit(kind, p01, p10):
        c = tc.DMCircuit2(n, split={"max_singular_values": 8})
        prep(c, kind)
        k = kraus(p01, p10)
        for i in range(0, n - 1, 2):
            c.rxx(i, i + 1, theta=angle)
        for i in range(n):
            c.general_kraus(k, i)
        for i in range(1, n - 1, 2):
            c.rxx(i, i + 1, theta=angle)
        for i in range(n):
            c.general_kraus(k, i)
        return c

    def observable_table(kind, p01, p10):
        c = noisy_circuit(kind, p01, p10)
        nodes, front = c._copy()
        diag_edges = []
        for i in range(n):
            copy_node = tn.CopyNode(3, 2, dtype=np.complex64)
            copy_node[0] ^ front[i]
            copy_node[1] ^ front[i + n]
            nodes.append(copy_node)
            diag_edges.append(copy_node[2])
        obs_node = tn.Node(tc.num_to_tensor(label_np))
        for i, edge in enumerate(diag_edges):
            edge ^ obs_node.get_edge(i + 1)
        nodes.append(obs_node)
        contracted = tc.cons.contractor(
            nodes, output_edge_order=[obs_node.get_edge(0)]
        )
        return tc.backend.real(contracted.tensor)

    def table_all(p01, p10):
        return jnp.stack([observable_table(k, p01, p10) for k in range(4)])

    table_jit = jax.jit(table_all)
    target = np.asarray(
        table_jit(
            np.float32(config["true_p01"]),
            np.float32(config["true_p10"]),
        ).block_until_ready(),
        dtype=np.float32,
    )
    target_jax = jnp.asarray(target)

    def loss(r):
        p01 = tc.backend.sigmoid(r[0])
        p10 = tc.backend.sigmoid(r[1])
        return jnp.mean((table_all(p01, p10) - target_jax) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss))
    r = np.array([
        np.log(config["initial_p01"] / (1.0 - config["initial_p01"])),
        np.log(config["initial_p10"] / (1.0 - config["initial_p10"])),
    ], dtype=np.float32)
    m = np.zeros(2, dtype=np.float32)
    v = np.zeros(2, dtype=np.float32)
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    lr = np.float32(config["learning_rate"])
    loss_history = np.empty(max_steps, dtype=np.float64)

    for step in range(1, max_steps + 1):
        value, grad = loss_and_grad(r)
        loss_history[step - 1] = np.asarray(value)
        grad = np.asarray(grad)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        r = r - lr * m_hat / (np.sqrt(v_hat) + eps)

    final_probabilities = np.asarray(
        1.0 / (1.0 + np.exp(-r)), dtype=np.float64
    )
    fitted_expectations = np.asarray(
        table_jit(
            np.float32(final_probabilities[0]),
            np.float32(final_probabilities[1]),
        ).block_until_ready(),
        dtype=np.float64,
    )
    return {
        "loss_history": loss_history,
        "final_probabilities": final_probabilities,
        "fitted_expectations": fitted_expectations,
    }
