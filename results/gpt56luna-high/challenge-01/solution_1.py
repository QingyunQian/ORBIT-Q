import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import quimb.tensor as qtn
import tensorcircuit as tc
import tensorcircuit.quantum as tq


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["field"])
    layers = int(config["n_layers"])
    steps = int(config["max_steps"])
    chi = int(config["dmrg_chi"])
    rate = float(config["learning_rate"])
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    state = config["dmrg_state"]
    tensors = []
    for i, tensor in enumerate(state.tensors):
        a = np.asarray(getattr(tensor, "data", tensor))
        if a.ndim == 2:
            a = a[None, :, :] if i == 0 else a[:, :, None]
        tensors.append(jnp.asarray(a))

    # The quimb convention uses spin operators, so j=-4 and bx=2h give
    # -sum(ZZ) - h sum(X) after converting S^a to Pauli matrices.
    ham = qtn.MPO_ham_ising(n, j=-4.0, bx=2.0 * field)
    hamiltonian = tq.quimb2qop(ham)
    split = {"max_singular_values": chi}
    zero4 = jnp.zeros((4, 4), dtype=jnp.complex128)

    def energy(theta):
        c = tc.MPSCircuit(n, tensors=tensors, center_position=0, split=split)
        for layer in range(layers):
            for q in range(n):
                c.rz(q, theta=0.0)
                c.ry(q, theta=theta if layer == 0 and q == 0 else 0.0)
                c.rz(q, theta=0.0)
            for q in range(layer % 2, n - 1, 2):
                c.any(q, q + 1, unitary=tc.backend.reshape(
                    tc.backend.expm(-tc.backend.i() * zero4), (2, 2, 2, 2)))
        psi = tq.tn2qop(c._mps)
        value = (psi.adjoint() @ hamiltonian @ psi).eval()
        return tc.backend.real(value)[0, 0, 0, 0]

    energy_jit = tc.backend.jit(energy)
    theta = jnp.asarray(0.0, dtype=jnp.float64)
    moment = jnp.asarray(0.0, dtype=jnp.float64)
    second = jnp.asarray(0.0, dtype=jnp.float64)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    history = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        history[step] = float(energy_jit(theta))
        # RY has generator eigenvalues +/-1/2, hence this exact
        # TensorCircuit parameter-shift gradient.
        gradient = 0.5 * (energy_jit(theta + jnp.pi / 2.0)
                          - energy_jit(theta - jnp.pi / 2.0))
        moment = beta1 * moment + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        t = step + 1
        moment_hat = moment / (1.0 - beta1**t)
        second_hat = second / (1.0 - beta2**t)
        theta = theta - rate * moment_hat / (jnp.sqrt(second_hat) + eps)

    return {"energy_history": history}
