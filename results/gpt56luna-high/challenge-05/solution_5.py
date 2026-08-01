import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    K = tc.backend
    n = int(config["n_qubits"])
    nl = int(config["n_layers"])
    field = float(config["transverse_field"])
    steps = int(config["max_steps"])
    rate = float(config["learning_rate"])
    plus = jnp.ones(2**n, dtype=jnp.complex64) / jnp.sqrt(2.0**n)
    x = tc.gates.x().tensor
    ident = tc.gates.i().tensor
    zz = jnp.kron(tc.gates.z().tensor, tc.gates.z().tensor)

    structures, weights = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = s[i + 1] = 3
        structures.append(s)
        weights.append(-1.0)
    for i in range(n):
        s = [0] * n
        s[i] = 1
        structures.append(s)
        weights.append(-field)
    h_mvp = tc.quantum.PauliStringSum2MVP(structures, weights)

    def loss(params):
        c = tc.Circuit(n, inputs=plus)
        for layer in range(nl):
            one = tc.gates.exponential_gate(x, 1j * params[0, layer]).tensor
            two = tc.gates.exponential_gate(zz, 1j * params[1, layer]).tensor
            for q in range(n):
                c.any(q, unitary=one)
            for q in range(layer % 2, n - 1, 2):
                c.any(q, q + 1, unitary=two)
            state = c.state()
            norm = K.sqrt(K.real(K.sum(K.conj(state) * state)))
            c.any(0, unitary=ident / norm)
        state = c.state()
        return K.real(K.sum(K.conj(state) * h_mvp(state))) / n

    value_grad = jax.value_and_grad(loss)
    params = jnp.full((2, nl), float(config["initial_filter_strength"]), dtype=jnp.float32)
    first = jnp.zeros_like(params)
    second = jnp.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    def adam_step(carry, _):
        params, first, second, step = carry
        energy, grad = value_grad(params)
        step = step + 1
        first = beta1 * first + (1.0 - beta1) * grad
        second = beta2 * second + (1.0 - beta2) * grad * grad
        mhat = first / (1.0 - beta1**step)
        vhat = second / (1.0 - beta2**step)
        params = params - rate * mhat / (jnp.sqrt(vhat) + eps)
        return (params, first, second, step), energy

    chunk_size = 10
    run_chunk = jax.jit(lambda carry: jax.lax.scan(
        adam_step, carry, jnp.arange(chunk_size)
    ))
    carry = (params, first, second, jnp.array(0, dtype=jnp.int32))
    history = []
    for _ in range(steps // chunk_size):
        carry, energies = run_chunk(carry)
        history.extend(np.asarray(jax.device_get(energies)).tolist())
    params, first, second, step = carry
    for _ in range(steps % chunk_size):
        energy, grad = value_grad(params)
        step = step + 1
        history.append(float(energy))
        first = beta1 * first + (1.0 - beta1) * grad
        second = beta2 * second + (1.0 - beta2) * grad * grad
        mhat = first / (1.0 - beta1**step)
        vhat = second / (1.0 - beta2**step)
        params = params - rate * mhat / (jnp.sqrt(vhat) + eps)

    params = np.asarray(jax.device_get(params))
    return {
        "final_a": params[0].reshape(5, 2),
        "final_b": params[1].reshape(5, 2),
        "energy_density_history": np.asarray(history, dtype=np.float32),
    }
