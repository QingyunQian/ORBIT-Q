"""
Challenge Suite Problem 11: spin-1 Haldane-chain VQE.

Performance-optimized variant of the reference solution with an unchanged
protocol: identical parameter layout, seeded initialization, layer structure,
Adam schedule, energy density, and returned quantities. Restructured for
speed: (1) exact gate fusion - the per-site rz/ry/rz rotations are composed
into one 3x3 unitary and absorbed into the even-bond entanglers, so every
layer applies 11 two-qudit unitaries instead of 47 gate applications;
(2) all 9x9 entangler exponentials of a layer are built in one batched fixed
2^5 scaling-and-squaring diagonal Pade(3,3) pass (exactly unitary for
anti-Hermitian input, error below the complex64 noise floor here);
(3) the diagonal single-ion observable is evaluated with one precomputed
per-basis-state coefficient vector instead of 12 expectation contractions;
(4) the 500 Adam updates run in one jax.lax.scan and the post-training
readout block is jit-compiled.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

DIM = 3
SQRT2 = np.sqrt(2.0).astype(np.float32)

SX = K.convert_to_tensor(
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex64)
    / SQRT2
)
SY = K.convert_to_tensor(
    np.array(
        [[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]],
        dtype=np.complex64,
    )
    / SQRT2
)
SZ = K.convert_to_tensor(np.diag([1.0, 0.0, -1.0]).astype(np.complex64))
SZ2_DIAG = np.array([1.0, 0.0, 1.0], dtype=np.float32)
STRING_MIDDLE = K.convert_to_tensor(np.diag([-1.0, 1.0, -1.0]).astype(np.complex64))

DOT_BOND = K.kron(SX, SX) + K.kron(SY, SY) + K.kron(SZ, SZ)
DOT_BOND_SQUARED = DOT_BOND @ DOT_BOND
ZZ_BOND = K.kron(SZ, SZ)


def string_pairs(config):
    return tuple((i, config["n_sites"] - 1 - i) for i in range(3))


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    scale = config["initial_parameter_scale"]
    n_layers, n_sites = config["n_layers"], config["n_sites"]
    shapes = {
        "single_rz1": (n_layers, n_sites),
        "single_ry": (n_layers, n_sites),
        "single_rz2": (n_layers, n_sites),
        "even_theta": (n_layers, n_sites // 2),
        "even_phi": (n_layers, n_sites // 2),
        "odd_theta": (n_layers, (n_sites - 1) // 2),
        "odd_phi": (n_layers, (n_sites - 1) // 2),
    }
    return {
        name: K.convert_to_tensor(
            rng.normal(scale=scale, size=shape).astype(np.float32)
        )
        for name, shape in shapes.items()
    }


def initial_state(config):
    neel = np.zeros(DIM ** config["n_sites"], dtype=np.complex64)
    index = 0
    for i in range(config["n_sites"]):
        index = index * DIM + (0 if i % 2 == 0 else 2)
    neel[index] = 1.0
    return K.convert_to_tensor(neel)


def basis_digit_table(n_sites):
    digits = np.zeros((DIM**n_sites, n_sites), dtype=np.int8)
    values = np.arange(DIM**n_sites, dtype=np.int64)
    for site in range(n_sites - 1, -1, -1):
        digits[:, site] = values % DIM
        values //= DIM
    return digits


def rz_batch(theta):
    theta = theta.astype(jnp.complex64)
    zero, one = jnp.zeros_like(theta), jnp.ones_like(theta)
    rows = [
        jnp.stack([jnp.exp(-1j * theta), zero, zero], axis=-1),
        jnp.stack([zero, one, zero], axis=-1),
        jnp.stack([zero, zero, jnp.exp(1j * theta)], axis=-1),
    ]
    return jnp.stack(rows, axis=-2)


def ry_batch(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    rows = [
        jnp.stack([(1.0 + c) / 2.0, -s / SQRT2, (1.0 - c) / 2.0], axis=-1),
        jnp.stack([s / SQRT2, c, -s / SQRT2], axis=-1),
        jnp.stack([(1.0 - c) / 2.0, s / SQRT2, (1.0 + c) / 2.0], axis=-1),
    ]
    return jnp.stack(rows, axis=-2).astype(jnp.complex64)


def expm_pade33_fixed(a, s=5):
    """Batched fixed scaling-and-squaring diagonal Pade(3,3) exponential."""
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    a = a / (2**s)
    a2 = a @ a
    odd = a @ (a2 + 60.0 * eye)
    even = 12.0 * a2 + 120.0 * eye
    r = jnp.linalg.solve(even - odd, even + odd)
    for _ in range(s):
        r = r @ r
    return r


def entangler_batch(theta, phi, beta):
    generator = (
        theta[:, None, None].astype(jnp.complex64) * DOT_BOND
        + (phi - theta)[:, None, None].astype(jnp.complex64) * ZZ_BOND
        + jnp.complex64(beta) * DOT_BOND_SQUARED
    )
    return expm_pade33_fixed(-1j * generator)


def apply_layer(state, lp, config):
    singles = jnp.einsum(
        "sab,sbc,scd->sad",
        rz_batch(lp["single_rz2"]),
        ry_batch(lp["single_ry"]),
        rz_batch(lp["single_rz1"]),
    )
    pair = jnp.einsum("kac,kbd->kabcd", singles[0::2], singles[1::2])
    pair = pair.reshape(-1, DIM * DIM, DIM * DIM)
    even = entangler_batch(lp["even_theta"], lp["even_phi"], config["beta"]) @ pair
    odd = entangler_batch(lp["odd_theta"], lp["odd_phi"], config["beta"])

    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    for k in range(even.shape[0]):
        gate = tc.gates.Gate(even[k].reshape((DIM,) * 4))
        circuit.unitary(2 * k, 2 * k + 1, unitary=gate, name="spin1_even_fused")
    for k in range(odd.shape[0]):
        gate = tc.gates.Gate(odd[k].reshape((DIM,) * 4))
        circuit.unitary(2 * k + 1, 2 * k + 2, unitary=gate, name="spin1_odd")
    return circuit.state()


def build_state(params, config):
    return K.scan(
        lambda s, p: apply_layer(s, p, config), params, initial_state(config)
    )


def make_energy_from_state(config):
    bond_gate = tc.gates.Gate(
        K.reshape(
            DOT_BOND + jnp.complex64(config["beta"]) * DOT_BOND_SQUARED,
            (DIM,) * 4,
        )
    )
    digits = basis_digit_table(config["n_sites"])
    onsite_coeffs = jnp.asarray(
        config["single_ion_anisotropy"] * SZ2_DIAG[digits].sum(axis=1),
        dtype=jnp.float32,
    )

    def energy_from_state(state):
        circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
        energy = K.cast(0.0, "complex64")
        for left in range(config["n_sites"] - 1):
            energy += circuit.expectation((bond_gate, [left, left + 1]))
        onsite = jnp.sum(onsite_coeffs * jnp.abs(state) ** 2)
        return (K.real(energy) + onsite) / config["n_sites"]

    return energy_from_state


def string_orders_from_state(state, config):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    values = []
    for i, j in string_pairs(config):
        operators = [(tc.gates.Gate(SZ), [i])]
        for site in range(i + 1, j):
            operators.append((tc.gates.Gate(STRING_MIDDLE), [site]))
        operators.append((tc.gates.Gate(SZ), [j]))
        values.append(K.real(circuit.expectation(*operators)))
    return K.stack(values)


def run_solution(config):
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    energy_from_state = make_energy_from_state(config)

    def loss_fn(p):
        return energy_from_state(build_state(p, config))

    def train_step(carry, _):
        p, state = carry
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        return (optax.apply_updates(p, updates), state), value

    @jax.jit
    def train(p, state):
        return jax.lax.scan(train_step, (p, state), None, length=config["max_steps"])

    @jax.jit
    def finalize(p):
        state = build_state(p, config)
        return energy_from_state(state), string_orders_from_state(state, config)

    (params, _), history = train(params, opt_state)
    final_energy_density, final_string_orders = finalize(params)

    return {
        "energy_density_history": K.numpy(history),
        "final_energy_density": K.numpy(final_energy_density),
        "final_string_orders": K.numpy(final_string_orders),
    }
