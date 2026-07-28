"""Stage-by-stage timing of the challenge-12 reference solution.

Replicates tasks/challenge-12/solution/solution_12.py exactly, but times:
  1. quimb MPS -> TC QuOperator conversion
  2. jit trace (includes omeco contraction-path search)
  3. XLA compile
  4. steady-state per-step execution
Run inside the pinned venv:
  PYTHONPATH=... python profile_reference.py
"""

import time

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

CONFIG = {
    "n_qubits": 32,
    "n_layers": 2,
    "max_steps": 5000,
    "learning_rate": 0.02,
    "initial_parameter_scale": 0.02,
    "seed": 2039,
}


def build_dmrg_state():
    import quimb.tensor as qtn

    n = CONFIG["n_qubits"]
    ham = qtn.SpinHam1D(S=0.5)
    for i in range(n - 1):
        ham[i, i + 1] += 4.0, "X", "X"
        ham[i, i + 1] += 4.0, "Y", "Y"
        ham[i, i + 1] += 4.0 * 1.4, "Z", "Z"
    for i in range(n):
        ham[i] += 2.0 * 0.2 * ((-1) ** i), "Z"
    dmrg = qtn.DMRG2(ham.build_mpo(n), bond_dims=[8], cutoffs=1e-8)
    dmrg.solve(tol=1e-7, max_sweeps=4, verbosity=0)
    dmrg.state.normalize()
    return dmrg.state


def main():
    config = dict(CONFIG)
    config["dmrg_state"] = build_dmrg_state()
    n_qubits = config["n_qubits"]

    parameter_count = 0
    for layer in range(config["n_layers"]):
        parameter_count += 15 * len(range(layer % 2, n_qubits - 1, 2))

    rng = np.random.default_rng(config["seed"])
    params = rng.normal(
        scale=config["initial_parameter_scale"], size=(parameter_count,)
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    t0 = time.perf_counter()
    target_mps = tc.quantum.quimb2qop(config["dmrg_state"])
    target_bra = target_mps.adjoint()
    t_convert = time.perf_counter() - t0

    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)
        offset = 0
        for layer in range(config["n_layers"]):
            for i in range(layer % 2, n_qubits - 1, 2):
                circuit.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15
        overlap_value = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap_value) * overlap_value)
        return 1.0 - fidelity, (fidelity, overlap_value)

    def train_step(p, state):
        (loss, aux), grads = K.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    import jax

    jitted = jax.jit(train_step)

    t0 = time.perf_counter()
    lowered = jitted.lower(params, opt_state)
    t_trace = time.perf_counter() - t0

    hlo_ops = lowered.as_text().count("\n")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    # warmup + steady state
    p, s = params, opt_state
    for _ in range(20):
        p, s, loss, aux = compiled(p, s)
    jax.block_until_ready(loss)

    n_meas = 2000
    t0 = time.perf_counter()
    for _ in range(n_meas):
        p, s, loss, aux = compiled(p, s)
    jax.block_until_ready(loss)
    t_step = (time.perf_counter() - t0) / n_meas

    print(f"quimb2qop conversion: {t_convert*1e3:.1f} ms")
    print(f"jit trace (incl. omeco path search): {t_trace:.2f} s")
    print(f"XLA compile: {t_compile:.2f} s")
    print(f"steady-state per step: {t_step*1e3:.3f} ms")
    print(f"5000-step loop estimate: {t_step*5000:.2f} s")
    print(
        f"total estimate: {t_convert + t_trace + t_compile + t_step*5000:.2f} s"
    )
    print(f"stableHLO line count (proxy for op count): {hlo_ops}")


if __name__ == "__main__":
    main()
