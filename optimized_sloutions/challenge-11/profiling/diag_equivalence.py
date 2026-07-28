"""Diagnostics for the optimized challenge-11 solution:

1. Early-trajectory equivalence: per-step |E_ref - E_fused| for the first K
   steps from the identical seeded initialization (fusion changes float
   rounding, so deltas start at the complex64 noise floor rather than zero,
   and grow only through optimizer dynamics).
2. Entangler generator-norm coverage over all 500 steps (accuracy-margin
   input for the fixed-order Pade exponential).
3. Final-metric comparison at 500 steps: energy density, gap, string orders.
"""

import importlib
import sys

sys.path.insert(0, "/workspace/tasks/challenge-11/solution")
sys.path.insert(0, "/workspace/optimized_sloutions/challenge-11")

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp

ref = importlib.import_module("solution_11")
fus = importlib.import_module("solution_11_fused")

CONFIG = {
    "n_sites": 12,
    "n_layers": 5,
    "beta": 0.20,
    "single_ion_anisotropy": 0.15,
    "max_steps": 500,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.05,
    "seed": 2041,
}
CHECK_STEPS = 100


def scan_run(objective, params, steps, extra=None):
    opt = optax.adam(CONFIG["learning_rate"])

    def body(carry, _):
        p, s = carry
        v, g = jax.value_and_grad(objective)(p)
        upd, s = opt.update(g, s, p)
        ys = (v,) if extra is None else (v, extra(p))
        return (optax.apply_updates(p, upd), s), ys

    @jax.jit
    def run(p):
        (p, _), ys = jax.lax.scan(body, (p, opt.init(p)), None, length=steps)
        return p, ys

    return run(params)


def main():
    params = ref.initial_parameters(CONFIG)
    energy_fused = fus.make_energy_from_state(CONFIG)

    obj_ref = lambda p: ref.energy_density(p, CONFIG)
    obj_fus = lambda p: energy_fused(fus.build_state(p, CONFIG))

    _, (hist_ref,) = scan_run(obj_ref, params, CHECK_STEPS)
    def gen_norm_bound(p):
        # spectral-norm bound: |theta|*|DOT| + |phi-theta|*|ZZ| + beta*|DOT^2|
        t = jnp.concatenate([p["even_theta"].reshape(-1), p["odd_theta"].reshape(-1)])
        f = jnp.concatenate([p["even_phi"].reshape(-1), p["odd_phi"].reshape(-1)])
        return jnp.max(jnp.abs(t) * 2.0 + jnp.abs(f - t) * 1.0) + CONFIG["beta"] * 4.0

    p_final, (hist_fus, norms) = scan_run(obj_fus, params, 500, extra=gen_norm_bound)

    d = np.abs(np.asarray(hist_ref) - np.asarray(hist_fus)[:CHECK_STEPS])
    for k in (5, 20, 50, 100):
        print(f"max |dE| first {k:3d} steps: {d[:k].max():.3e}")
    print(f"max entangler generator norm bound over 500 steps: "
          f"{float(np.asarray(norms).max()):.4f} "
          f"(scaled by 2**-5 -> {float(np.asarray(norms).max())/32:.4f})")

    # final metrics from the fused final parameters
    @jax.jit
    def finalize(p):
        st = fus.build_state(p, CONFIG)
        return energy_fused(st), fus.string_orders_from_state(st, CONFIG)

    e_fin, strings = finalize(p_final)
    print(f"fused final energy density (500 steps): {float(e_fin):+.8f}")
    print("fused final string orders:", np.asarray(strings))


if __name__ == "__main__":
    main()
