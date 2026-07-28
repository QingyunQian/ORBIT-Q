"""Profile task-11: expert reference vs Fable 5 candidate.

Decomposes end-to-end time into (a) JIT compile of the training step and
(b) steady-state per-step cost, and separately measures forward-energy vs
value_and_grad, to locate the bottleneck. Runs both in the identical local
environment.
"""
import importlib.util
import sys
import time

import numpy as np

CONFIG = {
    "n_sites": 12, "n_layers": 5, "beta": 0.20, "single_ion_anisotropy": 0.15,
    "max_steps": 500, "learning_rate": 0.03, "initial_parameter_scale": 0.05,
    "seed": 2041,
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def profile_reference(path):
    import jax
    m = load(path, "ref11mod")
    K = m.K
    params = m.initial_parameters(CONFIG)

    # forward energy compile + steady
    f = K.jit(lambda p: m.energy_density(p, CONFIG))
    t0 = time.monotonic(); e = f(params); e.block_until_ready(); t_fwd_compile = time.monotonic() - t0
    t0 = time.monotonic()
    for _ in range(10):
        e = f(params); e.block_until_ready()
    t_fwd = (time.monotonic() - t0) / 10

    # value_and_grad compile + steady
    vg = K.jit(K.value_and_grad(lambda p: m.energy_density(p, CONFIG)))
    t0 = time.monotonic(); v, g = vg(params); jax.block_until_ready((v, g)); t_vg_compile = time.monotonic() - t0
    t0 = time.monotonic()
    for _ in range(10):
        v, g = vg(params); jax.block_until_ready((v, g))
    t_vg = (time.monotonic() - t0) / 10
    return dict(fwd_compile=t_fwd_compile, fwd=t_fwd, vg_compile=t_vg_compile, vg=t_vg)


def profile_candidate(path):
    """The candidate keeps its optimizer loop internal; reconstruct the same
    energy function from its building blocks for an apples-to-apples split."""
    import jax
    import jax.numpy as jnp
    import tensorcircuit as tc
    m = load(path, "cand11mod")
    # Rebuild the candidate's energy_density closure by calling run_solution
    # internals is awkward; instead re-express via its module-level helpers.
    # The candidate exposes run_solution only, so we time the full call and
    # also its internal energy via monkeypatching the step counter.
    return None


if __name__ == "__main__":
    which = sys.argv[1]
    path = sys.argv[2]
    r = profile_reference(path)
    print(f"[{which}] forward-energy: compile {r['fwd_compile']:.1f}s, steady {r['fwd']*1000:.1f}ms/call")
    print(f"[{which}] value_and_grad: compile {r['vg_compile']:.1f}s, steady {r['vg']*1000:.1f}ms/call")
    print(f"[{which}] projected 500-step train (vg steady only): {r['vg']*500:.1f}s + compile {r['vg_compile']:.0f}s")
