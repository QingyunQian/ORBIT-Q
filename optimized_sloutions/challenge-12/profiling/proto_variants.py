"""Prototype + cross-validation of challenge-12 optimization variants.

Checks numerical equivalence of the objective against the reference
implementation and stage-times each variant (trace / compile / per-step).
"""

import time

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm

N = 32
N_LAYERS = 2
LAYER_BONDS = [list(range(l % 2, N - 1, 2)) for l in range(N_LAYERS)]
N_GATES = sum(len(b) for b in LAYER_BONDS)  # 31

# --- su(4) generators, exact order used by tc.gates.su4_gate ---
_G_NAMES = ["ix", "iy", "iz", "xi", "xx", "xy", "xz", "yi", "yx", "yy", "yz",
            "zi", "zx", "zy", "zz"]
GENS = np.stack([getattr(tc.gates, f"_{n}_matrix") for n in _G_NAMES])

_P1 = {"i": np.eye(2), "x": np.array([[0, 1], [1, 0]]),
       "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1.0, -1.0])}
GENS_MANUAL = np.stack([
    np.einsum("ab,cd->acbd", _P1[n[0]], _P1[n[1]]).reshape(4, 4)
    for n in _G_NAMES
])
assert np.allclose(GENS, GENS_MANUAL), "generator order mismatch"

GENS_J = jnp.asarray(GENS, dtype=jnp.complex64)


def build_dmrg_state():
    import quimb.tensor as qtn

    ham = qtn.SpinHam1D(S=0.5)
    for i in range(N - 1):
        ham[i, i + 1] += 4.0, "X", "X"
        ham[i, i + 1] += 4.0, "Y", "Y"
        ham[i, i + 1] += 4.0 * 1.4, "Z", "Z"
    for i in range(N):
        ham[i] += 2.0 * 0.2 * ((-1) ** i), "Z"
    dmrg = qtn.DMRG2(ham.build_mpo(N), bond_dims=[8], cutoffs=1e-8)
    dmrg.solve(tol=1e-7, max_sweeps=4, verbosity=0)
    dmrg.state.normalize()
    return dmrg.state


def su4_batch(thetas):
    """(G,15) real -> (G,4,4) SU4 matrices, identical math to tc su4_gate."""
    h = jnp.einsum("gi,iab->gab", thetas.astype(jnp.complex64), GENS_J)
    return jexpm(-1j * h)


# ---------------- reference objective ----------------
def make_ref_objective(target_bra):
    def objective(p):
        c = tc.Circuit(N)
        for i in range(1, N, 2):
            c.x(i)
        offset = 0
        for layer in range(N_LAYERS):
            for i in LAYER_BONDS[layer]:
                c.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15
        ov = (target_bra @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


# ---------------- V2: batched gate construction ----------------
def make_v2_objective(target_bra):
    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        c = tc.Circuit(N)
        for i in range(1, N, 2):
            c.x(i)
        k = 0
        for layer in range(N_LAYERS):
            for i in LAYER_BONDS[layer]:
                c.any(i, i + 1, unitary=u[k])
                k += 1
        ov = (target_bra @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


# ---------------- V3: pair-fused ququart circuit ----------------
def fused_bra(dmrg_state):
    mps = dmrg_state.copy()
    mps.permute_arrays("lpr")
    arrs = [np.asarray(a, dtype=np.complex64) for a in mps.arrays]
    arrs[0] = arrs[0].reshape(1, *arrs[0].shape)
    arrs[-1] = arrs[-1].reshape(*arrs[-1].shape, 1)
    fused = []
    for j in range(N // 2):
        a, b = arrs[2 * j], arrs[2 * j + 1]
        t = np.einsum("lpm,mqr->lpqr", a, b)
        fused.append(t.reshape(t.shape[0], 4, t.shape[3]))
    fused[0] = fused[0][0]          # (4, chi)
    fused[-1] = fused[-1][..., 0]   # (chi, 4)
    nodes = [tc.quantum.Node(t) for t in fused]
    for j in range(len(nodes) - 1):
        nodes[j][-1] ^ nodes[j + 1][0]
    out_edges = [nodes[0][0]] + [nodes[j][1] for j in range(1, len(nodes))]
    return tc.quantum.QuVector(out_edges).adjoint()


PERM_01 = np.zeros((4, 4), dtype=np.complex64)
for _f in range(4):
    PERM_01[_f ^ 1, _f] = 1.0  # |ab> -> |a, b xor 1>: maps |00..>->|01..>
I2_J = jnp.eye(2, dtype=jnp.complex64)
PERM_J = jnp.asarray(PERM_01)


def make_v3_objective(bra_fused):
    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        u1 = jnp.einsum("gab,bc->gac", u[:16], PERM_J)          # (16,4,4)
        w2 = u[16:].reshape(15, 2, 2, 2, 2)                     # [g,p',q',p,q]
        wt = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I2_J, w2, I2_J)
        wt = wt.reshape(15, 16, 16)
        c = tc.QuditCircuit(N // 2, dim=4)
        for j in range(16):
            c.any(j, unitary=u1[j])
        for j in range(15):
            c.any(j, j + 1, unitary=wt[j])
        ov = (bra_fused @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


def stage_times(objective, params, label, n_meas=500):
    opt = optax.adam(0.02)
    state = opt.init(params)

    def train_step(p, s):
        (loss, aux), g = jax.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss, aux

    jitted = jax.jit(train_step)
    t0 = time.perf_counter()
    lowered = jitted.lower(params, state)
    t_trace = time.perf_counter() - t0
    hlo = lowered.as_text().count("\n")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    p, s = params, state
    for _ in range(20):
        p, s, loss, aux = compiled(p, s)
    jax.block_until_ready(loss)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        p, s, loss, aux = compiled(p, s)
    jax.block_until_ready(loss)
    t_step = (time.perf_counter() - t0) / n_meas
    total = t_trace + t_compile + 5000 * t_step
    print(
        f"{label}: trace {t_trace:.2f}s  compile {t_compile:.2f}s  "
        f"step {t_step*1e3:.3f}ms  hlo {hlo}  est-total {total:.2f}s"
    )
    return loss


def main():
    dmrg_state = build_dmrg_state()
    target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
    bra_f = fused_bra(dmrg_state)

    rng = np.random.default_rng(2039)
    params = jnp.asarray(
        rng.normal(scale=0.02, size=(N_GATES * 15,)).astype(np.float32)
    )

    ref = make_ref_objective(target_bra)
    v2 = make_v2_objective(target_bra)
    v3 = make_v3_objective(bra_f)

    # numerical equivalence on a few random parameter vectors
    for trial in range(3):
        p = jnp.asarray(
            rng.normal(scale=0.3, size=(N_GATES * 15,)).astype(np.float32)
        )
        (l0, (f0, o0)) = jax.jit(ref)(p)
        (l2, (f2, o2)) = jax.jit(v2)(p)
        (l3, (f3, o3)) = jax.jit(v3)(p)
        print(
            f"trial {trial}: ov ref {complex(o0):.8f}  "
            f"v2 diff {abs(complex(o2)-complex(o0)):.2e}  "
            f"v3 diff {abs(complex(o3)-complex(o0)):.2e}"
        )

    stage_times(ref, params, "reference")
    stage_times(v2, params, "v2-batched")
    stage_times(v3, params, "v3-fused  ")


if __name__ == "__main__":
    main()
