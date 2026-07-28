"""Round 2: scan-based loop timing for challenge-12 variants.

Each mode runs in its own process (XLA flags are process-global):
  dispatch   -- trivial jitted-call dispatch floor
  v1scan     -- reference objective + lax.scan
  v3scan     -- ququart-fused objective + lax.scan
  v3scan-u4  -- same with scan unroll=4
  v16scan    -- dim-16 fusion (8 sites of 4 qubits) + lax.scan
"""

import argparse
import sys
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
N_GATES = sum(len(b) for b in LAYER_BONDS)

_G_NAMES = ["ix", "iy", "iz", "xi", "xx", "xy", "xz", "yi", "yx", "yy", "yz",
            "zi", "zx", "zy", "zz"]
GENS_J = jnp.asarray(
    np.stack([getattr(tc.gates, f"_{n}_matrix") for n in _G_NAMES]),
    dtype=jnp.complex64,
)
I2_J = jnp.eye(2, dtype=jnp.complex64)
PERM_J = jnp.asarray(np.eye(4)[:, [1, 0, 3, 2]], dtype=jnp.complex64)


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
    h = jnp.einsum("gi,iab->gab", thetas.astype(jnp.complex64), GENS_J)
    return jexpm(-1j * h)


def mps_arrays(dmrg_state):
    mps = dmrg_state.copy()
    mps.permute_arrays("lpr")
    arrs = [np.asarray(a, dtype=np.complex64) for a in mps.arrays]
    arrs[0] = arrs[0].reshape(1, *arrs[0].shape)
    arrs[-1] = arrs[-1].reshape(*arrs[-1].shape, 1)
    return arrs


def fused_bra(dmrg_state, group=2):
    arrs = mps_arrays(dmrg_state)
    n_sites = N // group
    fused = []
    for j in range(n_sites):
        t = arrs[group * j]
        for k in range(1, group):
            t = np.einsum("l...m,mqr->l...qr", t, arrs[group * j + k])
        fused.append(t.reshape(t.shape[0], 2**group, t.shape[-1]))
    fused[0] = fused[0][0]
    fused[-1] = fused[-1][..., 0]
    nodes = [tc.quantum.Node(t) for t in fused]
    for j in range(len(nodes) - 1):
        nodes[j][-1] ^ nodes[j + 1][0]
    out_edges = [nodes[0][0]] + [nodes[j][1] for j in range(1, len(nodes))]
    return tc.quantum.QuVector(out_edges).adjoint()


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


def expm_pade33_fixed(a, s=5):
    a = a / (2**s)
    eye = jnp.eye(4, dtype=a.dtype)
    a2 = a @ a
    u = a @ (a2 + 60.0 * eye)
    v = 12.0 * a2 + 120.0 * eye
    r = jnp.linalg.solve(v - u, v + u)
    for _ in range(s):
        r = r @ r
    return r


def make_v2p_objective(target_bra):
    """Unfused 32-qubit network + batched fixed-order Pade expm."""

    def objective(p):
        h = jnp.einsum("gi,iab->gab",
                       p.reshape(N_GATES, 15).astype(jnp.complex64), GENS_J)
        u = expm_pade33_fixed(-1j * h)
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


def make_v3_objective(bra_f):
    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        u1 = jnp.einsum("gab,bc->gac", u[:16], PERM_J)
        w2 = u[16:].reshape(15, 2, 2, 2, 2)
        wt = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I2_J, w2, I2_J)
        wt = wt.reshape(15, 16, 16)
        c = tc.QuditCircuit(N // 2, dim=4)
        for j in range(16):
            c.any(j, unitary=u1[j])
        for j in range(15):
            c.any(j, j + 1, unitary=wt[j])
        ov = (bra_f @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


def make_v3a_objective(bra_f):
    """v3 with the 16 single-ququart layer-1 gates absorbed into the 15
    two-ququart layer-2 blocks: network = 16 inputs + 15 gates + 16 bra."""

    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        u1 = jnp.einsum("gab,bc->gac", u[:16], PERM_J)          # (16,4,4)
        w2 = u[16:].reshape(15, 2, 2, 2, 2)
        wt = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I2_J, w2, I2_J)
        wt = wt.reshape(15, 16, 16)
        # absorb: W'_0 = W_0 (U1_0 x U1_1); W'_j = W_j (I4 x U1_{j+1}), j>=1
        u1kron = jnp.einsum("ab,gcd->gacbd", jnp.eye(4, dtype=jnp.complex64),
                            u1[2:]).reshape(14, 16, 16)
        w0 = wt[0] @ jnp.einsum("ab,cd->acbd", u1[0], u1[1]).reshape(16, 16)
        wp = jnp.concatenate(
            [w0[None], jnp.einsum("gab,gbc->gac", wt[1:], u1kron)], axis=0
        )
        c = tc.QuditCircuit(N // 2, dim=4)
        for j in range(15):
            c.any(j, j + 1, unitary=wp[j])
        ov = (bra_f @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


def make_gates_only_objective():
    """Cost decomposition helper: gate construction only."""

    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        u1 = jnp.einsum("gab,bc->gac", u[:16], PERM_J)
        w2 = u[16:].reshape(15, 2, 2, 2, 2)
        wt = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I2_J, w2, I2_J)
        wt = wt.reshape(15, 16, 16)
        s = jnp.sum(jnp.abs(u1) ** 2) + jnp.sum(jnp.abs(wt) ** 2)
        return jnp.real(s), (jnp.real(s), s.astype(jnp.complex64))

    return objective


def make_v16_objective(bra_f):
    """8 sites of 4 qubits each (dim=16).

    Per 4-qubit cell j (qubits 4j..4j+3): layer-1 gates on (4j,4j+1) and
    (4j+2,4j+3), layer-2 gate on (4j+1,4j+2) stay inside the cell; layer-2
    gate on (4j+3,4j+4) crosses to cell j+1.
    """
    I4 = jnp.eye(4, dtype=jnp.complex64)

    def objective(p):
        u = su4_batch(p.reshape(N_GATES, 15))
        u1 = jnp.einsum("gab,bc->gac", u[:16], PERM_J)  # 16 layer-1 (Neel folded)
        w = u[16:]                                       # 15 layer-2
        # inside-cell composite: (W_in x I) (U1a x U1b) -> (16,16), batch 8
        u1a, u1b = u1[0::2], u1[1::2]                    # (8,4,4)
        u11 = jnp.einsum("gab,gcd->gacbd", u1a, u1b).reshape(8, 16, 16)
        w_in = w[0::2]                                   # cells' inner W (8? -> 8 cells have inner W indices 0,2,4,..14)
        w2 = w_in.reshape(8, 2, 2, 2, 2)
        w_mid = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I2_J, w2, I2_J)
        w_mid = w_mid.reshape(8, 16, 16)
        cell = jnp.einsum("gab,gbc->gac", w_mid, u11)    # (8,16,16)
        # crossing gates: W on (4j+3, 4j+4), j=0..6 -> I4 x W x I4? careful:
        # site j qubits (4j..4j+3): crossing W acts on last qubit of cell j
        # and first qubit of cell j+1: Wc = I(8) x W x I(8) grouped (16,16)^2
        w_cr = w[1::2].reshape(7, 2, 2, 2, 2)            # indices p'q'pq
        I8 = jnp.eye(8, dtype=jnp.complex64).reshape(8, 8)
        wc = jnp.einsum("xX,gpqPQ,yY->gxpqyXPQY", I8, w_cr, I8)
        # grouping: out site j = (x(8),p(2)) = 16; out site j+1 = (q(2),y(8)) = 16
        wc = wc.reshape(7, 256, 256)

        c = tc.QuditCircuit(8, dim=16)
        for j in range(8):
            c.any(j, unitary=cell[j])
        for j in range(7):
            c.any(j, j + 1, unitary=wc[j])
        ov = (bra_f @ c.quvector()).eval()
        fid = K.real(K.conj(ov) * ov)
        return 1.0 - fid, (fid, ov)

    return objective


def run_scan(objective, params, label, unroll=1, steps=2000):
    opt = optax.adam(0.02)
    state = opt.init(params)

    def body(carry, _):
        p, s = carry
        (loss, (fid, ov)), g = jax.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(g, s, p)
        return (optax.apply_updates(p, upd), s), (loss, fid, ov)

    @jax.jit
    def train(p, s):
        (p, s), ys = jax.lax.scan(body, (p, s), None, length=steps,
                                  unroll=unroll)
        return p, s, ys

    t0 = time.perf_counter()
    lowered = train.lower(params, state)
    t_trace = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = compiled(params, state)
    jax.block_until_ready(out)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = compiled(params, state)
    jax.block_until_ready(out)
    t_second = time.perf_counter() - t0
    per_step = t_second / steps
    est = t_trace + t_compile + per_step * 5000
    print(
        f"{label}: trace {t_trace:.2f}s compile {t_compile:.2f}s "
        f"first {t_first:.2f}s per-step {per_step*1e3:.3f}ms "
        f"est-total(5000) {est:.2f}s  loss[-1]={float(out[2][0][-1]):.6f}"
    )


def run_dispatch(params):
    @jax.jit
    def f(p, x):
        return p, x + 1.0

    x = jnp.float32(0)
    p = params
    for _ in range(50):
        p, x = f(p, x)
    jax.block_until_ready(x)
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        p, x = f(p, x)
    jax.block_until_ready(x)
    print(f"dispatch floor: {(time.perf_counter()-t0)/n*1e6:.1f} us/call")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(2039)
    params = jnp.asarray(
        rng.normal(scale=0.02, size=(N_GATES * 15,)).astype(np.float32)
    )

    if args.mode == "dispatch":
        run_dispatch(params)
        return

    dmrg_state = build_dmrg_state()

    if args.mode == "v1scan":
        target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
        run_scan(make_ref_objective(target_bra), params, "v1scan(ref+scan)")
    elif args.mode == "v3a":
        obj3 = make_v3_objective(fused_bra(dmrg_state, 2))
        obj3a = make_v3a_objective(fused_bra(dmrg_state, 2))
        p = jnp.asarray(
            np.random.default_rng(7).normal(scale=0.3, size=(N_GATES * 15,))
            .astype(np.float32)
        )
        (_, (_, o3)) = jax.jit(obj3)(p)
        (_, (_, o3a)) = jax.jit(obj3a)(p)
        print(f"v3a vs v3 overlap diff: {abs(complex(o3a)-complex(o3)):.2e}")
        run_scan(obj3a, params, "v3a-absorb")
    elif args.mode == "gatesonly":
        run_scan(make_gates_only_objective(), params, "gates-only")
    elif args.mode == "v2p":
        target_bra = tc.quantum.quimb2qop(dmrg_state).adjoint()
        run_scan(make_v2p_objective(target_bra), params, "v2p(pade+scan,unfused)")
    elif args.mode == "v3scan":
        run_scan(make_v3_objective(fused_bra(dmrg_state, 2)), params, "v3scan")
    elif args.mode == "v3scan-u4":
        run_scan(
            make_v3_objective(fused_bra(dmrg_state, 2)), params,
            "v3scan-u4", unroll=4,
        )
    elif args.mode == "v16scan":
        # correctness check against v3 objective first
        obj3 = make_v3_objective(fused_bra(dmrg_state, 2))
        obj16 = make_v16_objective(fused_bra(dmrg_state, 4))
        p = jnp.asarray(
            np.random.default_rng(7).normal(scale=0.3, size=(N_GATES * 15,))
            .astype(np.float32)
        )
        (l3, (f3, o3)) = jax.jit(obj3)(p)
        (l16, (f16, o16)) = jax.jit(obj16)(p)
        print(f"v16 vs v3 overlap diff: {abs(complex(o16)-complex(o3)):.2e}")
        run_scan(obj16, params, "v16scan")
    else:
        raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
