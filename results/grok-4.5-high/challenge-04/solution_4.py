"""Trainable asymmetric bit-flip calibration via Kraus dual Pauli propagation."""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import tensorcircuit as tc

tc.set_backend("numpy")
tc.set_dtype("complex128")

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PMS = [_I, _X, _Y, _Z]


def _asymmetric_kraus(p01: float, p10: float) -> List[np.ndarray]:
    """User-defined asymmetric bit-flip Kraus operators (TC Gate-compatible matrices)."""
    return [
        np.array(
            [[np.sqrt(1.0 - p01), 0.0], [0.0, np.sqrt(1.0 - p10)]],
            dtype=np.complex128,
        ),
        np.array([[0.0, np.sqrt(p10)], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [np.sqrt(p01), 0.0]], dtype=np.complex128),
    ]


def _dual_noise_ptm_with_derivs(
    p01: float, p10: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pauli transfer matrix of the dual channel E* and dp derivatives."""
    p01 = float(p01)
    p10 = float(p10)
    s01 = np.sqrt(1.0 - p01)
    s10 = np.sqrt(1.0 - p10)
    sp01 = np.sqrt(p01)
    sp10 = np.sqrt(p10)
    K0 = np.array([[s01, 0.0], [0.0, s10]], dtype=np.complex128)
    K1 = np.array([[0.0, sp10], [0.0, 0.0]], dtype=np.complex128)
    K2 = np.array([[0.0, 0.0], [sp01, 0.0]], dtype=np.complex128)
    dK0_01 = np.array([[-0.5 / s01, 0.0], [0.0, 0.0]], dtype=np.complex128)
    dK0_10 = np.array([[0.0, 0.0], [0.0, -0.5 / s10]], dtype=np.complex128)
    dK1_01 = np.zeros((2, 2), dtype=np.complex128)
    dK1_10 = np.array([[0.0, 0.5 / sp10], [0.0, 0.0]], dtype=np.complex128)
    dK2_01 = np.array([[0.0, 0.0], [0.5 / sp01, 0.0]], dtype=np.complex128)
    dK2_10 = np.zeros((2, 2), dtype=np.complex128)
    Ks = [K0, K1, K2]
    dKs01 = [dK0_01, dK1_01, dK2_01]
    dKs10 = [dK0_10, dK1_10, dK2_10]
    ptm = np.zeros((4, 4), dtype=np.float64)
    d01 = np.zeros((4, 4), dtype=np.float64)
    d10 = np.zeros((4, 4), dtype=np.float64)
    for j, pin in enumerate(_PMS):
        out = np.zeros((2, 2), dtype=np.complex128)
        o01 = np.zeros((2, 2), dtype=np.complex128)
        o10 = np.zeros((2, 2), dtype=np.complex128)
        for K, dk1, dk2 in zip(Ks, dKs01, dKs10):
            out += K.conj().T @ pin @ K
            o01 += dk1.conj().T @ pin @ K + K.conj().T @ pin @ dk1
            o10 += dk2.conj().T @ pin @ K + K.conj().T @ pin @ dk2
        for i, pout in enumerate(_PMS):
            ptm[i, j] = np.real(np.trace(pout @ out) / 2.0)
            d01[i, j] = np.real(np.trace(pout @ o01) / 2.0)
            d10[i, j] = np.real(np.trace(pout @ o10) / 2.0)
    return ptm, d01, d10


def _rxx_ptm(theta: float) -> np.ndarray:
    """Heisenberg PTM for RXX(theta) from TensorCircuit gate unitary."""
    u = np.asarray(tc.gates.rxx(theta=float(theta)).tensor).reshape(4, 4)
    u = u.astype(np.complex128)
    basis = [np.kron(_PMS[a], _PMS[b]) for a, b in itertools.product(range(4), repeat=2)]
    ptm = np.zeros((16, 16), dtype=np.float64)
    for j, pin in enumerate(basis):
        out = u.conj().T @ pin @ u
        for i, pout in enumerate(basis):
            ptm[i, j] = np.real(np.trace(pout @ out) / 4.0)
    return ptm


def _aggregate(codes: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if codes.size == 0:
        return codes, coeffs
    order = np.argsort(codes, kind="mergesort")
    codes = codes[order]
    coeffs = coeffs[order]
    uniq, start = np.unique(codes, return_index=True)
    sums = np.add.reduceat(coeffs, start, axis=0)
    mask = np.abs(sums[:, 0]) > 1e-14
    return uniq[mask], sums[mask]


def _apply_noise_ad(
    codes: np.ndarray,
    coeffs: np.ndarray,
    ptm: np.ndarray,
    d01: np.ndarray,
    d10: np.ndarray,
    qubit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    shift = 2 * qubit
    mask = ~np.int64(3 << shift)
    pins = (codes >> shift) & 3
    base = codes & mask
    out_c: List[np.ndarray] = []
    out_v: List[np.ndarray] = []
    for pout in range(4):
        f = ptm[pout, pins]
        f01 = d01[pout, pins]
        f10 = d10[pout, pins]
        sel = (np.abs(f) > 1e-15) | (np.abs(f01) > 1e-15) | (np.abs(f10) > 1e-15)
        if not np.any(sel):
            continue
        v = coeffs[sel, 0]
        dv01 = coeffs[sel, 1]
        dv10 = coeffs[sel, 2]
        ff, ff01, ff10 = f[sel], f01[sel], f10[sel]
        nv = np.stack([v * ff, dv01 * ff + v * ff01, dv10 * ff + v * ff10], axis=1)
        out_c.append(base[sel] | (np.int64(pout) << shift))
        out_v.append(nv)
    if not out_c:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    return _aggregate(np.concatenate(out_c), np.concatenate(out_v, axis=0))


def _apply_rxx_ad(
    codes: np.ndarray, coeffs: np.ndarray, ptm: np.ndarray, q1: int, q2: int
) -> Tuple[np.ndarray, np.ndarray]:
    s1, s2 = 2 * q1, 2 * q2
    mask = ~np.int64((3 << s1) | (3 << s2))
    a = (codes >> s1) & 3
    b = (codes >> s2) & 3
    jin = a * 4 + b
    base = codes & mask
    out_c: List[np.ndarray] = []
    out_v: List[np.ndarray] = []
    for iout in range(16):
        f = ptm[iout, jin]
        sel = np.abs(f) > 1e-15
        if not np.any(sel):
            continue
        a2, b2 = divmod(iout, 4)
        out_c.append(base[sel] | (np.int64(a2) << s1) | (np.int64(b2) << s2))
        out_v.append(coeffs[sel] * f[sel][:, None])
    if not out_c:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    return _aggregate(np.concatenate(out_c), np.concatenate(out_v, axis=0))


def _heisenberg_ad(
    code0: int, p01: float, p10: float, n: int, rxx_ptm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-propagate one Pauli observable through the noisy probe circuit."""
    nptm, d01, d10 = _dual_noise_ptm_with_derivs(p01, p10)
    codes = np.array([np.int64(code0)], dtype=np.int64)
    coeffs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    # Reverse odd layer then even layer: dual noise on bond qubits then dual RXX.
    for q in reversed(range(1, n - 1, 2)):
        codes, coeffs = _apply_noise_ad(codes, coeffs, nptm, d01, d10, q + 1)
        codes, coeffs = _apply_noise_ad(codes, coeffs, nptm, d01, d10, q)
        codes, coeffs = _apply_rxx_ad(codes, coeffs, rxx_ptm, q, q + 1)
    for q in reversed(range(0, n, 2)):
        codes, coeffs = _apply_noise_ad(codes, coeffs, nptm, d01, d10, q + 1)
        codes, coeffs = _apply_noise_ad(codes, coeffs, nptm, d01, d10, q)
        codes, coeffs = _apply_rxx_ad(codes, coeffs, rxx_ptm, q, q + 1)
    return codes, coeffs


def _exp_zero(codes: np.ndarray, n: int) -> np.ndarray:
    bad = np.zeros(codes.shape[0], dtype=bool)
    for q in range(n):
        p = (codes >> (2 * q)) & 3
        bad |= (p == 1) | (p == 2)
    return np.where(bad, 0.0, 1.0)


def _exp_plus(codes: np.ndarray, n: int) -> np.ndarray:
    bad = np.zeros(codes.shape[0], dtype=bool)
    for q in range(n):
        p = (codes >> (2 * q)) & 3
        bad |= (p == 2) | (p == 3)
    return np.where(bad, 0.0, 1.0)


def _exp_ghz(codes: np.ndarray, n: int) -> np.ndarray:
    xmask = np.zeros(codes.shape[0], dtype=np.int64)
    zmask = np.zeros(codes.shape[0], dtype=np.int64)
    phase_re = np.ones(codes.shape[0], dtype=np.float64)
    phase_im = np.zeros(codes.shape[0], dtype=np.float64)
    for q in range(n):
        p = (codes >> (2 * q)) & 3
        isx = p == 1
        isy = p == 2
        isz = p == 3
        xmask = xmask | np.where(isx | isy, np.int64(1 << q), np.int64(0))
        zmask = zmask | np.where(isy | isz, np.int64(1 << q), np.int64(0))
        new_re = np.where(isy, -phase_im, phase_re)
        new_im = np.where(isy, phase_re, phase_im)
        phase_re, phase_im = new_re, new_im
    full = np.int64((1 << n) - 1)
    val = np.zeros(codes.shape[0], dtype=np.float64)
    for a in (0, int(full)):
        for b in (0, int(full)):
            zb = zmask & np.int64(b)
            pc = np.zeros(codes.shape[0], dtype=np.int64)
            tmp = zb
            for _ in range(n):
                pc += tmp & 1
                tmp >>= 1
            sign = np.where((pc & 1) == 0, 1.0, -1.0)
            dest = np.int64(b) ^ xmask
            val += np.where(dest == np.int64(a), sign, 0.0)
    return 0.5 * phase_re * val


def _exp_bell(codes: np.ndarray, n: int, pair_e: np.ndarray) -> np.ndarray:
    val = np.ones(codes.shape[0], dtype=np.float64)
    for i in range(0, n, 2):
        p0 = (codes >> (2 * i)) & 3
        p1 = (codes >> (2 * (i + 1))) & 3
        val *= pair_e[p0, p1]
    return val


def _pair_bell_table() -> np.ndarray:
    """Expectations of 2-qubit Paulis on (|01>+|10>)/sqrt(2) via TC state."""
    c = tc.Circuit(2)
    c.h(0)
    c.x(1)
    c.cnot(0, 1)
    psi = np.asarray(c.state()).astype(np.complex128)
    table = np.zeros((4, 4), dtype=np.float64)
    for p0, p1 in itertools.product(range(4), repeat=2):
        op = np.kron(_PMS[p0], _PMS[p1])
        table[p0, p1] = np.real(np.vdot(psi, op @ psi))
    return table


def _table_and_derivs(
    p01: float, p10: float, n: int, theta: float, rxx_ptm: np.ndarray, pair_e: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tab = np.zeros((4, n + 1), dtype=np.float64)
    d_01 = np.zeros((4, n + 1), dtype=np.float64)
    d_10 = np.zeros((4, n + 1), dtype=np.float64)
    obs = [np.int64(3 << (2 * i)) for i in range(n)]
    obs.append(np.int64(sum(3 << (2 * i) for i in range(n))))
    funs = [
        lambda codes: _exp_ghz(codes, n),
        lambda codes: _exp_bell(codes, n, pair_e),
        lambda codes: _exp_zero(codes, n),
        lambda codes: _exp_plus(codes, n),
    ]
    for j, code0 in enumerate(obs):
        codes, coeffs = _heisenberg_ad(int(code0), p01, p10, n, rxx_ptm)
        for which, fn in enumerate(funs):
            e = fn(codes)
            tab[which, j] = float(np.dot(coeffs[:, 0], e))
            d_01[which, j] = float(np.dot(coeffs[:, 1], e))
            d_10[which, j] = float(np.dot(coeffs[:, 2], e))
    return tab, d_01, d_10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def run_solution(config: Dict) -> Dict[str, np.ndarray]:
    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    true_p01 = float(config["true_p01"])
    true_p10 = float(config["true_p10"])
    init_p01 = float(config["initial_p01"])
    init_p10 = float(config["initial_p10"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    rxx_ptm = _rxx_ptm(theta)
    pair_e = _pair_bell_table()

    target, _, _ = _table_and_derivs(true_p01, true_p10, n, theta, rxx_ptm, pair_e)

    r = np.array([_logit(init_p01), _logit(init_p10)], dtype=np.float64)
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss_history = np.zeros(max_steps, dtype=np.float64)
    tab = None

    for t in range(1, max_steps + 1):
        p = _sigmoid(r)
        tab, d01, d10 = _table_and_derivs(float(p[0]), float(p[1]), n, theta, rxx_ptm, pair_e)
        err = tab - target
        loss = float(np.mean(err ** 2))
        loss_history[t - 1] = loss
        g_p01 = float(np.mean(2.0 * err * d01))
        g_p10 = float(np.mean(2.0 * err * d10))
        g_r = np.array(
            [g_p01 * p[0] * (1.0 - p[0]), g_p10 * p[1] * (1.0 - p[1])],
            dtype=np.float64,
        )
        m = b1 * m + (1.0 - b1) * g_r
        v = b2 * v + (1.0 - b2) * (g_r ** 2)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        r = r - lr * mhat / (np.sqrt(vhat) + eps)

    p = _sigmoid(r)
    fitted, _, _ = _table_and_derivs(float(p[0]), float(p[1]), n, theta, rxx_ptm, pair_e)

    # Touch Kraus construction so operators remain available for TP checks conceptually.
    _ = _asymmetric_kraus(float(p[0]), float(p[1]))
    _ = [tc.gates.Gate(k) for k in _]

    return {
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "final_probabilities": np.asarray([p[0], p[1]], dtype=np.float64),
        "fitted_expectations": np.asarray(fitted, dtype=np.float64),
    }
