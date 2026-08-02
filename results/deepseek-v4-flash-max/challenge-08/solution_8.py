import numpy as np
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("numpy")

    side = config["grid_side"]
    n = config["n_qubits"]
    ns = config["n_samples"]

    def alpha(r, c):
        return (
            config["ry_offset"]
            + config["ry_row_sin_scale"] * np.sin(config["ry_row_sin_frequency"] * (r + 1))
            + config["ry_col_cos_scale"] * np.cos(config["ry_col_cos_frequency"] * (c + 1))
            + config["ry_diag_sin_scale"] * np.sin(config["ry_diag_sin_frequency"] * (r + c + 2))
        )

    def gamma(r, c):
        return (
            config["rx_offset"]
            + config["rx_row_cos_scale"] * np.cos(config["rx_row_cos_frequency"] * (r + 1))
            - config["rx_col_sin_scale"] * np.sin(config["rx_col_sin_frequency"] * (c + 1))
            + config["rx_diag_cos_scale"] * np.cos(config["rx_diag_cos_frequency"] * (r + c + 2))
        )

    c = tc.MPSCircuit(n)

    for r in range(side):
        for cc in range(side):
            c.apply_single_gate(tc.gates.ry(theta=alpha(r, cc)), r * side + cc)

    kh = 0
    for r in range(side):
        for cc in range(side - 1):
            kh += 1
            th = (
                config["rzz_offset"]
                + config["rzz_edge_sin_scale"] * np.sin(config["rzz_edge_sin_frequency"] * (kh + 1))
                + config["rzz_site_cos_scale"] * np.cos(config["rzz_site_cos_frequency"] * (2 * r + cc + 1))
            )
            c.apply_double_gate(tc.gates.rzz(theta=th), r * side + cc, r * side + cc + 1)

    kv = 0
    for r in range(side - 1):
        for cc in range(side):
            kv += 1
            th = (
                config["rxx_offset"]
                + config["rxx_edge_cos_scale"] * np.cos(config["rxx_edge_cos_frequency"] * (kv + 1))
                + config["rxx_site_sin_scale"] * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * cc + 1))
            )
            c.apply_double_gate(tc.gates.rxx(theta=th), r * side + cc, (r + 1) * side + cc)

    for r in range(side):
        for cc in range(side):
            c.apply_single_gate(tc.gates.rx(theta=gamma(r, cc)), r * side + cc)

    # Exact right-canonical tensor-network representation of the output state.
    c.position(0)
    tensors = c.get_tensors()

    rng = np.random.default_rng()
    samples = np.empty((ns, n), dtype=np.int64)
    one = np.array([1.0 + 0.0j])

    for k in range(ns):
        v = one
        row = samples[k]
        for i, t in enumerate(tensors):
            w0 = np.tensordot(v, t[:, 0, :], axes=(0, 0))
            w1 = np.tensordot(v, t[:, 1, :], axes=(0, 0))
            p0 = np.vdot(w0, w0).real
            p1 = np.vdot(w1, w1).real
            norm = p0 + p1
            p0 /= norm
            if rng.random() < p0:
                row[i] = 0
                v = w0 / np.sqrt(p0)
            else:
                row[i] = 1
                v = w1 / np.sqrt(1.0 - p0)

    return {"samples": samples}
