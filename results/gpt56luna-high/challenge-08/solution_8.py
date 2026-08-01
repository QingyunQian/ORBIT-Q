import numpy as np
import jax
import tensorcircuit as tc


def run_solution(config):
    s = int(config["grid_side"])
    n = int(config["n_qubits"])
    shots = int(config["n_samples"])
    tc.set_backend("jax")
    tc.set_contractor("omeco-4-8")
    c = tc.Circuit(n)

    for r in range(s):
        for col in range(s):
            q = s * r + col
            a = (config["ry_offset"]
                 + config["ry_row_sin_scale"] * np.sin(config["ry_row_sin_frequency"] * (r + 1))
                 + config["ry_col_cos_scale"] * np.cos(config["ry_col_cos_frequency"] * (col + 1))
                 + config["ry_diag_sin_scale"] * np.sin(config["ry_diag_sin_frequency"] * (r + col + 2)))
            c.ry(q, theta=a)

    k = 0
    for r in range(s):
        for col in range(s - 1):
            q = s * r + col
            b = (config["rzz_offset"]
                 + config["rzz_edge_sin_scale"] * np.sin(config["rzz_edge_sin_frequency"] * (k + 1))
                 + config["rzz_site_cos_scale"] * np.cos(config["rzz_site_cos_frequency"] * (2 * r + col + 1)))
            c.rzz(q, q + 1, theta=b)
            k += 1

    k = 0
    for r in range(s - 1):
        for col in range(s):
            q = s * r + col
            b = (config["rxx_offset"]
                 + config["rxx_edge_cos_scale"] * np.cos(config["rxx_edge_cos_frequency"] * (k + 1))
                 + config["rxx_site_sin_scale"] * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * col + 1)))
            c.rxx(q, q + s, theta=b)
            k += 1

    for r in range(s):
        for col in range(s):
            g = (config["rx_offset"]
                 + config["rx_row_cos_scale"] * np.cos(config["rx_row_cos_frequency"] * (r + 1))
                 - config["rx_col_sin_scale"] * np.sin(config["rx_col_sin_frequency"] * (col + 1))
                 + config["rx_diag_cos_scale"] * np.cos(config["rx_diag_cos_frequency"] * (r + col + 2)))
            c.rx(s * r + col, theta=g)

    chunk = 1024
    total = ((shots + chunk - 1) // chunk) * chunk
    key = jax.random.PRNGKey(8)
    u = jax.random.uniform(key, (total // 2, n))
    status = jax.numpy.concatenate((u, 1.0 - u), axis=0)
    sampler = jax.jit(jax.vmap(c.perfect_sampling))
    out = []
    for start in range(0, total, chunk):
        bits, _ = sampler(status[start:start + chunk])
        out.append(np.asarray(bits, dtype=np.int8))
    return {"samples": np.concatenate(out, axis=0)[:shots]}
