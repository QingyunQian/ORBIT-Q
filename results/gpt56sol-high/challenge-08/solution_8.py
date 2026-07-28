import numpy as np


def run_solution(config):
    import tensorcircuit as tc
    from scipy.stats import qmc

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    # The split is exact for RZZ and RXX (operator Schmidt rank two).  OMECO
    # avoids the state-vector-like paths selected by the default contractor.
    tc.set_contractor("omeco-4-8", preprocessing=True)

    g = int(config["grid_side"])
    n = int(config["n_qubits"])
    c = tc.Circuit(n, split={"max_singular_values": 2})

    for r in range(g):
        for col in range(g):
            a = (
                config["ry_offset"]
                + config["ry_row_sin_scale"]
                * np.sin(config["ry_row_sin_frequency"] * (r + 1))
                + config["ry_col_cos_scale"]
                * np.cos(config["ry_col_cos_frequency"] * (col + 1))
                + config["ry_diag_sin_scale"]
                * np.sin(config["ry_diag_sin_frequency"] * (r + col + 2))
            )
            c.ry(g * r + col, theta=a)

    k = 0
    for r in range(g):
        for col in range(g - 1):
            b = (
                config["rzz_offset"]
                + config["rzz_edge_sin_scale"]
                * np.sin(config["rzz_edge_sin_frequency"] * (k + 1))
                + config["rzz_site_cos_scale"]
                * np.cos(config["rzz_site_cos_frequency"] * (2 * r + col + 1))
            )
            c.rzz(g * r + col, g * r + col + 1, theta=b)
            k += 1

    k = 0
    for r in range(g - 1):
        for col in range(g):
            b = (
                config["rxx_offset"]
                + config["rxx_edge_cos_scale"]
                * np.cos(config["rxx_edge_cos_frequency"] * (k + 1))
                + config["rxx_site_sin_scale"]
                * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * col + 1))
            )
            c.rxx(g * r + col, g * (r + 1) + col, theta=b)
            k += 1

    for r in range(g):
        for col in range(g):
            a = (
                config["rx_offset"]
                + config["rx_row_cos_scale"]
                * np.cos(config["rx_row_cos_frequency"] * (r + 1))
                - config["rx_col_sin_scale"]
                * np.sin(config["rx_col_sin_frequency"] * (col + 1))
                + config["rx_diag_cos_scale"]
                * np.cos(config["rx_diag_cos_frequency"] * (r + col + 2))
            )
            c.rx(g * r + col, theta=a)

    shots = int(config["n_samples"])
    if shots > 0 and shots & (shots - 1) == 0:
        status = qmc.Sobol(n, scramble=True, seed=8128).random_base2(
            int(np.log2(shots))
        )
    else:
        status = np.random.default_rng(8128).random((shots, n))
    status = status.astype(np.float32)

    backend = tc.backend
    sample_one = lambda u: c.perfect_sampling(status=u)[0]
    sample_batch = backend.jit(backend.vmap(sample_one, vectorized_argnums=0))
    batch = 256
    blocks = []
    for start in range(0, shots, batch):
        block = sample_batch(backend.convert_to_tensor(status[start : start + batch]))
        blocks.append(np.asarray(backend.numpy(block), dtype=np.int8))
    samples = np.concatenate(blocks, axis=0) if blocks else np.empty((0, n), np.int8)
    return {"samples": samples}
