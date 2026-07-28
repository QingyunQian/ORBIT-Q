import numpy as np, tensornetwork as tn, tensorcircuit as tc, cotengra as ctg, omeco

from tensorcircuit.cons import OMEOptimizer, _extract_topology


def _probability_engine(config):
    tc.set_backend("numpy")
    tc.set_dtype("complex64")
    side = int(config["grid_side"])
    nq = int(config["n_qubits"])
    circuit = tc.Circuit(nq)
    owners = list(range(nq))
    split = {"max_singular_values": 2, "fixed_choice": 1}

    def one(gate, q, theta):
        start = len(circuit._nodes)
        gate(q, theta=theta)
        owners.extend([q] * (len(circuit._nodes) - start))

    def two(gate, q0, q1, theta):
        start = len(circuit._nodes)
        gate(q0, q1, theta=theta, split=split)
        owners.extend([q0, q1][: len(circuit._nodes) - start])

    for r in range(side):
        for col in range(side):
            theta = (
                config["ry_offset"]
                + config["ry_row_sin_scale"]
                * np.sin(config["ry_row_sin_frequency"] * (r + 1))
                + config["ry_col_cos_scale"]
                * np.cos(config["ry_col_cos_frequency"] * (col + 1))
                + config["ry_diag_sin_scale"]
                * np.sin(config["ry_diag_sin_frequency"] * (r + col + 2))
            )
            one(circuit.ry, side * r + col, theta)

    edge = 0
    for r in range(side):
        for col in range(side - 1):
            theta = (
                config["rzz_offset"]
                + config["rzz_edge_sin_scale"]
                * np.sin(config["rzz_edge_sin_frequency"] * (edge + 1))
                + config["rzz_site_cos_scale"]
                * np.cos(config["rzz_site_cos_frequency"] * (2 * r + col + 1))
            )
            two(circuit.rzz, side * r + col, side * r + col + 1, theta)
            edge += 1

    edge = 0
    for r in range(side - 1):
        for col in range(side):
            theta = (
                config["rxx_offset"]
                + config["rxx_edge_cos_scale"]
                * np.cos(config["rxx_edge_cos_frequency"] * (edge + 1))
                + config["rxx_site_sin_scale"]
                * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * col + 1))
            )
            two(circuit.rxx, side * r + col, side * (r + 1) + col, theta)
            edge += 1

    for r in range(side):
        for col in range(side):
            theta = (
                config["rx_offset"]
                + config["rx_row_cos_scale"]
                * np.cos(config["rx_row_cos_frequency"] * (r + 1))
                - config["rx_col_sin_scale"]
                * np.sin(config["rx_col_sin_frequency"] * (col + 1))
                + config["rx_diag_cos_scale"]
                * np.cos(config["rx_diag_cos_frequency"] * (r + col + 2))
            )
            one(circuit.rx, side * r + col, theta)

    ket, ek = circuit._copy()
    bra, eb = circuit._copy(conj=True)
    nodes = ket + bra
    node_owners = owners + owners
    delta = np.zeros((2, 2, 2), dtype=np.complex64)
    delta[0, 0, 0] = delta[1, 1, 1] = 1
    outputs = []
    for q in range(nq):
        node = tn.Node(delta)
        ek[q] ^ node[0]
        eb[q] ^ node[1]
        nodes.append(node)
        node_owners.append(q)
        outputs.append(node[2])

    sites = []
    for q in range(nq):
        group = [x for x, owner in zip(nodes, node_owners) if owner == q]
        while len(group) > 1:
            members = set(group)
            for i, node in enumerate(group):
                other = next(
                    (
                        e.node2 if e.node1 is node else e.node1
                        for e in node.edges
                        if not e.is_dangling()
                        and (e.node2 if e.node1 is node else e.node1) in members
                        and (e.node2 if e.node1 is node else e.node1) is not node
                    ),
                    None,
                )
                if other is not None:
                    j = group.index(other)
                    merged = tn.contract_between(node, other)
                    group = [
                        x for k, x in enumerate(group) if k not in (i, j)
                    ] + [merged]
                    break
        sites.append(group[0])

    for r in range(side):
        for col in range(side - 1):
            a, b = sites[side * r + col : side * r + col + 2]
            tn.flatten_edges(list(tn.get_shared_edges(a, b)))
    for r in range(side - 1):
        for col in range(side):
            a, b = sites[side * r + col], sites[side * (r + 1) + col]
            tn.flatten_edges(list(tn.get_shared_edges(a, b)))

    score = omeco.ScoreFunction(
        tc_weight=1.0, sc_weight=0.1, rw_weight=8.0, sc_target=24.0
    )
    pair_contractor = tc.get_contractor(
        "custom",
        optimizer=OMEOptimizer(omeco.TreeSA(ntrials=4, niters=16, score=score)),
        preprocessing=True,
        use_primitives=False,
    )
    row_triples = []
    for r in range(side - 2):
        copied, edges = tn.copy(sites)
        selected = set(range(side * r, side * (r + 3)))
        open_edges = [edges[outputs[q]] for q in sorted(selected)]
        traces = []
        for q in range(nq):
            if q not in selected:
                node = tn.Node(np.ones(2, dtype=np.complex64))
                edges[outputs[q]] ^ node[0]
                traces.append(node)
        value = pair_contractor(list(copied.values()) + traces, output_edge_order=open_edges)
        joint = np.maximum(np.real(value.tensor).reshape(128, 128, 128), 0)
        row_triples.append(joint / joint.sum())

    selectors = []
    for q in range(nq):
        node = tn.Node(np.ones(2, dtype=np.complex64))
        outputs[q] ^ node[0]
        selectors.append(node)
    arrays, inputs, output, sizes = _extract_topology(sites + selectors)
    locations = {
        q: next(i for i, x in enumerate(arrays) if x is node.tensor)
        for q, node in enumerate(selectors)
    }
    optimizer = OMEOptimizer(omeco.TreeSA(ntrials=6, niters=24, score=score))
    path = optimizer(inputs, output, sizes)
    tree = ctg.ContractionTree.from_path(inputs, output, sizes, path=path)
    contract = tree.get_contractor(implementation="autoray")
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.complex64)

    def probability(bits):
        args = list(arrays)
        for q, bit in enumerate(bits):
            args[locations[q]] = vectors[bit]
        return max(float(np.real(contract(*args))), 1e-30)

    return probability, row_triples


def run_solution(config):
    probability, triples = _probability_engine(config)
    nq = int(config["n_qubits"])
    shots = int(config["n_samples"])
    rng = np.random.default_rng(8049)
    pairs = [x.sum(2) for x in triples]
    conditionals = [x / np.maximum(x.sum(2, keepdims=True), 1e-30) for x in triples]
    cdfs = [np.cumsum(x, axis=2) for x in conditionals]
    first_cdf = np.cumsum(pairs[0].reshape(-1))
    for cdf in cdfs:
        cdf[..., -1] = 1
    first_cdf[-1] = 1
    logt = [np.log(np.maximum(x, 1e-30)) for x in triples]
    logj = [np.log(np.maximum(x, 1e-30)) for x in pairs]

    def proposal():
        rows = np.empty(7, dtype=np.int16)
        rows[:2] = divmod(np.searchsorted(first_cdf, rng.random()), 128)
        for r in range(5):
            rows[r + 2] = np.searchsorted(cdfs[r][rows[r], rows[r + 1]], rng.random())
        return ((rows[:, None] >> np.arange(6, -1, -1)) & 1).reshape(-1).astype(np.int8)

    def weight(bits):
        rows = (bits.reshape(7, 7) * (1 << np.arange(6, -1, -1))).sum(1)
        logq = logj[0][rows[0], rows[1]]
        for r in range(5):
            logq += logt[r][rows[r], rows[r + 1], rows[r + 2]]
            logq -= logj[r][rows[r], rows[r + 1]]
        return np.log(probability(bits)) - logq

    current = proposal()
    current_w = weight(current)
    samples = np.empty((shots, nq), dtype=np.int8)
    burn = 256
    for step in range(burn + 2 * shots):
        candidate = proposal()
        candidate_w = weight(candidate)
        if np.log(rng.random()) < min(0.0, candidate_w - current_w):
            current, current_w = candidate, candidate_w
        if step >= burn and (step - burn) % 2 == 1:
            samples[(step - burn) // 2] = current
    return {"samples": samples}
