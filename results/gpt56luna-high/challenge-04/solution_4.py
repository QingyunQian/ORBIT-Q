import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.pauliprop import PauliPropagationEngine


def run_solution(config):
    tc.set_backend("jax")
    n = int(config["n_qubits"])
    angle = float(config["entangler_angle"])
    even = tuple((i, i + 1) for i in range(0, n, 2))
    odd = tuple((i, i + 1) for i in range(1, n - 1, 2))
    pp = PauliPropagationEngine(2, 2)
    u = np.asarray(pp.get_ptm_2q(tc.gates.rxx(theta=angle).tensor.reshape(4, 4)))
    umap = [[(j, float(u[i, j])) for j in range(16) if abs(u[i, j]) > 1e-6] for i in range(16)]
    cmap = [[(0, "I")], [(1, "s")], [(2, "s")], [(0, "d"), (3, "a")]]
    ops = []
    for a, b in even + odd:
        ops += [("u", (a, b)), ("c", a), ("c", b)]

    def initial_expectation(word, probe):
        if probe == 0:  # GHZ
            if all(x in (0, 3) for x in word):
                return float(sum(x == 3 for x in word) % 2 == 0)
            if all(x in (1, 2) for x in word):
                ny = sum(x == 2 for x in word)
                return float((-1) ** (ny // 2)) if ny % 2 == 0 else 0.0
            return 0.0
        if probe == 1:  # product of |01>+|10> pairs
            value = 1.0
            for q in range(0, n, 2):
                a, b = word[q : q + 2]
                if (a, b) == (0, 0) or (a == b and a in (1, 2)):
                    continue
                if a == b == 3:
                    value *= -1.0
                    continue
                return 0.0
            return value
        if probe == 2:
            return float(all(x in (0, 3) for x in word))
        return float(all(x in (0, 1) for x in word))

    def build(observables):
        stages = [set(observables)]
        for typ, site in ops[::-1]:
            nxt = set()
            for word in stages[-1]:
                if typ == "c":
                    for j, _ in cmap[word[site]]:
                        w = list(word)
                        w[site] = j
                        nxt.add(tuple(w))
                else:
                    a, b = site
                    for j, _ in umap[4 * word[a] + word[b]]:
                        w = list(word)
                        w[a], w[b] = j // 4, j % 4
                        nxt.add(tuple(w))
            stages.append(nxt)
        words = sorted(set().union(*stages))
        index = {w: i for i, w in enumerate(words)}
        transitions = []
        for (typ, site), current in zip(ops[::-1], stages[:-1]):
            src, dst, q0, q1, factor = [], [], [], [], []
            for word in current:
                if typ == "c":
                    for j, _ in cmap[word[site]]:
                        w = list(word)
                        w[site] = j
                        src.append(index[word]); dst.append(index[tuple(w)])
                        q0.append(word[site]); q1.append(j); factor.append(1.0)
                else:
                    a, b = site
                    for j, f in umap[4 * word[a] + word[b]]:
                        w = list(word)
                        w[a], w[b] = j // 4, j % 4
                        src.append(index[word]); dst.append(index[tuple(w)])
                        q0.append(0); q1.append(0); factor.append(f)
            transitions.append((typ, np.asarray(src, np.int32), np.asarray(dst, np.int32),
                                np.asarray(q0, np.int32), np.asarray(q1, np.int32),
                                np.asarray(factor, np.float32)))
        ex = np.asarray([[initial_expectation(w, p) for p in range(4)] for w in words], np.float32)
        ids = np.asarray([index[w] for w in observables], np.int32)
        return transitions, jnp.asarray(ex), ids, len(observables), len(words)

    def channel_ptm(r):
        p01, p10 = jax.nn.sigmoid(r[0]), jax.nn.sigmoid(r[1])
        z = jnp.zeros((), dtype=jnp.complex64)
        kraus = (
            tc.gates.Gate(jnp.array([[jnp.sqrt(1 - p01), z], [z, jnp.sqrt(1 - p10)]])),
            tc.gates.Gate(jnp.array([[z, jnp.sqrt(p10)], [z, z]])),
            tc.gates.Gate(jnp.array([[z, z], [jnp.sqrt(p01), z]])),
        )
        kt = jnp.stack([k.tensor for k in kraus])
        pauli = jnp.asarray(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]],
                                      [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], np.complex64))
        heis = jnp.einsum("kba,ibc,kcd->iad", jnp.conj(kt), pauli, kt)
        return jnp.real(0.5 * jnp.einsum("jda,iad->ij", pauli, heis))

    def make_evaluator(observables):
        transitions, ex, ids, nobs, nwords = build(observables)

        def evaluate(r):
            value = channel_ptm(r)
            derivative = jax.jacfwd(channel_ptm)(r)
            state = jnp.zeros((nobs, nwords), jnp.float32).at[np.arange(nobs), ids].set(1.0)
            tangent = jnp.zeros((2, nobs, nwords), jnp.float32)
            for typ, src, dst, q0, q1, factor in transitions:
                si, di = jnp.asarray(src), jnp.asarray(dst)
                if typ == "c":
                    qi, qj = jnp.asarray(q0), jnp.asarray(q1)
                    f = value[qi, qj]
                    df = derivative[qi, qj, :]
                else:
                    f = jnp.asarray(factor)
                    df = jnp.zeros((len(src), 2), jnp.float32)
                old = state
                state = jnp.zeros_like(state).at[:, di].add(old[:, si] * f)
                for k in range(2):
                    tangent = tangent.at[k].set(
                        jnp.zeros_like(tangent[k]).at[:, di].add(
                            tangent[k][:, si] * f + old[:, si] * df[:, k]
                        )
                    )
            result = (state @ ex).T
            jacobian = jnp.stack([(tangent[k] @ ex).T for k in range(2)])
            return result, jacobian

        return jax.jit(evaluate)

    singles = [tuple(3 if q == i else 0 for q in range(n)) for i in range(n)]
    singles_eval = make_evaluator(singles)
    parity_eval = make_evaluator([tuple([3] * n)])
    logit = lambda p: jnp.log(p / (1.0 - p))
    [REDACTED]_r = jnp.array([logit(config["[REDACTED]_p01"]), logit(config["[REDACTED]_p10"])])
    r = jnp.array([logit(config["initial_p01"]), logit(config["initial_p10"])])
    target = jnp.concatenate((singles_eval([REDACTED]_r)[0], parity_eval([REDACTED]_r)[0]), axis=1)
    m, v = jnp.zeros(2), jnp.zeros(2)
    history = []
    for step in range(int(config["max_steps"])):
        ys, dys = singles_eval(r)
        yp, dyp = parity_eval(r)
        fitted = jnp.concatenate((ys, yp), axis=1)
        jacobian = jnp.concatenate((dys, dyp), axis=2)
        difference = fitted - target
        loss = jnp.mean(difference * difference)
        grad = 2.0 * jnp.mean(difference[None] * jacobian, axis=(1, 2))
        loss.block_until_ready()
        history.append(float(loss))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1.0 - 0.9 ** (step + 1))
        vh = v / (1.0 - 0.999 ** (step + 1))
        r = r - float(config["learning_rate"]) * mh / (jnp.sqrt(vh) + 1e-8)
    final = jnp.concatenate((singles_eval(r)[0], parity_eval(r)[0]), axis=1)
    return {"loss_history": np.asarray(history),
            "final_probabilities": np.asarray(jax.nn.sigmoid(r), dtype=np.float64),
            "fitted_expectations": np.asarray(final)}
