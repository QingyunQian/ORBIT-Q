#!/usr/bin/env python3
"""Compare Task 06 termwise and TensorCircuit-native sparse actions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
import tensorcircuit as tc

from profile_reference import CONFIG, load_reference, ready


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def pauli_terms(n: int) -> tuple[list[list[int]], list[float], list[list[int]], list[float], list[list[int]], list[float]]:
    xy_ls, xy_w = [], []
    for i in range(n - 1):
        for p in (1, 2):
            s = [0] * n
            s[i] = p
            s[i + 1] = p
            xy_ls.append(s)
            xy_w.append(1.0)

    field_ls, field_w = [], []
    for i in range(n):
        s = [0] * n
        s[i] = 3
        field_ls.append(s)
        field_w.append((-1.0) ** i)

    target_ls, target_w = [], []
    for i in range(n - 1):
        for p, coefficient in ((1, 0.7), (2, 0.7), (3, 1.1)):
            s = [0] * n
            s[i] = p
            s[i + 1] = p
            target_ls.append(s)
            target_w.append(coefficient)
    for i in range(n):
        s = [0] * n
        s[i] = 3
        target_ls.append(s)
        target_w.append(0.25 * ((-1.0) ** i))
    return xy_ls, xy_w, field_ls, field_w, target_ls, target_w


def measure(
    function: Callable[[Any], Any],
    state: Any,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    value = ready(compiled(state))
    first_sec = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = ready(compiled(state))
        samples.append(time.perf_counter() - started)
    return value, {
        "first_compile_and_exec_sec": first_sec,
        "steady_runtime_sec": samples,
        "steady_mean_sec": statistics.mean(samples),
        "steady_median_sec": statistics.median(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/workspace/references/task-06/solution_6.py"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    hxy_mvp, hfield_mvp, htarget_mvp = reference.build_hamiltonians(CONFIG)
    state = reference.initial_state(CONFIG)
    terms = pauli_terms(CONFIG["n_qubits"])
    xy_ls, xy_w, field_ls, field_w, target_ls, target_w = terms

    started = time.perf_counter()
    hxy_coo = tc.quantum.PauliStringSum2COO(xy_ls, xy_w, numpy=True)
    hfield_coo = tc.quantum.PauliStringSum2COO(
        field_ls, field_w, numpy=True
    )
    htarget_coo = tc.quantum.PauliStringSum2COO(
        target_ls, target_w, numpy=True
    )
    scipy_build_sec = time.perf_counter() - started
    started = time.perf_counter()
    hxy_sparse = K.coo_sparse_matrix_from_numpy(hxy_coo)
    hfield_sparse = K.coo_sparse_matrix_from_numpy(hfield_coo)
    htarget_sparse = K.coo_sparse_matrix_from_numpy(htarget_coo)
    backend_convert_sec = time.perf_counter() - started

    j_value = K.cast(K.tanh(K.convert_to_tensor(0.1)), tc.dtypestr)
    d_value = K.cast(K.tanh(K.convert_to_tensor(0.1)), tc.dtypestr)

    def termwise_analog(y: Any) -> Any:
        return j_value * hxy_mvp(y) + d_value * hfield_mvp(y)

    def sparse_analog(y: Any) -> Any:
        return j_value * K.sparse_dense_matmul(
            hxy_sparse, y
        ) + d_value * K.sparse_dense_matmul(hfield_sparse, y)

    termwise_value, termwise = measure(termwise_analog, state, args.repeats)
    sparse_value, sparse = measure(sparse_analog, state, args.repeats)
    target_termwise_value, target_termwise = measure(
        htarget_mvp, state, args.repeats
    )
    target_sparse_value, target_sparse = measure(
        lambda y: K.sparse_dense_matmul(htarget_sparse, y),
        state,
        args.repeats,
    )

    report = {
        "schema_version": 1,
        "task_id": "06",
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "state_shape": list(state.shape),
        "construction": {
            "tensorcircuit_coo_numpy_sec": scipy_build_sec,
            "backend_sparse_convert_sec": backend_convert_sec,
            "xy_nnz": int(hxy_coo.nnz),
            "field_nnz": int(hfield_coo.nnz),
            "target_nnz": int(htarget_coo.nnz),
        },
        "analog_termwise": termwise,
        "analog_sparse": sparse,
        "analog_max_abs_error": float(
            np.max(np.abs(np.asarray(termwise_value) - np.asarray(sparse_value)))
        ),
        "analog_steady_speedup": (
            termwise["steady_mean_sec"] / sparse["steady_mean_sec"]
        ),
        "target_termwise": target_termwise,
        "target_sparse": target_sparse,
        "target_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(target_termwise_value)
                    - np.asarray(target_sparse_value)
                )
            )
        ),
        "target_steady_speedup": (
            target_termwise["steady_mean_sec"] / target_sparse["steady_mean_sec"]
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
