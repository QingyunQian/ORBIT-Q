#!/usr/bin/env python3
"""Validate the measured-ancilla RZZ to data-RZ identity for both branches."""

from __future__ import annotations

import json

import numpy as np
import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype("complex64")
TOLERANCE = 1e-7


def main():
    data = np.asarray([0.6 + 0.2j, -0.3 + 0.7j], dtype=np.complex64)
    data /= np.linalg.norm(data)
    cases = []
    for bit, theta in ((0, -0.37), (0, 1.23), (1, -0.37), (1, 1.23)):
        ancilla = np.eye(2, dtype=np.complex64)[bit]
        input_state = np.kron(ancilla, data)
        rzz = np.asarray(K.numpy(tc.gates.rzz(theta=theta).tensor)).reshape(4, 4)
        rz = np.asarray(
            K.numpy(tc.gates.rz(theta=(1 - 2 * bit) * theta).tensor)
        ).reshape(2, 2)
        lhs = rzz @ input_state
        rhs = np.kron(ancilla, rz @ data)
        error = float(np.max(np.abs(lhs - rhs)))
        cases.append({"bit": bit, "theta": theta, "max_abs_error": error})

    maximum = max(case["max_abs_error"] for case in cases)
    report = {
        "schema_version": 1,
        "task_id": "07",
        "experiment": "e02",
        "identity": "RZZ(theta_b)|b,psi> = |b> RZ((1-2b)theta_b)|psi>",
        "dtype": "complex64",
        "tolerance": TOLERANCE,
        "cases": cases,
        "max_abs_error": maximum,
        "passed": maximum <= TOLERANCE,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
