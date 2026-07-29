#!/usr/bin/env python3
"""Compare Task 06 reference/candidate energy, gradient, and one Adam update."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import optax

from profile_reference import CONFIG, ready


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loss_factory(module: Any) -> tuple[Any, Any]:
    hxy, hfield, htarget = module.build_hamiltonians(CONFIG)
    psi0 = module.initial_state(CONFIG)
    params = module.initial_parameters(CONFIG)

    def loss(p: Any) -> Any:
        return module.forward(p, psi0, hxy, hfield, htarget, CONFIG)

    return params, loss


def tree_max_abs_difference(left: Any, right: Any) -> float:
    differences = [
        np.max(np.abs(np.asarray(a) - np.asarray(b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    ]
    return float(max(differences, default=0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/workspace/references/task-06/solution_6.py"),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("/workspace/src/solutions/task-06/solution_6.py"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_module("task06_reference_audit", args.reference)
    candidate = load_module("task06_candidate_audit", args.candidate)
    reference_params, reference_loss = loss_factory(reference)
    candidate_params, candidate_loss = loss_factory(candidate)

    reference_energy, reference_grads = ready(
        jax.jit(jax.value_and_grad(reference_loss))(reference_params)
    )
    candidate_energy, candidate_grads = ready(
        jax.jit(jax.value_and_grad(candidate_loss))(candidate_params)
    )

    optimizer = optax.adam(CONFIG["learning_rate"])
    reference_state = optimizer.init(reference_params)
    candidate_state = optimizer.init(candidate_params)
    reference_updates, reference_state = optimizer.update(
        reference_grads, reference_state, reference_params
    )
    candidate_updates, candidate_state = optimizer.update(
        candidate_grads, candidate_state, candidate_params
    )
    reference_post_params = optax.apply_updates(
        reference_params, reference_updates
    )
    candidate_post_params = optax.apply_updates(
        candidate_params, candidate_updates
    )
    reference_post_energy = ready(jax.jit(reference_loss)(reference_post_params))
    candidate_post_energy = ready(jax.jit(candidate_loss)(candidate_post_params))

    report = {
        "schema_version": 1,
        "task_id": "06",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "initial_parameter_max_abs_error": tree_max_abs_difference(
            reference_params, candidate_params
        ),
        "initial_energy": {
            "reference": float(reference_energy),
            "candidate": float(candidate_energy),
            "abs_error": abs(float(reference_energy) - float(candidate_energy)),
            "tolerance": 5e-5,
        },
        "gradient_max_abs_error": {
            "value": tree_max_abs_difference(reference_grads, candidate_grads),
            "tolerance": 5e-4,
        },
        "post_update_parameter_max_abs_error": tree_max_abs_difference(
            reference_post_params, candidate_post_params
        ),
        "post_update_energy": {
            "reference": float(reference_post_energy),
            "candidate": float(candidate_post_energy),
            "abs_error": abs(
                float(reference_post_energy) - float(candidate_post_energy)
            ),
            "tolerance": 2e-3,
        },
    }
    report["passed"] = bool(
        report["initial_parameter_max_abs_error"] == 0.0
        and report["initial_energy"]["abs_error"]
        <= report["initial_energy"]["tolerance"]
        and report["gradient_max_abs_error"]["value"]
        <= report["gradient_max_abs_error"]["tolerance"]
        and report["post_update_energy"]["abs_error"]
        <= report["post_update_energy"]["tolerance"]
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
