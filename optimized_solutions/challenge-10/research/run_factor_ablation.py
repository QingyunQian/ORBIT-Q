#!/usr/bin/env python3
"""Run isolated Task 10 factor ablations against the promoted candidate.

The script derives temporary candidates from the tracked solution with
asserted, exact source replacements. This keeps every quantum representation
and optimizer choice fixed while independently removing either:

* the whole-training ``K.jaxy_scan``; or
* the fused ``RX -> RZ -> RY`` one-site application.

The generated source lives in a temporary directory and is never promoted as
an optimized solution. This runner supplies the counterbalanced, fresh-process
Docker measurements under an explicit six-CPU resource limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/solutions/task-10/solution_10.py"
EVALUATOR = ROOT / "tasks/task-10/evaluator/evaluate_10.py"
SITECUSTOMIZE = ROOT / "envs/tensorcircuit-py311/sitecustomize.py"
RUNTIME_RE = re.compile(r"End-to-end solution time:\s*([0-9.]+)s")

SCAN_BLOCK = """\
    @K.jit
    def optimize(current, state):
        return K.jaxy_scan(train, (current, state), K.arange(max_steps))

    (params, _), energy_history = optimize(params, opt_state)
"""

PYTHON_LOOP_BLOCK = """\
    @K.jit
    def train_once(carry):
        return train(carry, None)

    carry = (params, opt_state)
    history = []
    for _ in range(max_steps):
        carry, value = train_once(carry)
        history.append(value)
    (params, _), energy_history = carry, K.stack(history)
"""

FUSED_ROTATION_BLOCK = """\
            tensors = [
                _apply_one_site(tensor, _rotation(params[layer, q]))
                for q, tensor in enumerate(tensors)
            ]
"""

UNFUSED_ROTATION_BLOCK = """\
            rotated = []
            for q, tensor in enumerate(tensors):
                theta = params[layer, q]
                tensor = _apply_one_site(
                    tensor, tc.gates.rx(theta=theta[0]).tensor
                )
                tensor = _apply_one_site(
                    tensor, tc.gates.rz(theta=theta[1]).tensor
                )
                tensor = _apply_one_site(
                    tensor, tc.gates.ry(theta=theta[2]).tensor
                )
                rotated.append(tensor)
            tensors = rotated
"""


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one replacement target, found {source.count(old)}"
        )
    return source.replace(old, new)


def derive_source(variant: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    if variant == "no-scan":
        source = replace_once(source, SCAN_BLOCK, PYTHON_LOOP_BLOCK)
    elif variant == "unfused-rotations":
        source = replace_once(
            source,
            FUSED_ROTATION_BLOCK,
            UNFUSED_ROTATION_BLOCK,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    compile(source, f"<task-10-{variant}>", "exec")
    return source


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def summary(values: list[float]) -> dict[str, float | int | None]:
    stdev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_stdev": stdev,
        "stderr": stdev / math.sqrt(len(values)) if stdev is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        choices=("no-scan", "unfused-rotations"),
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cpus", type=float, default=6.0)
    parser.add_argument("--memory", default="7g")
    parser.add_argument(
        "--image",
        default="orbitbreakers-expert-benchmarks:tensorcircuit-py311",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    (ROOT / ".tmp").mkdir(exist_ok=True)
    output = args.output.expanduser().resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    container = f"orbit-task10-ablation-{uuid.uuid4().hex[:10]}"
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="task10-ablation-", dir=ROOT / ".tmp") as tmp:
        staging = Path(tmp)
        shutil.copy2(EVALUATOR, staging / "evaluate_10.py")
        environment = staging / "environment"
        environment.mkdir()
        shutil.copy2(SITECUSTOMIZE, environment / "sitecustomize.py")
        optimized = staging / "solution_10_optimized.py"
        ablation = staging / "solution_10_ablation.py"
        shutil.copy2(SOURCE, optimized)
        ablation.write_text(derive_source(args.variant), encoding="utf-8")
        source_hashes = {
            "optimized": sha256(optimized),
            "ablation": sha256(ablation),
            "evaluator": sha256(staging / "evaluate_10.py"),
        }

        start = run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container,
                "--network",
                "none",
                "--mount",
                f"type=bind,src={staging.resolve()},dst=/session,readonly",
                "--workdir",
                "/session",
                "--env",
                "NUMBA_DISABLE_JIT=1",
                "--env",
                "PYTHONPATH=/session:/session/environment",
                "--cpus",
                str(args.cpus),
                "--memory",
                args.memory,
                args.image,
                "tail",
                "-f",
                "/dev/null",
            ],
            timeout=60,
        )
        if start.returncode:
            raise SystemExit(start.stderr.strip() or "container start failed")

        plan = []
        for pair in range(1, args.repeat + 1):
            roles = (
                ("optimized", "ablation")
                if pair % 2
                else ("ablation", "optimized")
            )
            for position, role in enumerate(roles, start=1):
                plan.append((pair, position, role, "->".join(roles)))

        try:
            for cell, (pair, position, role, order) in enumerate(plan, start=1):
                command = [
                    "docker",
                    "exec",
                    "--workdir",
                    "/session",
                    "--env",
                    "NUMBA_DISABLE_JIT=1",
                    "--env",
                    "PYTHONPATH=/session:/session/environment",
                    container,
                    "python",
                    "/session/evaluate_10.py",
                    "--solution",
                    f"solution_10_{role}",
                ]
                started = time.perf_counter()
                result = run(command, timeout=min(args.timeout, 300))
                wall_sec = time.perf_counter() - started
                stdout_path = logs / f"cell-{cell:02d}-{role}.stdout.txt"
                stderr_path = logs / f"cell-{cell:02d}-{role}.stderr.txt"
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
                match = RUNTIME_RE.search(result.stdout)
                runtime = float(match.group(1)) if match else None
                passed = (
                    result.returncode == 0
                    and runtime is not None
                    and "Overall: PASS" in result.stdout
                )
                rows.append(
                    {
                        "pair": pair,
                        "position": position,
                        "order": order,
                        "role": role,
                        "runtime_sec": runtime,
                        "wall_sec": wall_sec,
                        "passed": passed,
                        "returncode": result.returncode,
                        "source_sha256": source_hashes[role],
                        "stdout_sha256": sha256(stdout_path),
                    }
                )
                print(
                    f"cell {cell:02d}/{len(plan)} pair={pair} role={role} "
                    f"runtime={runtime} passed={passed}",
                    flush=True,
                )
        finally:
            run(["docker", "stop", container], timeout=60)

    optimized_values = [
        float(row["runtime_sec"])
        for row in rows
        if row["role"] == "optimized" and row["passed"]
    ]
    ablation_values = [
        float(row["runtime_sec"])
        for row in rows
        if row["role"] == "ablation" and row["passed"]
    ]
    if len(optimized_values) != args.repeat or len(ablation_values) != args.repeat:
        raise SystemExit("one or more ablation cells failed")
    paired = []
    for pair in range(1, args.repeat + 1):
        pair_rows = {row["role"]: row for row in rows if row["pair"] == pair}
        paired.append(
            float(pair_rows["ablation"]["runtime_sec"])
            / float(pair_rows["optimized"]["runtime_sec"])
        )
    report = {
        "schema_version": 1,
        "task_id": "10",
        "variant": args.variant,
        "comparison": "ablation / promoted optimized candidate",
        "repeat": args.repeat,
        "cpus": args.cpus,
        "memory": args.memory,
        "image": args.image,
        "source_hashes": source_hashes,
        "optimized": summary(optimized_values),
        "ablation": summary(ablation_values),
        "paired_slowdown": summary(paired),
        "pair_wins_for_optimized": sum(value > 1.0 for value in paired),
        "rows": rows,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in (
        "variant",
        "optimized",
        "ablation",
        "paired_slowdown",
        "pair_wins_for_optimized",
    )}, indent=2))


if __name__ == "__main__":
    main()
