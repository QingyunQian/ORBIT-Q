#!/usr/bin/env python3
"""Run counterbalanced Task 07 reference/candidate pairs in one container."""

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
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_RE = re.compile(r"End-to-end solution time:\s*([0-9.]+)s")
T_CRITICAL_95 = {5: 2.5705818366}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def stats(values: list[float]) -> dict[str, float | int | None]:
    stdev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_stdev": stdev,
        "stderr": stdev / math.sqrt(len(values)) if stdev is not None else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def host_record() -> dict[str, object]:
    commands = {
        "uname": ["uname", "-a"],
        "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
    }
    record: dict[str, object] = {}
    for key, command in commands.items():
        try:
            result = run(command)
            record[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            record[key] = None
    record["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    return record


def image_record(reference: str) -> dict[str, object]:
    result = run(["docker", "image", "inspect", reference])
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"cannot inspect {reference}")
    raw = json.loads(result.stdout)[0]
    return {
        "reference": reference,
        "id": raw.get("Id"),
        "repo_digests": raw.get("RepoDigests") or [],
        "created": raw.get("Created"),
        "architecture": raw.get("Architecture"),
        "os": raw.get("Os"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cpus", type=float, default=6.0)
    parser.add_argument("--memory", default="7g")
    parser.add_argument(
        "--image",
        default="orbitbreakers-expert-benchmarks:tensorcircuit-py311",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0 or args.max_steps <= 0:
        raise SystemExit("repeat and max-steps must be positive")
    timeout = min(args.timeout, 300.0)
    output = args.output.expanduser().resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    sources = {
        "reference": ROOT / "references/task-07/solution_7.py",
        "candidate": ROOT / "src/solutions/task-07/solution_7.py",
    }
    evaluator = ROOT / "tasks/task-07/evaluator/evaluate_7.py"
    sitecustomize = ROOT / "envs/tensorcircuit-py311/sitecustomize.py"
    for path in [*sources.values(), evaluator, sitecustomize]:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")

    image = image_record(args.image)
    host = host_record()
    container_name = f"orbit-task07-matrix-{uuid.uuid4().hex[:10]}"
    started_at = utc_now()
    session_started = time.perf_counter()
    rows: list[dict[str, object]] = []

    staging_root = ROOT / ".tmp"
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task07-matrix-", dir=staging_root) as tmp:
        staging = Path(tmp)
        shutil.copy2(evaluator, staging / "evaluate_7.py")
        environment = staging / "environment"
        environment.mkdir()
        shutil.copy2(sitecustomize, environment / "sitecustomize.py")
        modules: dict[str, str] = {}
        snapshot: dict[str, str] = {}
        for role, source in sources.items():
            module = f"solution_7_{role}"
            modules[role] = module
            target = staging / f"{module}.py"
            shutil.copy2(source, target)
            snapshot[role] = sha256(target)
        snapshot["evaluator"] = sha256(staging / "evaluate_7.py")
        snapshot["sitecustomize"] = sha256(environment / "sitecustomize.py")
        snapshot_sha256 = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()

        start_command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=1g",
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
        ]
        started = run(start_command, timeout=60)
        if started.returncode:
            raise SystemExit(started.stderr.strip() or "container start failed")
        container_id = started.stdout.strip()

        plan: list[tuple[int, int, str, str]] = []
        for pair in range(1, args.repeat + 1):
            roles = (
                ("reference", "candidate")
                if pair % 2
                else ("candidate", "reference")
            )
            order = "->".join(roles)
            for position, role in enumerate(roles, start=1):
                plan.append((pair, position, role, order))

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
                    container_name,
                    "python",
                    "/session/evaluate_7.py",
                    "--solution",
                    modules[role],
                    "--max-steps",
                    str(args.max_steps),
                ]
                wall_started = time.perf_counter()
                timed_out = False
                try:
                    result = run(command, timeout=timeout)
                    stdout, stderr, returncode = (
                        result.stdout,
                        result.stderr,
                        result.returncode,
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    returncode = None
                wall_sec = time.perf_counter() - wall_started
                stdout_path = logs / f"cell-{cell:02d}-{role}.stdout.txt"
                stderr_path = logs / f"cell-{cell:02d}-{role}.stderr.txt"
                stdout_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                match = RUNTIME_RE.search(stdout)
                runtime = float(match.group(1)) if match else None
                passed = (
                    not timed_out
                    and returncode == 0
                    and runtime is not None
                    and "Overall: PASS" in stdout
                )
                row = {
                    "cell_id": f"task07-{cell:02d}",
                    "pair": pair,
                    "position": position,
                    "order": order,
                    "task_id": "07",
                    "solution": role,
                    "repeat_index": pair,
                    "planned_repeats": args.repeat,
                    "max_steps": args.max_steps,
                    "runtime_sec": runtime,
                    "wall_sec": wall_sec,
                    "passed": passed,
                    "timed_out": timed_out,
                    "terminal_status": "SUCCESS" if passed else "FAILED",
                    "returncode": returncode,
                    "engine": "docker",
                    "environment": "tensorcircuit-py311",
                    "environment_image_provenance": image,
                    "container_id": container_id,
                    "container_name": container_name,
                    "cpu_limit": str(args.cpus),
                    "memory_limit": args.memory,
                    "timeout_sec": timeout,
                    "source_sha256": snapshot[role],
                    "evaluator_sha256": snapshot["evaluator"],
                    "staging_snapshot_sha256": snapshot_sha256,
                    "stdout_path": str(stdout_path.relative_to(output)),
                    "stderr_path": str(stderr_path.relative_to(output)),
                    "stdout_sha256": sha256(stdout_path),
                    "stderr_sha256": sha256(stderr_path),
                }
                rows.append(row)
                (output / "checkpoint.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "task_id": "07",
                            "configuration": {
                                "repeat": args.repeat,
                                "max_steps": args.max_steps,
                                "cpus": args.cpus,
                                "memory": args.memory,
                            },
                            "host": host,
                            "image": image,
                            "snapshot": snapshot,
                            "staging_snapshot_sha256": snapshot_sha256,
                            "results": rows,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"cell {cell:02d}/{len(plan)} pair={pair} role={role} "
                    f"runtime={runtime} passed={passed}",
                    flush=True,
                )
        finally:
            run(["docker", "stop", container_name], timeout=60)

    reference = [
        float(row["runtime_sec"])
        for row in rows
        if row["solution"] == "reference" and row["passed"]
    ]
    candidate = [
        float(row["runtime_sec"])
        for row in rows
        if row["solution"] == "candidate" and row["passed"]
    ]
    by_pair: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["passed"]:
            by_pair.setdefault(int(row["pair"]), {})[str(row["solution"])] = float(
                row["runtime_sec"]
            )
    pair_rows = []
    speedups = []
    for pair in sorted(by_pair):
        values = by_pair[pair]
        if set(values) == {"reference", "candidate"}:
            speedup = values["reference"] / values["candidate"]
            speedups.append(speedup)
            pair_rows.append(
                {
                    "pair": pair,
                    "reference_runtime_sec": values["reference"],
                    "candidate_runtime_sec": values["candidate"],
                    "speedup": speedup,
                    "candidate_won": values["candidate"] < values["reference"],
                }
            )

    speedup_stats = stats(speedups)
    ci_low = ci_high = None
    if len(speedups) > 1:
        critical = T_CRITICAL_95.get(len(speedups) - 1)
        if critical is not None:
            radius = critical * float(speedup_stats["stderr"])
            ci_low = float(speedup_stats["mean"]) - radius
            ci_high = float(speedup_stats["mean"]) + radius
    ref_stats, cand_stats = stats(reference), stats(candidate)
    all_passed = len(rows) == 2 * args.repeat and all(row["passed"] for row in rows)
    promotion = (
        all_passed
        and len(speedups) == args.repeat
        and float(cand_stats["mean"]) < float(ref_stats["mean"])
        and float(cand_stats["median"]) < float(ref_stats["median"])
        and sum(row["candidate_won"] for row in pair_rows)
        >= math.ceil(0.8 * args.repeat)
        and ci_low is not None
        and ci_low > 1.0
    )
    report = {
        "schema_version": 1,
        "task_id": "07",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "session_wall_sec": time.perf_counter() - session_started,
        "configuration": {
            "repeat": args.repeat,
            "max_steps": args.max_steps,
            "timeout_sec": timeout,
            "cpus": args.cpus,
            "memory": args.memory,
            "pair_order": "odd reference->candidate; even candidate->reference",
            "fresh_evaluator_process_per_cell": True,
            "single_container": True,
        },
        "host": host,
        "image": image,
        "snapshot": snapshot,
        "staging_snapshot_sha256": snapshot_sha256,
        "results": rows,
        "pairs": pair_rows,
        "summary": {
            "all_cells_passed": all_passed,
            "reference": ref_stats,
            "candidate": cand_stats,
            "ratio_of_means_speedup": (
                float(ref_stats["mean"]) / float(cand_stats["mean"])
                if reference and candidate
                else None
            ),
            "ratio_of_means_improvement_pct": (
                100
                * (float(ref_stats["mean"]) - float(cand_stats["mean"]))
                / float(ref_stats["mean"])
                if reference and candidate
                else None
            ),
            "paired_speedup": speedup_stats,
            "paired_speedup_ci_95": {
                "method": "two-sided Student-t interval on mean pairwise speedup",
                "low": ci_low,
                "high": ci_high,
            },
            "candidate_wins": sum(row["candidate_won"] for row in pair_rows),
            "promotion_rule_passed": promotion,
        },
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    if not promotion:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
