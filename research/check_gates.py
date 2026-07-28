#!/usr/bin/env python3
"""Fail-closed validation for selected-task research and promotion gates.

Research readiness requires a cited survey and a versioned public dataset for
the selected task. Promotion additionally requires repeated passing reference
measurements. Hidden tuning data, controller attestations, and sealed holdouts
are intentionally outside this policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


TASK_IDS = tuple(f"{number:02d}" for number in range(1, 13))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path, label: str, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{label} is missing")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is unreadable or invalid JSON: {exc}")
    return None


def _command_option(command: object, option: str) -> str | None:
    if not isinstance(command, list):
        return None
    values = [str(value) for value in command]
    try:
        index = values.index(option)
    except ValueError:
        return None
    return values[index + 1] if index + 1 < len(values) else None


def check_survey(root: Path, task_id: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    path = root / "research" / "SURVEY.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False, ["research/SURVEY.md is missing or unreadable"]
    if "**Status: READY**" not in text:
        errors.append("research/SURVEY.md is not marked READY")
    marker = f"## Task {task_id}:"
    if marker not in text:
        errors.append(f"research/SURVEY.md does not cover task {task_id}")
    else:
        section = text.split(marker, 1)[1]
        next_task = re.search(r"^## Task \d{2}:", section, re.MULTILINE)
        if next_task is not None:
            section = section[: next_task.start()]
        if re.search(r"\bTODO\b", section):
            errors.append(
                f"research/SURVEY.md task {task_id} still contains TODO placeholders"
            )
    return not errors, errors


def check_public_dataset(
    root: Path,
    task_id: str,
) -> tuple[bool, str | None, list[str]]:
    errors: list[str] = []
    payload = _read_json(
        root / "datasets" / "public" / "manifest.json",
        "public dataset manifest",
        errors,
    )
    if not isinstance(payload, dict):
        return False, None, errors or ["public dataset manifest must be an object"]

    if payload.get("status") != "ready":
        errors.append("public dataset manifest is not marked ready")
    if payload.get("selected_task_id") != task_id:
        errors.append(
            f"public dataset manifest is not bound to selected task {task_id}"
        )
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append("public dataset manifest has no version")
        version = None
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("public dataset manifest has no cases")
        cases = []

    covered: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"public dataset case {index} is not an object")
            continue
        missing = [
            key
            for key in ("task_id", "case_id", "sha256", "provenance")
            if case.get(key) in (None, "")
        ]
        if missing:
            errors.append(
                f"public dataset case {index} lacks {', '.join(missing)}"
            )
        case_task_id = str(case.get("task_id", ""))
        if case_task_id in TASK_IDS:
            covered.add(case_task_id)
        else:
            errors.append(f"public dataset case {index} has invalid task_id")
        sha256 = case.get("sha256")
        if sha256 not in (None, "") and not SHA256_RE.fullmatch(str(sha256)):
            errors.append(f"public dataset case {index} has invalid sha256")

    if task_id not in covered:
        errors.append(f"public dataset lacks selected task coverage: {task_id}")
    return not errors, version, errors


def check_baseline_report(
    path: Path | None,
    task_id: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if path is None:
        return False, ["a selected-task reference baseline report was not supplied"]
    payload = _read_json(path, "reference baseline report", errors)
    if not isinstance(payload, dict):
        return False, errors or ["reference baseline report must be an object"]

    host = payload.get("host")
    fingerprint = host.get("fingerprint_sha256") if isinstance(host, dict) else None
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        errors.append("reference baseline report lacks a valid host fingerprint")

    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        return False, [*errors, "reference baseline report has no result rows"]

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_ids: set[str] = set()
    environments: set[str] = set()
    timeouts: set[float] = set()
    cpu_limits: set[str] = set()
    memory_limits: set[str] = set()
    source_hashes: dict[str, set[str]] = defaultdict(set)
    evaluator_hashes: dict[str, set[str]] = defaultdict(set)
    snapshot_hashes: dict[str, set[str]] = defaultdict(set)
    container_ids: dict[str, set[str]] = defaultdict(set)
    container_names: dict[str, set[str]] = defaultdict(set)
    repeat_indices: dict[str, list[int]] = defaultdict(list)
    planned_repeats: dict[str, set[int]] = defaultdict(set)
    compatibility_hashes: set[str] = set()

    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            errors.append(f"baseline row {index} is not an object")
            continue
        row: dict[str, Any] = raw_row
        row_task_id = str(row.get("task_id", ""))
        if row_task_id not in TASK_IDS:
            errors.append(f"baseline row {index} has invalid task_id")
            continue
        if row_task_id != task_id:
            continue
        by_task[row_task_id].append(row)
        if row.get("solution") != "reference":
            errors.append(f"baseline row {index} is not a reference run")
        runtime = row.get("runtime_sec")
        if (
            not row.get("passed")
            or row.get("timed_out")
            or row.get("terminal_status") not in (None, "SUCCESS")
            or not isinstance(runtime, (int, float))
            or isinstance(runtime, bool)
            or not math.isfinite(float(runtime))
            or float(runtime) <= 0
        ):
            errors.append(f"baseline row {index} is not a passing measured run")
        if row.get("engine") != "docker":
            errors.append(f"baseline row {index} did not use Docker")

        image = row.get("environment_image_provenance")
        image_id = image.get("id") if isinstance(image, dict) else None
        if isinstance(image_id, str) and image_id:
            image_ids.add(image_id)
        else:
            errors.append(f"baseline row {index} lacks a Docker image ID")
        environment = row.get("environment")
        if isinstance(environment, str) and environment:
            environments.add(environment)
        else:
            errors.append(f"baseline row {index} lacks an environment name")
        try:
            timeout = float(row.get("timeout_sec"))
        except (TypeError, ValueError):
            errors.append(f"baseline row {index} has an invalid timeout")
        else:
            if timeout <= 0 or timeout > 300:
                errors.append(f"baseline row {index} exceeds the 300-second cap")
            timeouts.add(timeout)
        source_hash = row.get("source_sha256")
        if isinstance(source_hash, str) and SHA256_RE.fullmatch(source_hash):
            source_hashes[row_task_id].add(source_hash)
        else:
            errors.append(f"baseline row {index} lacks a valid source hash")

        evaluator_hash = row.get("evaluator_sha256")
        if isinstance(evaluator_hash, str) and SHA256_RE.fullmatch(evaluator_hash):
            evaluator_hashes[row_task_id].add(evaluator_hash)
        else:
            errors.append(f"baseline row {index} lacks a valid evaluator hash")
        snapshot_hash = row.get("staging_snapshot_sha256")
        if isinstance(snapshot_hash, str) and SHA256_RE.fullmatch(snapshot_hash):
            snapshot_hashes[row_task_id].add(snapshot_hash)
        else:
            errors.append(f"baseline row {index} lacks a valid staging snapshot hash")
        compatibility_hash = row.get("compatibility_sha256")
        if (
            isinstance(compatibility_hash, str)
            and SHA256_RE.fullmatch(compatibility_hash)
        ):
            compatibility_hashes.add(compatibility_hash)
        else:
            errors.append(f"baseline row {index} lacks a valid compatibility hash")

        container_id = row.get("shared_container_id")
        container_name = row.get("shared_container_name")
        if isinstance(container_id, str) and container_id:
            container_ids[row_task_id].add(container_id)
        else:
            errors.append(f"baseline row {index} lacks a shared container ID")
        if isinstance(container_name, str) and container_name:
            container_names[row_task_id].add(container_name)
        else:
            errors.append(f"baseline row {index} lacks a shared container name")

        repeat = row.get("repeat")
        planned = row.get("planned_repeats")
        if (
            isinstance(repeat, int)
            and not isinstance(repeat, bool)
            and repeat > 0
        ):
            repeat_indices[row_task_id].append(repeat)
        else:
            errors.append(f"baseline row {index} has an invalid repeat index")
        if (
            isinstance(planned, int)
            and not isinstance(planned, bool)
            and planned >= 6
        ):
            planned_repeats[row_task_id].add(planned)
        else:
            errors.append(
                f"baseline row {index} has fewer than 6 planned repeats"
            )

        exec_command = row.get("command")
        if (
            not isinstance(exec_command, list)
            or [str(value) for value in exec_command[:2]] != ["docker", "exec"]
        ):
            errors.append(
                f"baseline row {index} is not a fresh docker exec measurement"
            )
        start_command = row.get("shared_container_start_command")
        cpu = _command_option(start_command, "--cpus")
        memory = _command_option(start_command, "--memory")
        if cpu is None or memory is None:
            errors.append(f"baseline row {index} lacks pinned resource limits")
        else:
            cpu_limits.add(cpu)
            memory_limits.add(memory)
        if _command_option(start_command, "--network") != "none":
            errors.append(f"baseline row {index} did not disable container network")
        mounts = (
            [str(value) for value in start_command]
            if isinstance(start_command, list)
            else []
        )
        if not any(
            "dst=/session" in value and "readonly" in value for value in mounts
        ):
            errors.append(
                f"baseline row {index} lacks a read-only shared staging mount"
            )

    count = len(by_task.get(task_id, []))
    if count < 6:
        errors.append(
            f"task {task_id} has {count} reference runs; at least 6 required"
        )
    if len(source_hashes.get(task_id, set())) > 1:
        errors.append(f"task {task_id} used multiple reference hashes")
    if len(evaluator_hashes.get(task_id, set())) != 1:
        errors.append(f"task {task_id} did not use one evaluator hash")
    if len(snapshot_hashes.get(task_id, set())) != 1:
        errors.append(f"task {task_id} did not use one staging snapshot")
    if len(container_ids.get(task_id, set())) != 1:
        errors.append(f"task {task_id} did not use one container ID")
    if len(container_names.get(task_id, set())) != 1:
        errors.append(f"task {task_id} did not use one container name")
    planned_values = planned_repeats.get(task_id, set())
    if len(planned_values) != 1:
        errors.append(f"task {task_id} has inconsistent planned repeats")
    else:
        planned = next(iter(planned_values))
        if sorted(repeat_indices.get(task_id, [])) != list(
            range(1, planned + 1)
        ):
            errors.append(
                f"task {task_id} has missing or duplicate repeat indices"
            )

    for label, values in (
        ("Docker image IDs", image_ids),
        ("environments", environments),
        ("timeout scopes", {str(value) for value in timeouts}),
        ("CPU limits", cpu_limits),
        ("memory limits", memory_limits),
        ("compatibility hashes", compatibility_hashes),
    ):
        if len(values) != 1:
            errors.append(f"reference baseline report must use one set of {label}")
    return not errors, errors


def evaluate_gates(
    root: Path,
    *,
    task_id: str,
    baseline_report: Path | None,
) -> dict[str, Any]:
    if task_id not in TASK_IDS:
        raise ValueError(f"unsupported task ID: {task_id}")
    survey_ready, survey_errors = check_survey(root, task_id)
    dataset_ready, public_version, dataset_errors = check_public_dataset(
        root,
        task_id,
    )
    baseline_ready, baseline_errors = check_baseline_report(
        baseline_report,
        task_id,
    )
    checks = {
        "survey": {"ready": survey_ready, "errors": survey_errors},
        "public_dataset": {
            "ready": dataset_ready,
            "version": public_version,
            "errors": dataset_errors,
        },
        "reference_baselines": {
            "ready": baseline_ready,
            "errors": baseline_errors,
        },
    }
    research_ready = all(
        checks[name]["ready"] for name in ("survey", "public_dataset")
    )
    promotion_ready = bool(
        research_ready and checks["reference_baselines"]["ready"]
    )
    return {
        # ``ready`` remains the process exit criterion: an agent may begin
        # proposing candidates once the selected-task survey and public-data
        # gates pass. Valid repeated baselines are required later for promotion.
        "ready": research_ready,
        "task_id": task_id,
        "research_ready": research_ready,
        "promotion_ready": promotion_ready,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate selected-task survey/public-data gates for autoresearch "
            "and the repeated-baseline gate for performance promotion."
        )
    )
    parser.add_argument("--task", choices=TASK_IDS, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_gates(
        args.root.expanduser().resolve(),
        task_id=args.task,
        baseline_report=args.baseline_report,
    )
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["promotion_ready"]:
        print("Autoresearch and runtime-promotion gates passed.")
    elif report["research_ready"]:
        print(
            "Autoresearch survey/public-data gates passed; the runtime-promotion "
            "gate remains closed."
        )
    else:
        print("Autoresearch survey/public-data gates are closed.", file=sys.stderr)
        for name in ("survey", "public_dataset"):
            check = report["checks"][name]
            for error in check["errors"]:
                print(f"- {name}: {error}", file=sys.stderr)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
