from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .manifest import (
    ManifestError,
    TaskConfig,
    discover_tasks,
    load_bench_config,
    verify_benchmark,
)
from .runner import (
    BenchmarkError,
    DockerTaskSession,
    build_environment,
    doctor_environment,
    dry_run_command,
    dry_run_container_command,
    ensure_docker_image,
    ensure_output_paths_available,
    inspect_docker_image,
    output_paths,
    run_repetition,
    staged_module_name,
    write_report,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Run and compare the ORBIT-Q human expert solutions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered tasks.")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    subparsers.add_parser("verify", help="Validate benchmark manifests and files.")

    env_parser = subparsers.add_parser("env", help="Inspect or build environments.")
    env_subparsers = env_parser.add_subparsers(dest="env_command", required=True)
    env_subparsers.add_parser("doctor", help="Check Docker and benchmark environments.")
    build_parser = env_subparsers.add_parser("build", help="Build a Docker environment.")
    build_parser.add_argument(
        "name",
        nargs="?",
        help="Environment name. Defaults to default_environment.",
    )
    build_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="Run one task or the complete 12-task expert benchmark.",
    )
    run_parser.add_argument("task", help="Task id (01, 1, challenge-01) or all.")
    run_parser.add_argument(
        "--solution",
        default="optimized",
        help="Registered solution name (default: optimized).",
    )
    run_parser.add_argument(
        "--candidate",
        type=Path,
        help="Explicit candidate Python file; overrides --solution.",
    )
    run_parser.add_argument(
        "--compare-to",
        metavar="NAME",
        help=(
            "Registered reference solution. Runs paired, counterbalanced "
            "measurements and reports improvement and speedup."
        ),
    )
    run_parser.add_argument(
        "--repeat",
        type=int,
        help="Measurements per solution (default: bench.toml default_repeats).",
    )
    run_parser.add_argument(
        "--engine",
        choices=("docker", "local"),
        default="docker",
    )
    run_parser.add_argument("--timeout", type=float)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--no-build", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    return parser


def _normalize_task_selector(value: str) -> str:
    value = value.strip().lower()
    if value == "all":
        return value
    if value.startswith("challenge-"):
        value = value.removeprefix("challenge-")
    try:
        number = int(value)
    except ValueError as exc:
        raise BenchmarkError(f"Invalid task selector: {value!r}") from exc
    if number < 1 or number > 99:
        raise BenchmarkError(f"Invalid task selector: {value!r}")
    return f"{number:02d}"


def _select_tasks(tasks: Sequence[TaskConfig], selector: str) -> list[TaskConfig]:
    normalized = _normalize_task_selector(selector)
    if normalized == "all":
        return list(tasks)
    selected = [task for task in tasks if task.id == normalized]
    if not selected:
        raise BenchmarkError(f"Task {normalized} is not registered")
    return selected


def _list_payload(tasks: Sequence[TaskConfig]) -> list[dict[str, object]]:
    return [
        {
            "id": task.id,
            "title": task.title,
            "environment": task.environment,
            "module": task.module,
            "timeout_sec": task.timeout_sec,
            "solutions": [
                {
                    "name": solution.name,
                    "kind": solution.kind,
                    "path": str(solution.path),
                }
                for solution in task.solutions
            ],
        }
        for task in tasks
    ]


def _print_task_list(tasks: Sequence[TaskConfig]) -> None:
    for task in tasks:
        solutions = ", ".join(solution.name for solution in task.solutions)
        print(
            f"{task.id}  {task.title}  "
            f"[env={task.environment}; solutions={solutions}]"
        )


def _resolve_explicit_candidate(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise BenchmarkError(f"Candidate file not found: {path}")
    if path.suffix != ".py":
        raise BenchmarkError(f"Candidate must be a Python file: {path}")
    return path


def _measurement_order(
    repeat_index: int,
    candidate_target: tuple[Path, str, str],
    reference_target: tuple[Path, str, str] | None,
) -> tuple[
    str | None,
    tuple[tuple[str | None, tuple[Path, str, str], int | None], ...],
]:
    if reference_target is None:
        return None, ((None, candidate_target, None),)
    if repeat_index % 2 == 1:
        return (
            "reference->candidate",
            (
                ("reference", reference_target, 1),
                ("candidate", candidate_target, 2),
            ),
        )
    return (
        "candidate->reference",
        (
            ("candidate", candidate_target, 1),
            ("reference", reference_target, 2),
        ),
    )


def _print_comparison(summary: dict[str, object]) -> None:
    comparison = summary.get("comparison")
    if not isinstance(comparison, dict):
        return
    print(
        "Comparison: "
        f"{comparison['candidate_solution']} vs "
        f"{comparison['reference_solution']}",
        file=sys.stderr,
    )
    task_comparisons = comparison.get("tasks", {})
    if not isinstance(task_comparisons, dict):
        return
    for task_id, row in sorted(task_comparisons.items()):
        if not isinstance(row, dict):
            continue
        reference_mean = row.get("reference_mean")
        reference_stderr = row.get("reference_stderr")
        candidate_mean = row.get("candidate_mean")
        candidate_stderr = row.get("candidate_stderr")
        improvement = row.get("paired_improvement_mean")
        improvement_stderr = row.get("paired_improvement_stderr")
        speedup = row.get("speedup_mean")
        speedup_stderr = row.get("speedup_stderr")

        def metric(value: object, suffix: str = "") -> str:
            return (
                f"{value:.6f}{suffix}"
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                else "n/a"
            )

        print(
            f"[{task_id}] reference mean={metric(reference_mean, 's')} "
            f"± {metric(reference_stderr, 's')}; "
            f"candidate mean={metric(candidate_mean, 's')} "
            f"± {metric(candidate_stderr, 's')}; "
            f"paired improvement={metric(improvement, '%')} "
            f"± {metric(improvement_stderr, '%')}; "
            f"paired speedup={metric(speedup, 'x')} "
            f"± {metric(speedup_stderr, 'x')}",
            file=sys.stderr,
        )


def _handle_run(
    args: argparse.Namespace,
    *,
    root: Path,
    argv: Sequence[str],
) -> int:
    config = load_bench_config(root)
    tasks = discover_tasks(root, config)
    selected = _select_tasks(tasks, args.task)
    repeat_count = (
        args.repeat if args.repeat is not None else config.default_repeats
    )
    if repeat_count < 1:
        raise BenchmarkError("--repeat must be at least 1")
    if args.timeout is not None and args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    if args.timeout is not None and args.timeout > 300:
        raise BenchmarkError("--timeout must not exceed the 300 second hard limit")
    if args.candidate is not None and len(selected) != 1:
        raise BenchmarkError("--candidate can only be used with one task")

    explicit_candidate = (
        _resolve_explicit_candidate(args.candidate)
        if args.candidate is not None
        else None
    )
    # task, candidate target, optional reference target; target is
    # (source path, solution name, solution kind).
    resolved: list[
        tuple[
            TaskConfig,
            tuple[Path, str, str],
            tuple[Path, str, str] | None,
        ]
    ] = []
    for task in selected:
        if explicit_candidate is not None:
            candidate_target = (
                explicit_candidate,
                "candidate",
                "explicit-candidate",
            )
        else:
            solution = task.solution(args.solution)
            if not solution.path.is_file():
                raise BenchmarkError(
                    f"Registered solution file not found: {solution.path}"
                )
            candidate_target = (
                solution.path,
                solution.name,
                solution.kind,
            )
        reference_target = None
        if args.compare_to is not None:
            reference = task.solution(args.compare_to)
            if not reference.path.is_file():
                raise BenchmarkError(
                    f"Registered reference file not found: {reference.path}"
                )
            reference_target = (
                reference.path,
                reference.name,
                reference.kind,
            )
            if reference.name == candidate_target[1]:
                raise BenchmarkError(
                    "--compare-to must name a different solution than the candidate"
                )
        resolved.append((task, candidate_target, reference_target))

    if args.dry_run:
        plans = []
        for task, candidate_target, reference_target in resolved:
            environment = config.environments[task.environment]
            container_command = (
                dry_run_container_command(
                    root=root,
                    task=task,
                    environment=environment,
                )
                if args.engine == "docker"
                else None
            )
            if reference_target is None:
                source, solution_name, solution_kind = candidate_target
                module = staged_module_name(task, solution_name, source)
                plans.append(
                    {
                        "task_id": task.id,
                        "solution": solution_name,
                        "solution_kind": solution_kind,
                        "source": str(source),
                        "repeat": repeat_count,
                        "engine": args.engine,
                        "timeout_sec": args.timeout or task.timeout_sec,
                        "container_start_command": container_command,
                        "command": dry_run_command(
                            root=root,
                            task=task,
                            environment=environment,
                            engine=args.engine,
                            module=module if args.engine == "docker" else None,
                        ),
                    }
                )
                continue
            for repeat_index in range(1, repeat_count + 1):
                pair_order, ordered_targets = _measurement_order(
                    repeat_index,
                    candidate_target,
                    reference_target,
                )
                for role, target, pair_position in ordered_targets:
                    source, solution_name, solution_kind = target
                    module = staged_module_name(task, solution_name, source)
                    plans.append(
                        {
                            "task_id": task.id,
                            "solution": solution_name,
                            "solution_kind": solution_kind,
                            "comparison_role": role,
                            "pair_order": pair_order,
                            "pair_position": pair_position,
                            "source": str(source),
                            "repeat": repeat_index,
                            "repeat_count": repeat_count,
                            "engine": args.engine,
                            "timeout_sec": args.timeout or task.timeout_sec,
                            "container_start_command": container_command,
                            "command": dry_run_command(
                                root=root,
                                task=task,
                                environment=environment,
                                engine=args.engine,
                                module=module if args.engine == "docker" else None,
                            ),
                        }
                    )
        print(json.dumps({"dry_run": True, "plans": plans}, indent=2))
        return 0

    report_path, summary_path, logs_dir = output_paths(root, args.output)
    if args.output is not None:
        ensure_output_paths_available(report_path, summary_path, logs_dir)

    image_provenance: dict[str, dict[str, object]] = {}
    if args.engine == "docker":
        for task in selected:
            if task.environment in image_provenance:
                continue
            environment = config.environments[task.environment]
            image_provenance[task.environment] = ensure_docker_image(
                root,
                environment,
                no_build=args.no_build,
            )

    results: list[dict[str, object]] = []
    for task, candidate_target, reference_target in resolved:
        environment = config.environments[task.environment]
        timeout_sec = min(float(args.timeout or task.timeout_sec), 300.0)

        def record_result(result: dict[str, object]) -> None:
            results.append(result)
            runtime = result.get("runtime_sec")
            outcome = "PASS" if result.get("passed") else "FAIL"
            runtime_text = (
                f"{runtime:.3f}s" if isinstance(runtime, float) else "n/a"
            )
            print(
                f"[{task.id}] {result['solution']} {outcome} "
                f"runtime={runtime_text}",
                file=sys.stderr,
            )

        if args.engine == "docker":
            docker_targets = {
                candidate_target[1]: candidate_target[0],
            }
            if reference_target is not None:
                docker_targets[reference_target[1]] = reference_target[0]
            with DockerTaskSession(
                root=root,
                task=task,
                environment=environment,
                targets=docker_targets,
                image_provenance=image_provenance.get(task.environment),
            ) as session:
                task_timed_out = False
                for repeat_index in range(1, repeat_count + 1):
                    pair_order, run_targets = _measurement_order(
                        repeat_index,
                        candidate_target,
                        reference_target,
                    )
                    for comparison_role, target, pair_position in run_targets:
                        source, solution_name, solution_kind = target
                        print(
                            f"[{task.id}] {solution_name} repeat "
                            f"{repeat_index}/{repeat_count}",
                            file=sys.stderr,
                        )
                        result = session.run_measurement(
                            solution_name=solution_name,
                            solution_kind=solution_kind,
                            comparison_role=comparison_role,
                            timeout_sec=timeout_sec,
                            repeat_index=repeat_index,
                            planned_repeats=repeat_count,
                            logs_dir=logs_dir,
                            pair_order=pair_order,
                            pair_position=pair_position,
                        )
                        record_result(result)
                        if result.get("timed_out"):
                            task_timed_out = True
                            break
                    if task_timed_out:
                        break
            continue

        for repeat_index in range(1, repeat_count + 1):
            pair_order, run_targets = _measurement_order(
                repeat_index,
                candidate_target,
                reference_target,
            )
            for comparison_role, target, pair_position in run_targets:
                source, solution_name, solution_kind = target
                print(
                    f"[{task.id}] {solution_name} repeat "
                    f"{repeat_index}/{repeat_count}",
                    file=sys.stderr,
                )
                result = run_repetition(
                    root=root,
                    task=task,
                    source=source,
                    solution_name=solution_name,
                    solution_kind=solution_kind,
                    environment=environment,
                    engine=args.engine,
                    timeout_sec=timeout_sec,
                    repeat_index=repeat_index,
                    planned_repeats=repeat_count,
                    logs_dir=logs_dir,
                    image_provenance=image_provenance.get(task.environment),
                    comparison_role=comparison_role,
                    pair_order=pair_order,
                    pair_position=pair_position,
                )
                record_result(result)

    candidate_name = resolved[0][1][1] if resolved else args.solution
    comparison_names = (
        (args.compare_to, candidate_name)
        if args.compare_to is not None
        else None
    )
    report = write_report(
        report_path=report_path,
        summary_path=summary_path,
        results=results,
        root=root,
        invocation=["bench", *argv],
        comparison=comparison_names,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    _print_comparison(report["summary"])
    print(f"Results: {report_path}", file=sys.stderr)
    print(f"Summary: {summary_path}", file=sys.stderr)
    return 0 if all(result.get("passed") for result in results) else 1


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    actual_argv = list(argv if argv is not None else sys.argv[1:])
    benchmark_root = (root or default_root()).expanduser().resolve()
    args = _parser().parse_args(actual_argv)
    try:
        if args.command == "list":
            tasks = discover_tasks(benchmark_root)
            if args.as_json:
                print(json.dumps(_list_payload(tasks), indent=2))
            else:
                _print_task_list(tasks)
            return 0

        if args.command == "verify":
            errors = verify_benchmark(benchmark_root)
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print("Benchmark verification passed: tasks 01 through 12 are complete.")
            return 0

        if args.command == "env":
            config = load_bench_config(benchmark_root)
            if args.env_command == "doctor":
                ok, report = doctor_environment(config)
                print(json.dumps(report, indent=2, sort_keys=True))
                if not ok:
                    for error in report.get("resource_errors", []):
                        print(f"RESOURCE ERROR: {error}", file=sys.stderr)
                return 0 if ok else 1
            name = args.name or config.default_environment
            if name not in config.environments:
                raise BenchmarkError(f"Unknown environment: {name}")
            command = build_environment(
                benchmark_root,
                config.environments[name],
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(json.dumps({"command": command}, indent=2))
            else:
                image = inspect_docker_image(config.environments[name].image)
                print(json.dumps(image or {}, indent=2, sort_keys=True))
            return 0

        if args.command == "run":
            return _handle_run(
                args,
                root=benchmark_root,
                argv=actual_argv,
            )
        raise BenchmarkError(f"Unknown command: {args.command}")
    except (BenchmarkError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
