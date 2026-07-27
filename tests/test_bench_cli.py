from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitbreakers_bench.cli import main  # noqa: E402
from orbitbreakers_bench.manifest import (  # noqa: E402
    ManifestError,
    discover_tasks,
    load_bench_config,
    verify_benchmark,
)
from orbitbreakers_bench.runner import (  # noqa: E402
    DockerTaskSession,
    ProcessResult,
    aggregate_results,
    doctor_environment,
    parse_evaluator_output,
    requested_memory_bytes,
    terminal_status,
)


SOLUTION_SOURCE = """\
def run_solution(config):
    return {"ok": True}
"""

FAILING_SOLUTION_SOURCE = """\
def run_solution(config):
    return {"ok": False}
"""

EVALUATOR_SOURCE = """\
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
args = parser.parse_args()
module = importlib.import_module(args.solution)
result = module.run_solution({})
print("End-to-end solution time: 0.125s")
print("Overall: PASS" if result.get("ok") else "Overall: FAIL")
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_benchmark(root: Path, *, task_count: int = 1) -> None:
    env_dir = root / "envs" / "test"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (env_dir / "requirements.lock").write_text("\n", encoding="utf-8")
    (env_dir / "sitecustomize.py").write_text(
        "ORBITBENCH_COMPATIBILITY_SHIM = True\n",
        encoding="utf-8",
    )
    (root / "bench.toml").write_text(
        textwrap.dedent(
            """\
            default_environment = "test"
            default_timeout_sec = 300
            default_repeats = 4

            [environments.test]
            image = "orbitbench-test:latest"
            dockerfile = "envs/test/Dockerfile"
            requirements = "envs/test/requirements.lock"
            cpus = 1
            memory = "1g"
            """
        ),
        encoding="utf-8",
    )

    for number in range(1, task_count + 1):
        task_id = f"{number:02d}"
        task = root / "tasks" / f"challenge-{task_id}"
        evaluator_dir = task / "evaluator"
        solution_dir = task / "solutions" / "optimized"
        evaluator_dir.mkdir(parents=True)
        solution_dir.mkdir(parents=True)
        evaluator = evaluator_dir / f"evaluate_{number}.py"
        solution = solution_dir / f"solution_{number}.py"
        evaluator.write_text(EVALUATOR_SOURCE, encoding="utf-8")
        solution.write_text(SOLUTION_SOURCE, encoding="utf-8")
        (task / "task.toml").write_text(
            textwrap.dedent(
                f"""\
                id = "{task_id}"
                title = "Task {task_id}"
                environment = "test"
                evaluator = "evaluator/evaluate_{number}.py"
                module = "solution_{number}"
                timeout_sec = 300

                [[solutions]]
                name = "optimized"
                path = "solutions/optimized/solution_{number}.py"
                kind = "optimized-human-expert"
                sha256 = "{_sha256(SOLUTION_SOURCE)}"
                """
            ),
            encoding="utf-8",
        )


def add_solution(
    root: Path,
    *,
    task_id: str,
    name: str,
    source: str = SOLUTION_SOURCE,
) -> Path:
    number = int(task_id)
    task = root / "tasks" / f"challenge-{task_id}"
    solution_dir = task / "solutions" / name
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution = solution_dir / f"solution_{number}_{name}.py"
    solution.write_text(source, encoding="utf-8")
    with (task / "task.toml").open("a", encoding="utf-8") as handle:
        handle.write(
            textwrap.dedent(
                f"""\

                [[solutions]]
                name = "{name}"
                path = "solutions/{name}/solution_{number}_{name}.py"
                kind = "test-{name}"
                sha256 = "{_sha256(source)}"
                """
            )
        )
    return solution


def paired_row(
    *,
    task_id: str = "01",
    pair_id: int,
    role: str,
    runtime: float,
    planned_repeats: int = 2,
    passed: bool = True,
) -> dict[str, object]:
    odd = pair_id % 2 == 1
    return {
        "task_id": task_id,
        "solution": role,
        "comparison_role": role,
        "pair_id": pair_id,
        "repeat": pair_id,
        "planned_repeats": planned_repeats,
        "pair_order": (
            "reference->candidate" if odd else "candidate->reference"
        ),
        "pair_position": (
            (1 if odd else 2)
            if role == "reference"
            else (2 if odd else 1)
        ),
        "passed": passed,
        "timed_out": False,
        "runtime_sec": runtime,
    }


class ManifestTests(unittest.TestCase):
    def test_discovers_task_relative_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root, task_count=2)

            config = load_bench_config(root)
            tasks = discover_tasks(root, config)

            self.assertEqual([task.id for task in tasks], ["01", "02"])
            self.assertEqual(tasks[0].module, "solution_1")
            self.assertEqual(tasks[0].solutions[0].name, "optimized")
            self.assertEqual(config.default_repeats, 4)
            self.assertTrue(tasks[0].evaluator.is_file())
            self.assertTrue(tasks[0].solutions[0].path.is_file())

    def test_verify_accepts_complete_twelve_task_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root, task_count=12)
            self.assertEqual(verify_benchmark(root), [])

    def test_rejects_manifest_path_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            manifest = root / "tasks" / "challenge-01" / "task.toml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    'path = "solutions/optimized/solution_1.py"',
                    'path = "../../../../outside.py"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "escapes benchmark root"):
                discover_tasks(root)


class ParserAndAggregationTests(unittest.TestCase):
    def test_parses_runtime_and_pass_marker(self) -> None:
        parsed = parse_evaluator_output(
            "setup\nEnd-to-end solution time: 12.375s\nOverall: PASS\n"
        )
        self.assertEqual(parsed.runtime_sec, 12.375)
        self.assertTrue(parsed.passed_marker)
        self.assertEqual(parsed.functional_status, "PASS")

    def test_missing_runtime_does_not_parse(self) -> None:
        parsed = parse_evaluator_output("Overall: PASS\n")
        self.assertIsNone(parsed.runtime_sec)
        self.assertTrue(parsed.passed_marker)
        self.assertEqual(parsed.functional_status, "PASS")

    def test_final_overall_marker_controls_status(self) -> None:
        parsed = parse_evaluator_output(
            "End-to-end solution time: 1.0s\n"
            "Overall: PASS\n"
            "late validation\n"
            "Overall: FAIL\n"
        )
        self.assertFalse(parsed.passed_marker)
        self.assertEqual(parsed.functional_status, "FAIL")
        process = ProcessResult(
            returncode=0,
            stdout="",
            stderr="",
            timed_out=False,
            wall_sec=1.0,
        )
        self.assertEqual(terminal_status(process, parsed), "FUNCTIONAL_FAILED")

    def test_terminal_status_classification(self) -> None:
        success = parse_evaluator_output(
            "End-to-end solution time: 1.0s\nOverall: PASS\n"
        )
        functional_failure = parse_evaluator_output(
            "End-to-end solution time: 1.0s\nOverall: FAIL\n"
        )
        zero_runtime = parse_evaluator_output(
            "End-to-end solution time: 0.0s\nOverall: PASS\n"
        )
        invalid = parse_evaluator_output("unstructured output\n")

        def process(*, returncode: int, timed_out: bool = False) -> ProcessResult:
            return ProcessResult(
                returncode=returncode,
                stdout="",
                stderr="",
                timed_out=timed_out,
                wall_sec=1.0,
            )

        self.assertEqual(terminal_status(process(returncode=0), success), "SUCCESS")
        self.assertEqual(
            terminal_status(process(returncode=-9, timed_out=True), invalid),
            "TIMEOUT",
        )
        self.assertEqual(
            terminal_status(process(returncode=2), success),
            "NONZERO_EXIT",
        )
        self.assertEqual(
            terminal_status(process(returncode=0), functional_failure),
            "FUNCTIONAL_FAILED",
        )
        self.assertEqual(
            terminal_status(process(returncode=0), invalid),
            "INVALID_OUTPUT",
        )
        self.assertEqual(
            terminal_status(process(returncode=0), zero_runtime),
            "INVALID_OUTPUT",
        )

    def test_aggregation_uses_only_passing_runtime_values(self) -> None:
        summary = aggregate_results(
            [
                {
                    "task_id": "01",
                    "passed": True,
                    "timed_out": False,
                    "runtime_sec": 1.0,
                },
                {
                    "task_id": "01",
                    "passed": True,
                    "timed_out": False,
                    "runtime_sec": 3.0,
                },
                {
                    "task_id": "01",
                    "passed": False,
                    "timed_out": True,
                    "runtime_sec": None,
                },
            ]
        )
        task = summary["tasks"]["01"]
        self.assertEqual(task["runs"], 3)
        self.assertEqual(task["passing_runs"], 2)
        self.assertEqual(task["timed_out_runs"], 1)
        self.assertEqual(task["runtime_sec"]["mean"], 2.0)
        self.assertEqual(task["runtime_sec"]["median"], 2.0)
        self.assertAlmostEqual(task["runtime_sec"]["stdev"], 2**0.5)
        self.assertAlmostEqual(task["runtime_sec"]["stderr"], 1.0)
        self.assertEqual(task["runtime_sec"]["min"], 1.0)
        self.assertEqual(task["runtime_sec"]["max"], 3.0)

    def test_single_runtime_has_null_stderr(self) -> None:
        summary = aggregate_results(
            [
                {
                    "task_id": "01",
                    "passed": True,
                    "timed_out": False,
                    "runtime_sec": 1.0,
                }
            ]
        )
        self.assertIsNone(summary["runtime_sec"]["stderr"])
        self.assertIsNone(summary["tasks"]["01"]["runtime_sec"]["stderr"])

    def test_paired_stats_use_actual_role_and_reject_missing_pair(self) -> None:
        rows = [
            paired_row(pair_id=1, role="reference", runtime=2.0),
            paired_row(pair_id=1, role="candidate", runtime=1.0),
            paired_row(pair_id=2, role="reference", runtime=4.0),
        ]
        summary = aggregate_results(
            rows,
            comparison=("reference", "candidate"),
        )
        comparison = summary["comparison"]["tasks"]["01"]
        self.assertFalse(comparison["pairing_valid"])
        self.assertFalse(comparison["eligible"])
        self.assertEqual(comparison["pairs"][1]["candidate_rows"], 0)
        self.assertIsNone(comparison["paired_improvement_mean"])
        self.assertIsNone(comparison["speedup_mean"])

    def test_paired_stats_reject_duplicate_role_and_zero_runtime(self) -> None:
        duplicate = [
            paired_row(pair_id=1, role="reference", runtime=2.0, planned_repeats=1),
            paired_row(pair_id=1, role="reference", runtime=2.1, planned_repeats=1),
            paired_row(pair_id=1, role="candidate", runtime=0.0, planned_repeats=1),
        ]
        summary = aggregate_results(
            duplicate,
            comparison=("reference", "candidate"),
        )
        comparison = summary["comparison"]["tasks"]["01"]
        self.assertFalse(comparison["pairing_valid"])
        self.assertEqual(comparison["pairs"][0]["reference_rows"], 2)
        self.assertIsNone(comparison["improvement_pct"])
        self.assertIsNone(comparison["speedup"])

    def test_paired_means_and_stderr_are_reported(self) -> None:
        rows = [
            paired_row(pair_id=1, role="reference", runtime=2.0),
            paired_row(pair_id=1, role="candidate", runtime=1.0),
            paired_row(pair_id=2, role="reference", runtime=4.0),
            paired_row(pair_id=2, role="candidate", runtime=1.0),
        ]
        comparison = aggregate_results(
            rows,
            comparison=("reference", "candidate"),
        )["comparison"]
        self.assertTrue(comparison["eligible"])
        self.assertEqual(comparison["paired_improvement_mean"], 62.5)
        self.assertEqual(comparison["paired_improvement_stderr"], 12.5)
        self.assertEqual(comparison["speedup_mean"], 3.0)
        self.assertEqual(comparison["speedup_stderr"], 1.0)

    def test_all_task_summary_does_not_report_pooled_runtime_speedup(self) -> None:
        rows = [
            paired_row(
                task_id="01",
                pair_id=1,
                role="reference",
                runtime=10.0,
                planned_repeats=1,
            ),
            paired_row(
                task_id="01",
                pair_id=1,
                role="candidate",
                runtime=5.0,
                planned_repeats=1,
            ),
            paired_row(
                task_id="02",
                pair_id=1,
                role="reference",
                runtime=1.0,
                planned_repeats=1,
            ),
            paired_row(
                task_id="02",
                pair_id=1,
                role="candidate",
                runtime=0.9,
                planned_repeats=1,
            ),
        ]
        comparison = aggregate_results(
            rows,
            comparison=("reference", "candidate"),
        )["comparison"]
        self.assertTrue(comparison["eligible"])
        self.assertFalse(comparison["pooled_runtime_ratio_reported"])
        self.assertFalse(comparison["pooled_pair_metrics_reported"])
        self.assertIsNone(comparison["improvement_pct"])
        self.assertIsNone(comparison["speedup"])
        self.assertIsNone(comparison["paired_improvement_mean"])
        self.assertIsNone(comparison["paired_improvement_stderr"])
        self.assertIsNone(comparison["speedup_mean"])
        self.assertIsNone(comparison["speedup_stderr"])
        self.assertEqual(comparison["paired_speedup"]["count"], 0)


class DoctorTests(unittest.TestCase):
    @staticmethod
    def _docker_run(
        command: list[str],
        **_: object,
    ):
        if command[1] == "version":
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                stdout="27.0.0\n",
                stderr="",
            )
        if command[1] == "info":
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "NCPU": 8,
                        "MemTotal": 16 * 1024**3,
                    }
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    def test_doctor_fails_when_environment_exceeds_docker_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            bench_path = root / "bench.toml"
            bench_path.write_text(
                bench_path.read_text(encoding="utf-8")
                .replace("cpus = 1", "cpus = 14")
                .replace('memory = "1g"', 'memory = "32g"'),
                encoding="utf-8",
            )
            config = load_bench_config(root)
            with mock.patch(
                "orbitbreakers_bench.runner.shutil.which",
                return_value="/usr/local/bin/docker",
            ), mock.patch(
                "orbitbreakers_bench.runner.subprocess.run",
                side_effect=self._docker_run,
            ), mock.patch(
                "orbitbreakers_bench.runner.inspect_docker_image",
                return_value=None,
            ):
                ok, report = doctor_environment(config)

            self.assertFalse(ok)
            self.assertEqual(report["docker_resources"]["cpus"], 8.0)
            self.assertEqual(
                report["docker_resources"]["memory_bytes"],
                16 * 1024**3,
            )
            environment = report["environments"]["test"]
            self.assertEqual(environment["requested_cpus_normalized"], 14.0)
            self.assertEqual(
                environment["requested_memory_bytes"],
                32 * 1024**3,
            )
            self.assertFalse(environment["resource_ok"])
            self.assertTrue(
                any("requests 14 CPUs but Docker has 8" in error for error in report["resource_errors"])
            )
            self.assertTrue(
                any("32.00 GiB" in error and "16.00 GiB" in error for error in report["resource_errors"])
            )

    def test_doctor_passes_when_requested_resources_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            config = load_bench_config(root)
            with mock.patch(
                "orbitbreakers_bench.runner.shutil.which",
                return_value="/usr/local/bin/docker",
            ), mock.patch(
                "orbitbreakers_bench.runner.subprocess.run",
                side_effect=self._docker_run,
            ), mock.patch(
                "orbitbreakers_bench.runner.inspect_docker_image",
                return_value={"id": "test"},
            ):
                ok, report = doctor_environment(config)

            self.assertTrue(ok)
            environment = report["environments"]["test"]
            self.assertTrue(environment["resource_ok"])
            self.assertEqual(environment["requested_memory_human"], "1.00 GiB")
            self.assertEqual(report["resource_errors"], [])

    def test_numeric_memory_uses_runner_mib_convention(self) -> None:
        self.assertEqual(requested_memory_bytes(32768), 32 * 1024**3)


class DockerTaskSessionTests(unittest.TestCase):
    @staticmethod
    def _process(
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = 0,
        timed_out: bool = False,
    ) -> ProcessResult:
        return ProcessResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            wall_sec=0.25,
        )

    def test_one_container_and_one_read_only_snapshot_mount_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            reference = add_solution(root, task_id="01", name="reference")
            config = load_bench_config(root)
            task = discover_tasks(root, config)[0]
            optimized = task.solution("optimized").path
            observed_commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                command = list(command)
                observed_commands.append(command)
                if command[1] == "run":
                    staging_dir = Path(kwargs["cwd"])
                    staged_python = list(staging_dir.rglob("*.py"))
                    self.assertEqual(len(staged_python), 4)
                    self.assertTrue(
                        (staging_dir / "evaluator" / "evaluate_1.py").is_file()
                    )
                    self.assertTrue(
                        (staging_dir / "environment" / "sitecustomize.py").is_file()
                    )
                    self.assertEqual(command.count("--mount"), 1)
                    mount = command[command.index("--mount") + 1]
                    self.assertIn("dst=/session", mount)
                    self.assertIn("readonly", mount)
                    return self._process(stdout="container-actual-id\n")
                if command[1] == "exec":
                    return self._process(
                        stdout=(
                            "End-to-end solution time: 0.125s\n"
                            "Overall: PASS\n"
                        )
                    )
                raise AssertionError(f"unexpected command: {command}")

            logs = root / "logs"
            with mock.patch(
                "orbitbreakers_bench.runner.run_process",
                side_effect=fake_run,
            ) as run_mock, mock.patch(
                "orbitbreakers_bench.runner._cleanup_container",
                return_value=True,
            ) as cleanup_mock:
                with DockerTaskSession(
                    root=root,
                    task=task,
                    environment=config.environments["test"],
                    targets={
                        "reference": reference,
                        "optimized": optimized,
                    },
                    image_provenance={"id": "image-id"},
                ) as session:
                    container_name = session.container_name
                    staging_dir = session.staging_dir
                    reference_result = session.run_measurement(
                        solution_name="reference",
                        solution_kind="test-reference",
                        comparison_role="reference",
                        timeout_sec=300,
                        repeat_index=1,
                        planned_repeats=1,
                        logs_dir=logs,
                        pair_order="reference->candidate",
                        pair_position=1,
                    )
                    candidate_result = session.run_measurement(
                        solution_name="optimized",
                        solution_kind="optimized-human-expert",
                        comparison_role="candidate",
                        timeout_sec=300,
                        repeat_index=1,
                        planned_repeats=1,
                        logs_dir=logs,
                        pair_order="reference->candidate",
                        pair_position=2,
                    )
                    self.assertTrue(session.active)
                    self.assertIsNotNone(staging_dir)
                    self.assertTrue(staging_dir.is_dir())

            self.assertEqual(run_mock.call_count, 3)
            self.assertEqual(
                [command[1] for command in observed_commands],
                ["run", "exec", "exec"],
            )
            self.assertEqual(reference_result["shared_container_name"], container_name)
            self.assertEqual(candidate_result["shared_container_name"], container_name)
            self.assertEqual(
                reference_result["shared_container_id"],
                "container-actual-id",
            )
            self.assertEqual(
                candidate_result["shared_container_id"],
                "container-actual-id",
            )
            self.assertEqual(
                reference_result["staging_snapshot_sha256"],
                candidate_result["staging_snapshot_sha256"],
            )
            self.assertEqual(
                reference_result["shared_container_start_command"],
                candidate_result["shared_container_start_command"],
            )
            start_command = reference_result["shared_container_start_command"]
            self.assertEqual(start_command[:2], ["docker", "run"])
            self.assertEqual(start_command[-3:], ["tail", "-f", "/dev/null"])
            self.assertEqual(start_command[start_command.index("--cpus") + 1], "1")
            self.assertEqual(
                start_command[start_command.index("--memory") + 1],
                "1g",
            )
            self.assertEqual(
                set(reference_result["staged_solution_hashes"]),
                {"reference", "optimized"},
            )
            self.assertEqual(
                reference_result["compatibility_sha256"],
                _sha256("ORBITBENCH_COMPATIBILITY_SHIM = True\n"),
            )
            self.assertEqual(
                reference_result["staged_sitecustomize_sha256"],
                reference_result["compatibility_sha256"],
            )
            self.assertFalse(staging_dir.exists())
            cleanup_mock.assert_called_once_with(container_name)

    def test_single_solution_repeats_reuse_one_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            config = load_bench_config(root)
            task = discover_tasks(root, config)[0]
            calls: list[list[str]] = []

            def fake_run(command, **_):
                command = list(command)
                calls.append(command)
                if command[1] == "run":
                    return self._process(stdout="single-container\n")
                return self._process(
                    stdout="End-to-end solution time: 0.125s\nOverall: PASS\n"
                )

            with mock.patch(
                "orbitbreakers_bench.runner.run_process",
                side_effect=fake_run,
            ), mock.patch(
                "orbitbreakers_bench.runner._cleanup_container",
                return_value=True,
            ):
                with DockerTaskSession(
                    root=root,
                    task=task,
                    environment=config.environments["test"],
                    targets={"optimized": task.solution("optimized").path},
                    image_provenance={},
                ) as session:
                    results = [
                        session.run_measurement(
                            solution_name="optimized",
                            solution_kind="optimized-human-expert",
                            comparison_role=None,
                            timeout_sec=300,
                            repeat_index=repeat,
                            planned_repeats=2,
                            logs_dir=root / "logs",
                            pair_order=None,
                            pair_position=None,
                        )
                        for repeat in (1, 2)
                    ]

            self.assertEqual([call[1] for call in calls], ["run", "exec", "exec"])
            self.assertEqual(
                {result["shared_container_id"] for result in results},
                {"single-container"},
            )

    def test_timeout_enforces_cap_and_stops_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            config = load_bench_config(root)
            task = discover_tasks(root, config)[0]
            with mock.patch(
                "orbitbreakers_bench.runner.run_process",
                side_effect=[
                    self._process(stdout="timeout-container\n"),
                    self._process(returncode=-9, timed_out=True),
                ],
            ) as run_mock, mock.patch(
                "orbitbreakers_bench.runner._cleanup_container",
                return_value=True,
            ) as cleanup_mock:
                with DockerTaskSession(
                    root=root,
                    task=task,
                    environment=config.environments["test"],
                    targets={"optimized": task.solution("optimized").path},
                    image_provenance={},
                ) as session:
                    container_name = session.container_name
                    result = session.run_measurement(
                        solution_name="optimized",
                        solution_kind="optimized-human-expert",
                        comparison_role=None,
                        timeout_sec=999,
                        repeat_index=1,
                        planned_repeats=4,
                        logs_dir=root / "logs",
                        pair_order=None,
                        pair_position=None,
                    )
                    self.assertFalse(session.active)

            self.assertEqual(result["terminal_status"], "TIMEOUT")
            self.assertEqual(result["timeout_sec"], 300.0)
            self.assertEqual(run_mock.call_args_list[1].kwargs["timeout_sec"], 300.0)
            cleanup_mock.assert_called_once_with(container_name)

    def test_interrupt_during_exec_cleans_container_and_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            config = load_bench_config(root)
            task = discover_tasks(root, config)[0]
            with mock.patch(
                "orbitbreakers_bench.runner.run_process",
                side_effect=[
                    self._process(stdout="interrupt-container\n"),
                    KeyboardInterrupt(),
                ],
            ), mock.patch(
                "orbitbreakers_bench.runner._cleanup_container",
                return_value=True,
            ) as cleanup_mock:
                with self.assertRaises(KeyboardInterrupt):
                    with DockerTaskSession(
                        root=root,
                        task=task,
                        environment=config.environments["test"],
                        targets={"optimized": task.solution("optimized").path},
                        image_provenance={},
                    ) as session:
                        container_name = session.container_name
                        staging_dir = session.staging_dir
                        session.run_measurement(
                            solution_name="optimized",
                            solution_kind="optimized-human-expert",
                            comparison_role=None,
                            timeout_sec=300,
                            repeat_index=1,
                            planned_repeats=1,
                            logs_dir=root / "logs",
                            pair_order=None,
                            pair_position=None,
                        )

            cleanup_mock.assert_called_once_with(container_name)
            self.assertIsNotNone(staging_dir)
            self.assertFalse(staging_dir.exists())


class CliTests(unittest.TestCase):
    def test_list_json_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root, task_count=2)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                returncode = main(["list", "--json"], root=root)

            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual([row["id"] for row in payload], ["01", "02"])
            self.assertEqual(payload[0]["solutions"][0]["name"], "optimized")

    def test_local_dry_run_prints_plan_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                returncode = main(
                    ["run", "01", "--engine", "local", "--dry-run"],
                    root=root,
                )

            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["plans"][0]["task_id"], "01")
            self.assertEqual(payload["plans"][0]["repeat"], 4)
            command = payload["plans"][0]["command"]
            self.assertIn("--solution", command)
            self.assertIn("solution_1", command)

    def test_local_run_writes_results_summary_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            output = root / "artifacts"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--engine",
                        "local",
                        "--repeat",
                        "2",
                        "--output",
                        str(output),
                    ],
                    root=root,
                )

            self.assertEqual(returncode, 0, stderr.getvalue())
            report = json.loads((output / "results.json").read_text(encoding="utf-8"))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(
                all(
                    row["terminal_status"] == "SUCCESS"
                    for row in report["results"]
                )
            )
            self.assertEqual(summary["passing_runs"], 2)
            self.assertEqual(summary["tasks"]["01"]["runtime_sec"]["mean"], 0.125)
            self.assertEqual(
                report["results"][0]["compatibility_sha256"],
                _sha256("ORBITBENCH_COMPATIBILITY_SHIM = True\n"),
            )
            self.assertEqual(
                len(list((output / "logs").glob("*.stdout.log"))),
                2,
            )
            staging_root = Path(report["results"][0]["staging_root"])
            staging_dir = Path(report["results"][0]["staging_dir"])
            self.assertEqual(staging_root, root.resolve() / ".tmp")
            self.assertTrue(staging_dir.is_relative_to(root.resolve() / ".tmp"))
            self.assertFalse(staging_dir.exists())

    def test_explicit_candidate_may_be_outside_benchmark_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as other:
            root = Path(temp)
            create_benchmark(root)
            candidate = Path(other) / "candidate.py"
            candidate.write_text(SOLUTION_SOURCE, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--candidate",
                        str(candidate),
                        "--engine",
                        "local",
                        "--dry-run",
                    ],
                    root=root,
                )
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["plans"][0]["source"], str(candidate.resolve()))

    def test_compare_equal_solutions_reports_zero_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            add_solution(root, task_id="01", name="reference")
            output = root / "comparison"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--solution",
                        "optimized",
                        "--compare-to",
                        "reference",
                        "--repeat",
                        "3",
                        "--engine",
                        "local",
                        "--output",
                        str(output),
                    ],
                    root=root,
                )

            self.assertEqual(returncode, 0, stderr.getvalue())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            comparison = summary["comparison"]
            task_comparison = comparison["tasks"]["01"]
            self.assertTrue(task_comparison["eligible"])
            self.assertAlmostEqual(task_comparison["improvement_pct"], 0.0)
            self.assertAlmostEqual(task_comparison["speedup"], 1.0)
            self.assertEqual(task_comparison["reference_mean"], 0.125)
            self.assertEqual(task_comparison["reference_median"], 0.125)
            self.assertEqual(task_comparison["candidate_mean"], 0.125)
            self.assertEqual(task_comparison["candidate_median"], 0.125)
            self.assertEqual(task_comparison["paired_improvement_mean"], 0.0)
            self.assertEqual(task_comparison["paired_improvement_stderr"], 0.0)
            self.assertEqual(task_comparison["speedup_mean"], 1.0)
            self.assertEqual(task_comparison["speedup_stderr"], 0.0)
            report = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    (
                        row["repeat"],
                        row["comparison_role"],
                        row["pair_order"],
                        row["pair_position"],
                    )
                    for row in report["results"]
                ],
                [
                    (1, "reference", "reference->candidate", 1),
                    (1, "candidate", "reference->candidate", 2),
                    (2, "candidate", "candidate->reference", 1),
                    (2, "reference", "candidate->reference", 2),
                    (3, "reference", "reference->candidate", 1),
                    (3, "candidate", "reference->candidate", 2),
                ],
            )
            self.assertIn(
                "paired improvement=0.000000% ± 0.000000%",
                stderr.getvalue(),
            )
            self.assertIn(
                "paired speedup=1.000000x ± 0.000000x",
                stderr.getvalue(),
            )

    def test_invalid_compare_name_fails_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--compare-to",
                        "missing",
                        "--engine",
                        "local",
                    ],
                    root=root,
                )
            self.assertEqual(returncode, 2)
            self.assertIn("no solution named 'missing'", stderr.getvalue())

    def test_compare_failure_gates_improvement_and_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            add_solution(root, task_id="01", name="reference")
            optimized = (
                root
                / "tasks"
                / "challenge-01"
                / "solutions"
                / "optimized"
                / "solution_1.py"
            )
            optimized.write_text(FAILING_SOLUTION_SOURCE, encoding="utf-8")
            manifest = root / "tasks" / "challenge-01" / "task.toml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    _sha256(SOLUTION_SOURCE),
                    _sha256(FAILING_SOLUTION_SOURCE),
                    1,
                ),
                encoding="utf-8",
            )
            output = root / "comparison-failure"
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--solution",
                        "optimized",
                        "--compare-to",
                        "reference",
                        "--repeat",
                        "2",
                        "--engine",
                        "local",
                        "--output",
                        str(output),
                    ],
                    root=root,
                )

            self.assertEqual(returncode, 1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            comparison = summary["comparison"]["tasks"]["01"]
            self.assertFalse(comparison["eligible"])
            self.assertFalse(comparison["all_runs_passed"])
            self.assertIsNone(comparison["improvement_pct"])
            self.assertIsNone(comparison["speedup"])
            self.assertIsNone(comparison["paired_improvement_mean"])
            self.assertIsNone(comparison["speedup_mean"])
            report = json.loads(
                (output / "results.json").read_text(encoding="utf-8")
            )
            candidate_statuses = [
                row["terminal_status"]
                for row in report["results"]
                if row["comparison_role"] == "candidate"
            ]
            self.assertEqual(
                candidate_statuses,
                ["FUNCTIONAL_FAILED", "FUNCTIONAL_FAILED"],
            )

    def test_docker_timeout_aborts_remaining_task_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            add_solution(root, task_id="01", name="reference")
            output = root / "docker-timeout"
            instances = []

            class FakeSession:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.calls = []
                    instances.append(self)

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def run_measurement(self, **kwargs):
                    self.calls.append(kwargs)
                    return {
                        "task_id": "01",
                        "solution": kwargs["solution_name"],
                        "comparison_role": kwargs["comparison_role"],
                        "pair_id": kwargs["repeat_index"],
                        "repeat": kwargs["repeat_index"],
                        "planned_repeats": kwargs["planned_repeats"],
                        "pair_order": kwargs["pair_order"],
                        "pair_position": kwargs["pair_position"],
                        "passed": False,
                        "timed_out": True,
                        "runtime_sec": None,
                        "terminal_status": "TIMEOUT",
                    }

            with mock.patch(
                "orbitbreakers_bench.cli.ensure_docker_image",
                return_value={"id": "image"},
            ), mock.patch(
                "orbitbreakers_bench.cli.DockerTaskSession",
                FakeSession,
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                returncode = main(
                    [
                        "run",
                        "01",
                        "--solution",
                        "optimized",
                        "--compare-to",
                        "reference",
                        "--repeat",
                        "4",
                        "--engine",
                        "docker",
                        "--output",
                        str(output),
                    ],
                    root=root,
                )

            self.assertEqual(returncode, 1)
            self.assertEqual(len(instances), 1)
            self.assertEqual(len(instances[0].calls), 1)
            first = instances[0].calls[0]
            self.assertEqual(first["solution_name"], "reference")
            self.assertEqual(first["pair_order"], "reference->candidate")
            self.assertEqual(first["pair_position"], 1)

    def test_explicit_output_refuses_existing_evidence_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_benchmark(root)
            for evidence_name in ("results.json", "summary.json", "logs"):
                with self.subTest(evidence=evidence_name):
                    output = root / f"existing-{evidence_name.replace('.', '-')}"
                    output.mkdir()
                    evidence = output / evidence_name
                    if evidence_name == "logs":
                        evidence.mkdir()
                    else:
                        evidence.write_text("{}\n", encoding="utf-8")
                    stderr = io.StringIO()
                    with mock.patch(
                        "orbitbreakers_bench.cli.run_repetition"
                    ) as run_mock, contextlib.redirect_stderr(stderr):
                        returncode = main(
                            [
                                "run",
                                "01",
                                "--engine",
                                "local",
                                "--output",
                                str(output),
                            ],
                            root=root,
                        )
                    self.assertEqual(returncode, 2)
                    run_mock.assert_not_called()
                    self.assertIn(
                        "Output evidence already exists",
                        stderr.getvalue(),
                    )


if __name__ == "__main__":
    unittest.main()
