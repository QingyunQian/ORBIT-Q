from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .manifest import (
    BenchConfig,
    EnvironmentConfig,
    TaskConfig,
    file_sha256,
)


RUNTIME_RE = re.compile(
    r"End-to-end solution time:\s*"
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)s"
)
OVERALL_RE = re.compile(r"Overall:\s*(?P<status>PASS|FAIL)\b")


class BenchmarkError(RuntimeError):
    """Raised for runner setup or execution failures."""


@dataclass(frozen=True)
class EvaluatorParse:
    runtime_sec: float | None
    passed_marker: bool
    functional_status: str | None


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    wall_sec: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_evaluator_output(output: str) -> EvaluatorParse:
    runtime_matches = list(RUNTIME_RE.finditer(output))
    runtime = (
        float(runtime_matches[-1].group("seconds")) if runtime_matches else None
    )
    overall_matches = list(OVERALL_RE.finditer(output))
    functional_status = (
        overall_matches[-1].group("status") if overall_matches else None
    )
    return EvaluatorParse(
        runtime_sec=runtime,
        passed_marker=functional_status == "PASS",
        functional_status=functional_status,
    )


def terminal_status(
    process_result: ProcessResult,
    parsed: EvaluatorParse,
) -> str:
    if process_result.timed_out:
        return "TIMEOUT"
    if process_result.returncode != 0:
        return "NONZERO_EXIT"
    if (
        parsed.runtime_sec is None
        or not math.isfinite(parsed.runtime_sec)
        or parsed.runtime_sec <= 0
        or parsed.functional_status is None
    ):
        return "INVALID_OUTPUT"
    if parsed.functional_status == "FAIL":
        return "FUNCTIONAL_FAILED"
    if parsed.functional_status == "PASS":
        return "SUCCESS"
    return "INVALID_OUTPUT"


def _safe_git_output(root: Path, args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def host_provenance(root: Path) -> dict[str, Any]:
    details = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "git_commit": _safe_git_output(root, ["rev-parse", "HEAD"]),
        "git_branch": _safe_git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
    }
    fingerprint_source = json.dumps(
        {
            key: details[key]
            for key in (
                "hostname",
                "platform",
                "machine",
                "processor",
                "python",
                "cpu_count",
            )
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    details["fingerprint_sha256"] = hashlib.sha256(fingerprint_source).hexdigest()
    return details


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_sec: float,
) -> ProcessResult:
    start = time.perf_counter()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise BenchmarkError(f"Could not start {command[0]!r}: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
        timed_out = False
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        timed_out = True
    except BaseException:
        _terminate_process_tree(process)
        raise
    return ProcessResult(
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
        wall_sec=time.perf_counter() - start,
    )


def _environment_pythonpath(
    *,
    root: Path,
    environment: EnvironmentConfig,
    candidate_path: str,
    container: bool,
) -> str:
    if container:
        return f"{candidate_path}:/session/environment"
    compatibility_dir = environment.dockerfile.parent.resolve()
    try:
        compatibility_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise BenchmarkError(
            f"Environment directory is outside benchmark root: {compatibility_dir}"
        ) from exc
    return f"{candidate_path}{os.pathsep}{compatibility_dir}"


def _docker_container_start_command(
    *,
    root: Path,
    staging_dir: Path | str,
    task: TaskConfig,
    environment: EnvironmentConfig,
    container_name: str,
) -> list[str]:
    command = [
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
        f"type=bind,src={staging_dir},dst=/session,readonly",
        "--workdir",
        "/session",
        "--env",
        "NUMBA_DISABLE_JIT=1",
        "--env",
        "PYTHONPATH="
        + _environment_pythonpath(
            root=root,
            environment=environment,
            candidate_path="/session",
            container=True,
        ),
    ]
    if environment.cpus not in (None, ""):
        command.extend(["--cpus", str(environment.cpus)])
    if environment.memory not in (None, ""):
        memory = environment.memory
        if isinstance(memory, (int, float)) and not isinstance(memory, bool):
            memory = f"{memory}m"
        command.extend(["--memory", str(memory)])
    command.extend(
        [
            environment.image,
            "tail",
            "-f",
            "/dev/null",
        ]
    )
    return command


def _docker_exec_command(
    *,
    root: Path,
    task: TaskConfig,
    environment: EnvironmentConfig,
    container_name: str,
    module: str,
) -> list[str]:
    return [
        "docker",
        "exec",
        "--workdir",
        "/session",
        "--env",
        "NUMBA_DISABLE_JIT=1",
        "--env",
        "PYTHONPATH="
        + _environment_pythonpath(
            root=root,
            environment=environment,
            candidate_path="/session",
            container=True,
        ),
        container_name,
        "python",
        f"/session/evaluator/{task.evaluator.name}",
        "--solution",
        module,
    ]


def _local_command(task: TaskConfig, module: str | None = None) -> list[str]:
    return [
        sys.executable,
        str(task.evaluator),
        "--solution",
        module or task.module,
    ]


def dry_run_command(
    *,
    root: Path,
    task: TaskConfig,
    environment: EnvironmentConfig,
    engine: str,
    module: str | None = None,
) -> list[str]:
    if engine == "local":
        return _local_command(task, module)
    return _docker_exec_command(
        root=root,
        task=task,
        environment=environment,
        container_name=f"orbitbench-{task.id}-dry-run",
        module=module or task.module,
    )


def dry_run_container_command(
    *,
    root: Path,
    task: TaskConfig,
    environment: EnvironmentConfig,
) -> list[str]:
    return _docker_container_start_command(
        root=root,
        staging_dir="<staged-candidate-dir>",
        task=task,
        environment=environment,
        container_name=f"orbitbench-{task.id}-dry-run",
    )


def inspect_docker_image(image: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"reference": image}
    if not rows or not isinstance(rows[0], dict):
        return {"reference": image}
    row = rows[0]
    return {
        "reference": image,
        "id": row.get("Id"),
        "repo_digests": row.get("RepoDigests") or [],
        "created": row.get("Created"),
        "architecture": row.get("Architecture"),
        "os": row.get("Os"),
    }


def docker_build_command(root: Path, environment: EnvironmentConfig) -> list[str]:
    return [
        "docker",
        "build",
        "--file",
        str(environment.dockerfile),
        "--tag",
        environment.image,
        str(root.resolve()),
    ]


def build_environment(
    root: Path,
    environment: EnvironmentConfig,
    *,
    dry_run: bool = False,
) -> list[str]:
    if not environment.dockerfile.is_file():
        raise BenchmarkError(f"Dockerfile not found: {environment.dockerfile}")
    if not environment.requirements.is_file():
        raise BenchmarkError(f"Requirements file not found: {environment.requirements}")
    command = docker_build_command(root, environment)
    if dry_run:
        return command
    result = run_process(
        command,
        cwd=root,
        env=os.environ,
        timeout_sec=3600,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BenchmarkError(
            f"Failed to build Docker image {environment.image}: {detail[-4000:]}"
        )
    return command


def ensure_docker_image(
    root: Path,
    environment: EnvironmentConfig,
    *,
    no_build: bool,
) -> dict[str, Any]:
    image = inspect_docker_image(environment.image)
    if image is not None:
        return image
    if no_build:
        raise BenchmarkError(
            f"Docker image {environment.image!r} is missing and --no-build was set"
        )
    build_environment(root, environment)
    image = inspect_docker_image(environment.image)
    if image is None:
        raise BenchmarkError(
            f"Docker image {environment.image!r} is still unavailable after build"
        )
    return image


def _cleanup_container(name: str) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "rm", "-f", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _stage_candidate(source: Path, module: str, staging_dir: Path) -> Path:
    destination = staging_dir.joinpath(*module.split(".")).with_suffix(".py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _snapshot_sha256(files: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(sorted(files.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _environment_file_hashes(environment: EnvironmentConfig) -> dict[str, str | None]:
    sitecustomize = environment.dockerfile.parent / "sitecustomize.py"
    compatibility_sha256 = (
        file_sha256(sitecustomize) if sitecustomize.is_file() else None
    )
    return {
        "dockerfile_sha256": (
            file_sha256(environment.dockerfile)
            if environment.dockerfile.is_file()
            else None
        ),
        "requirements_sha256": (
            file_sha256(environment.requirements)
            if environment.requirements.is_file()
            else None
        ),
        "sitecustomize_sha256": compatibility_sha256,
        "compatibility_sha256": compatibility_sha256,
    }


def staged_module_name(
    task: TaskConfig,
    solution_name: str,
    source: Path,
) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", task.module).strip("_") or "solution"
    label = (
        re.sub(r"[^A-Za-z0-9_]+", "_", solution_name).strip("_")
        or "candidate"
    )
    identity = hashlib.sha256(
        f"{solution_name}\0{source.expanduser().resolve()}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{base}__orbitbench_{label}_{identity}"


def _measurement_result(
    *,
    root: Path,
    task: TaskConfig,
    source: Path,
    solution_name: str,
    solution_kind: str,
    environment: EnvironmentConfig,
    engine: str,
    timeout_sec: float,
    repeat_index: int,
    planned_repeats: int,
    logs_dir: Path,
    image_provenance: Mapping[str, Any] | None,
    comparison_role: str | None,
    process_result: ProcessResult,
    command: Sequence[str],
    started_at: str,
    staging_root: Path,
    staging_dir: Path,
    module: str,
    pair_order: str | None,
    pair_position: int | None,
    shared_container_name: str | None,
    shared_container_id: str | None,
    shared_container_start_command: Sequence[str] | None,
    staged_solution_hashes: Mapping[str, str] | None = None,
    staged_evaluator_sha256: str | None = None,
    staged_sitecustomize_sha256: str | None = None,
    staging_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    combined = process_result.stdout + "\n" + process_result.stderr
    parsed = parse_evaluator_output(combined)
    status = terminal_status(process_result, parsed)
    passed = bool(
        process_result.returncode == 0
        and not process_result.timed_out
        and parsed.passed_marker
        and parsed.runtime_sec is not None
        and math.isfinite(parsed.runtime_sec)
        and parsed.runtime_sec > 0
    )

    solution_slug = re.sub(
        r"[^a-z0-9_.-]+",
        "-",
        solution_name.lower(),
    ).strip("-") or "solution"
    stem = f"challenge-{task.id}-{solution_slug}-repeat-{repeat_index:03d}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{stem}.stdout.log"
    stderr_path = logs_dir / f"{stem}.stderr.log"
    stdout_path.write_text(process_result.stdout, encoding="utf-8")
    stderr_path.write_text(process_result.stderr, encoding="utf-8")
    staged = staging_dir.joinpath(*module.split(".")).with_suffix(".py")
    environment_hashes = _environment_file_hashes(environment)
    compatibility_sha256 = (
        staged_sitecustomize_sha256
        if staged_sitecustomize_sha256 is not None
        else environment_hashes["compatibility_sha256"]
    )

    return {
        "task_id": task.id,
        "title": task.title,
        "solution": solution_name,
        "solution_kind": solution_kind,
        "comparison_role": comparison_role,
        "pair_id": repeat_index if comparison_role is not None else None,
        "pair_order": pair_order,
        "pair_position": pair_position,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "staging_dir": str(staging_dir),
        "staging_root": str(staging_root),
        "staged_module_path": str(staged.relative_to(staging_dir)),
        "staged_source_sha256": file_sha256(staged),
        "staged_solution_hashes": dict(staged_solution_hashes or {}),
        "staged_evaluator_sha256": staged_evaluator_sha256,
        "staged_sitecustomize_sha256": staged_sitecustomize_sha256,
        "staging_snapshot_sha256": staging_snapshot_sha256,
        "task_manifest": str(task.manifest_path),
        "task_manifest_sha256": file_sha256(task.manifest_path),
        "evaluator_path": str(task.evaluator),
        "evaluator_sha256": file_sha256(task.evaluator),
        "environment": task.environment,
        "environment_image": environment.image,
        "environment_image_provenance": dict(image_provenance or {}),
        **environment_hashes,
        "compatibility_source_sha256": environment_hashes[
            "compatibility_sha256"
        ],
        "compatibility_sha256": compatibility_sha256,
        "engine": engine,
        "repeat": repeat_index,
        "planned_repeats": planned_repeats,
        "timeout_sec": timeout_sec,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_sec": parsed.runtime_sec,
        "controller_wall_sec": process_result.wall_sec,
        "passed": passed,
        "pass_marker": parsed.passed_marker,
        "timed_out": process_result.timed_out,
        "returncode": process_result.returncode,
        "terminal_status": status,
        "shared_container_name": shared_container_name,
        "shared_container_id": shared_container_id,
        "shared_session_id": shared_container_id,
        "shared_container_start_command": (
            list(shared_container_start_command)
            if shared_container_start_command is not None
            else None
        ),
        "command": list(command),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


class DockerTaskSession:
    """One task container with a fresh evaluator process per measurement."""

    def __init__(
        self,
        *,
        root: Path,
        task: TaskConfig,
        environment: EnvironmentConfig,
        targets: Mapping[str, Path],
        image_provenance: Mapping[str, Any] | None,
    ) -> None:
        self.root = root.resolve()
        self.task = task
        self.environment = environment
        self.targets = {
            name: path.expanduser().resolve() for name, path in targets.items()
        }
        self.image_provenance = dict(image_provenance or {})
        self.container_name = (
            f"orbitbench-{task.id}-{uuid.uuid4().hex[:12]}".lower()
        )
        self.container_id: str | None = None
        self.start_command: list[str] | None = None
        self.staging_root = self.root / ".tmp"
        self.staging_dir: Path | None = None
        self.modules: dict[str, str] = {}
        self.staged_solution_hashes: dict[str, str] = {}
        self.staged_evaluator_sha256: str | None = None
        self.staged_sitecustomize_sha256: str | None = None
        self.staging_snapshot_sha256: str | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._container_may_exist = False
        self._closed = False

    def __enter__(self) -> DockerTaskSession:
        for name, source in self.targets.items():
            if not source.is_file():
                raise BenchmarkError(
                    f"Solution {name!r} does not exist: {source}"
                )
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"orbitbench-{self.task.id}-",
            dir=self.staging_root,
        )
        self.staging_dir = Path(self._temporary_directory.name).resolve()
        try:
            for name, source in self.targets.items():
                module = staged_module_name(self.task, name, source)
                self.modules[name] = module
                staged = _stage_candidate(source, module, self.staging_dir)
                self.staged_solution_hashes[name] = file_sha256(staged)
            staged_evaluator = self.staging_dir / "evaluator" / self.task.evaluator.name
            staged_evaluator.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.task.evaluator, staged_evaluator)
            self.staged_evaluator_sha256 = file_sha256(staged_evaluator)
            sitecustomize = self.environment.dockerfile.parent / "sitecustomize.py"
            if sitecustomize.is_file():
                staged_sitecustomize = (
                    self.staging_dir / "environment" / "sitecustomize.py"
                )
                staged_sitecustomize.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sitecustomize, staged_sitecustomize)
                self.staged_sitecustomize_sha256 = file_sha256(
                    staged_sitecustomize
                )
            snapshot_files = {
                f"solution:{name}": digest
                for name, digest in self.staged_solution_hashes.items()
            }
            snapshot_files["evaluator"] = self.staged_evaluator_sha256
            if self.staged_sitecustomize_sha256 is not None:
                snapshot_files["sitecustomize"] = (
                    self.staged_sitecustomize_sha256
                )
            self.staging_snapshot_sha256 = _snapshot_sha256(snapshot_files)
            command = _docker_container_start_command(
                root=self.root,
                staging_dir=self.staging_dir,
                task=self.task,
                environment=self.environment,
                container_name=self.container_name,
            )
            self.start_command = list(command)
            self._container_may_exist = True
            result = run_process(
                command,
                cwd=self.staging_dir,
                env=os.environ,
                timeout_sec=60,
            )
            if result.timed_out:
                raise BenchmarkError(
                    f"Timed out starting task container {self.container_name}"
                )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise BenchmarkError(
                    f"Could not start task container {self.container_name}: "
                    f"{detail[-4000:]}"
                )
            container_id = result.stdout.strip().splitlines()
            if not container_id:
                raise BenchmarkError(
                    f"Docker did not return an id for {self.container_name}"
                )
            self.container_id = container_id[-1]
            return self
        except BaseException:
            self.close(raise_on_error=False)
            self._cleanup_staging()
            raise

    def _cleanup_staging(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def close(self, *, raise_on_error: bool) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_ok = True
        if self._container_may_exist:
            cleanup_ok = _cleanup_container(self.container_name)
            self._container_may_exist = False
        if not cleanup_ok and raise_on_error:
            raise BenchmarkError(
                f"Could not remove task container {self.container_name}"
            )

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close(raise_on_error=exc_type is None)
        finally:
            self._cleanup_staging()
        return False

    @property
    def active(self) -> bool:
        return not self._closed and self.container_id is not None

    def run_measurement(
        self,
        *,
        solution_name: str,
        solution_kind: str,
        comparison_role: str | None,
        timeout_sec: float,
        repeat_index: int,
        planned_repeats: int,
        logs_dir: Path,
        pair_order: str | None,
        pair_position: int | None,
    ) -> dict[str, Any]:
        if timeout_sec <= 0:
            raise BenchmarkError("Measurement timeout must be positive")
        timeout_sec = min(float(timeout_sec), 300.0)
        if not self.active or self.staging_dir is None:
            raise BenchmarkError(
                f"Task container {self.container_name} is not active"
            )
        if solution_name not in self.targets or solution_name not in self.modules:
            raise BenchmarkError(
                f"Solution {solution_name!r} was not staged for task {self.task.id}"
            )
        module = self.modules[solution_name]
        command = _docker_exec_command(
            root=self.root,
            task=self.task,
            environment=self.environment,
            container_name=self.container_name,
            module=module,
        )
        started_at = utc_now()
        try:
            process_result = run_process(
                command,
                cwd=self.staging_dir,
                env=os.environ,
                timeout_sec=timeout_sec,
            )
            result = _measurement_result(
                root=self.root,
                task=self.task,
                source=self.targets[solution_name],
                solution_name=solution_name,
                solution_kind=solution_kind,
                environment=self.environment,
                engine="docker",
                timeout_sec=timeout_sec,
                repeat_index=repeat_index,
                planned_repeats=planned_repeats,
                logs_dir=logs_dir,
                image_provenance=self.image_provenance,
                comparison_role=comparison_role,
                process_result=process_result,
                command=command,
                started_at=started_at,
                staging_root=self.staging_root,
                staging_dir=self.staging_dir,
                module=module,
                pair_order=pair_order,
                pair_position=pair_position,
                shared_container_name=self.container_name,
                shared_container_id=self.container_id,
                shared_container_start_command=self.start_command,
                staged_solution_hashes=self.staged_solution_hashes,
                staged_evaluator_sha256=self.staged_evaluator_sha256,
                staged_sitecustomize_sha256=self.staged_sitecustomize_sha256,
                staging_snapshot_sha256=self.staging_snapshot_sha256,
            )
        except BaseException:
            self.close(raise_on_error=False)
            raise
        if process_result.timed_out:
            self.close(raise_on_error=False)
        return result


def run_repetition(
    *,
    root: Path,
    task: TaskConfig,
    source: Path,
    solution_name: str,
    solution_kind: str,
    environment: EnvironmentConfig,
    engine: str,
    timeout_sec: float,
    repeat_index: int,
    planned_repeats: int = 1,
    logs_dir: Path,
    image_provenance: Mapping[str, Any] | None,
    comparison_role: str | None = None,
    pair_order: str | None = None,
    pair_position: int | None = None,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        raise BenchmarkError("Measurement timeout must be positive")
    timeout_sec = min(float(timeout_sec), 300.0)
    source = source.expanduser().resolve()
    if not source.is_file():
        raise BenchmarkError(f"Candidate does not exist: {source}")
    started_at = utc_now()
    if engine != "local":
        raise BenchmarkError(
            "Docker measurements require DockerTaskSession"
        )

    staging_root = root.resolve() / ".tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"orbitbench-{task.id}-",
        dir=staging_root,
    ) as temp:
        staging_dir = Path(temp).resolve()
        _stage_candidate(source, task.module, staging_dir)
        command = _local_command(task)
        process_env = dict(os.environ)
        process_env["PYTHONPATH"] = _environment_pythonpath(
            root=root,
            environment=environment,
            candidate_path=str(staging_dir),
            container=False,
        )
        process_env["NUMBA_DISABLE_JIT"] = "1"

        process_result = run_process(
            command,
            cwd=staging_dir,
            env=process_env,
            timeout_sec=timeout_sec,
        )
        return _measurement_result(
            root=root,
            task=task,
            source=source,
            solution_name=solution_name,
            solution_kind=solution_kind,
            environment=environment,
            engine="local",
            timeout_sec=timeout_sec,
            repeat_index=repeat_index,
            planned_repeats=planned_repeats,
            logs_dir=logs_dir,
            image_provenance=image_provenance,
            comparison_role=comparison_role,
            process_result=process_result,
            command=command,
            started_at=started_at,
            staging_root=staging_root,
            staging_dir=staging_dir,
            module=task.module,
            pair_order=pair_order,
            pair_position=pair_position,
            shared_container_name=None,
            shared_container_id=None,
            shared_container_start_command=None,
            staged_solution_hashes={
                solution_name: file_sha256(
                    staging_dir.joinpath(*task.module.split(".")).with_suffix(
                        ".py"
                    )
                )
            },
        )


def _runtime_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "stderr": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "stderr": (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else None
        ),
        "min": min(values),
        "max": max(values),
    }


def _rows_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passing = [
        float(row["runtime_sec"])
        for row in rows
        if row.get("passed") and row.get("runtime_sec") is not None
    ]
    return {
        "runs": len(rows),
        "passing_runs": len(passing),
        "failed_runs": sum(not bool(row.get("passed")) for row in rows),
        "timed_out_runs": sum(bool(row.get("timed_out")) for row in rows),
        "runtime_sec": _runtime_stats(passing),
    }


def _comparison_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    reference_rows = [
        row for row in rows if str(row.get("solution")) == reference_name
    ]
    candidate_rows = [
        row for row in rows if str(row.get("solution")) == candidate_name
    ]
    reference = _rows_summary(reference_rows)
    candidate = _rows_summary(candidate_rows)
    reference_mean = reference["runtime_sec"]["mean"]
    candidate_mean = candidate["runtime_sec"]["mean"]
    reference_median = reference["runtime_sec"]["median"]
    candidate_median = candidate["runtime_sec"]["median"]
    selected_rows = [*reference_rows, *candidate_rows]
    all_runs_passed = bool(reference_rows and candidate_rows) and all(
        bool(row.get("passed")) for row in selected_rows
    )
    balanced_runs = len(reference_rows) == len(candidate_rows)
    positive_means = bool(
        isinstance(reference_mean, (int, float))
        and not isinstance(reference_mean, bool)
        and isinstance(candidate_mean, (int, float))
        and not isinstance(candidate_mean, bool)
        and math.isfinite(float(reference_mean))
        and math.isfinite(float(candidate_mean))
        and reference_mean > 0
        and candidate_mean > 0
    )
    pair_groups: dict[
        tuple[str, int],
        dict[str, list[Mapping[str, Any]]],
    ] = {}
    planned_by_task: dict[str, int] = {}
    task_ids = {str(row.get("task_id")) for row in selected_rows}
    pair_metadata_valid = True
    docker_sessions: dict[str, set[str]] = {}
    docker_snapshots: dict[str, set[str]] = {}

    def positive_integer(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
            return None
        return parsed

    for row in selected_rows:
        task_id = str(row.get("task_id"))
        solution = str(row.get("solution"))
        expected_role = (
            "reference" if solution == reference_name else "candidate"
        )
        role = row.get("comparison_role")
        pair_id = positive_integer(row.get("pair_id"))
        planned = positive_integer(row.get("planned_repeats"))
        repeat = positive_integer(row.get("repeat"))
        if (
            role not in {"reference", "candidate"}
            or role != expected_role
            or pair_id is None
            or planned is None
            or repeat != pair_id
            or pair_id > planned
        ):
            pair_metadata_valid = False
            continue
        previous_planned = planned_by_task.get(task_id)
        if previous_planned is not None and previous_planned != planned:
            pair_metadata_valid = False
        planned_by_task[task_id] = planned
        group = pair_groups.setdefault(
            (task_id, pair_id),
            {"reference": [], "candidate": []},
        )
        group[str(role)].append(row)
        if row.get("engine") == "docker":
            container_id = row.get("shared_container_id")
            snapshot_hash = row.get("staging_snapshot_sha256")
            if not isinstance(container_id, str) or not container_id:
                pair_metadata_valid = False
            else:
                docker_sessions.setdefault(task_id, set()).add(container_id)
            if not isinstance(snapshot_hash, str) or not snapshot_hash:
                pair_metadata_valid = False
            else:
                docker_snapshots.setdefault(task_id, set()).add(snapshot_hash)
    if task_ids != set(planned_by_task):
        pair_metadata_valid = False
    if any(len(ids) != 1 for ids in docker_sessions.values()):
        pair_metadata_valid = False
    if any(len(hashes) != 1 for hashes in docker_snapshots.values()):
        pair_metadata_valid = False
    for task_id in task_ids:
        planned = planned_by_task.get(task_id)
        if planned is None:
            continue
        for pair_id in range(1, planned + 1):
            pair_groups.setdefault(
                (task_id, pair_id),
                {"reference": [], "candidate": []},
            )

    pairs: list[dict[str, Any]] = []
    paired_improvements: list[float] = []
    paired_speedups: list[float] = []
    for (task_id, pair_id), group in sorted(pair_groups.items()):
        reference_pair = group["reference"]
        candidate_pair = group["candidate"]
        exactly_one_each = len(reference_pair) == 1 and len(candidate_pair) == 1
        reference_row = reference_pair[0] if len(reference_pair) == 1 else None
        candidate_row = candidate_pair[0] if len(candidate_pair) == 1 else None

        def eligible_runtime(row: Mapping[str, Any] | None) -> float | None:
            if row is None:
                return None
            value = row.get("runtime_sec")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                return None
            return float(value)

        expected_order = (
            "reference->candidate"
            if pair_id % 2 == 1
            else "candidate->reference"
        )
        expected_reference_position = 1 if pair_id % 2 == 1 else 2
        expected_candidate_position = 2 if pair_id % 2 == 1 else 1
        reference_runtime = (
            eligible_runtime(reference_row)
        )
        candidate_runtime = eligible_runtime(candidate_row)
        ordered_pair = bool(
            exactly_one_each
            and reference_row is not None
            and candidate_row is not None
            and reference_row.get("pair_order") == expected_order
            and candidate_row.get("pair_order") == expected_order
            and reference_row.get("pair_position")
            == expected_reference_position
            and candidate_row.get("pair_position")
            == expected_candidate_position
        )
        pair_passed = bool(
            exactly_one_each
            and reference_row is not None
            and candidate_row is not None
            and reference_row.get("passed")
            and candidate_row.get("passed")
            and reference_runtime is not None
            and candidate_runtime is not None
            and ordered_pair
        )
        pair_improvement = (
            100.0
            * (reference_runtime - candidate_runtime)
            / reference_runtime
            if pair_passed
            else None
        )
        pair_speedup = (
            reference_runtime / candidate_runtime if pair_passed else None
        )
        if pair_improvement is not None and pair_speedup is not None:
            paired_improvements.append(pair_improvement)
            paired_speedups.append(pair_speedup)
        pairs.append(
            {
                "task_id": task_id,
                "pair_id": pair_id,
                "pair_order": expected_order,
                "reference_rows": len(reference_pair),
                "candidate_rows": len(candidate_pair),
                "reference_position": (
                    reference_row.get("pair_position")
                    if reference_row is not None
                    else None
                ),
                "candidate_position": (
                    candidate_row.get("pair_position")
                    if candidate_row is not None
                    else None
                ),
                "reference_runtime_sec": reference_runtime,
                "candidate_runtime_sec": candidate_runtime,
                "ordered_pair": ordered_pair,
                "eligible": pair_passed,
                "improvement_pct": pair_improvement,
                "speedup": pair_speedup,
            }
        )

    pairing_valid = bool(
        pair_metadata_valid
        and pairs
        and all(pair["eligible"] for pair in pairs)
    )
    eligible = (
        all_runs_passed
        and balanced_runs
        and positive_means
        and pairing_valid
    )
    improvement_pct = (
        100.0 * (reference_mean - candidate_mean) / reference_mean
        if eligible
        else None
    )
    speedup = reference_mean / candidate_mean if eligible else None
    paired_improvement = _runtime_stats(
        paired_improvements if eligible else []
    )
    paired_speedup = _runtime_stats(paired_speedups if eligible else [])
    return {
        "reference_solution": reference_name,
        "candidate_solution": candidate_name,
        "reference": reference,
        "candidate": candidate,
        "reference_mean": reference_mean,
        "reference_median": reference_median,
        "reference_stderr": reference["runtime_sec"]["stderr"],
        "candidate_mean": candidate_mean,
        "candidate_median": candidate_median,
        "candidate_stderr": candidate["runtime_sec"]["stderr"],
        "all_runs_passed": all_runs_passed,
        "balanced_runs": balanced_runs,
        "positive_means": positive_means,
        "pair_metadata_valid": pair_metadata_valid,
        "pairing_valid": pairing_valid,
        "pairs": pairs,
        "eligible": eligible,
        "improvement_pct": improvement_pct,
        "speedup": speedup,
        "paired_improvement_pct": paired_improvement,
        "paired_improvement_mean": paired_improvement["mean"],
        "paired_improvement_stderr": paired_improvement["stderr"],
        "paired_speedup": paired_speedup,
        "speedup_mean": paired_speedup["mean"],
        "speedup_stderr": paired_speedup["stderr"],
    }


def aggregate_results(
    results: Iterable[Mapping[str, Any]],
    *,
    comparison: tuple[str, str] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_id")), []).append(row)

    tasks: dict[str, dict[str, Any]] = {}
    all_passing_runtimes: list[float] = []
    for task_id, task_rows in sorted(grouped.items()):
        task_summary = _rows_summary(task_rows)
        passing = [
            float(row["runtime_sec"])
            for row in task_rows
            if row.get("passed") and row.get("runtime_sec") is not None
        ]
        all_passing_runtimes.extend(passing)
        solution_groups: dict[str, list[dict[str, Any]]] = {}
        for row in task_rows:
            solution_groups.setdefault(str(row.get("solution")), []).append(row)
        task_summary["solutions"] = {
            name: _rows_summary(solution_rows)
            for name, solution_rows in sorted(solution_groups.items())
        }
        if comparison is not None:
            reference_name, candidate_name = comparison
            task_summary["comparison"] = _comparison_summary(
                task_rows,
                reference_name=reference_name,
                candidate_name=candidate_name,
            )
        tasks[task_id] = task_summary

    summary = {
        "runs": len(rows),
        "passing_runs": sum(bool(row.get("passed")) for row in rows),
        "failed_runs": sum(not bool(row.get("passed")) for row in rows),
        "timed_out_runs": sum(bool(row.get("timed_out")) for row in rows),
        "runtime_sec": _runtime_stats(all_passing_runtimes),
        "tasks": tasks,
    }
    if comparison is not None:
        reference_name, candidate_name = comparison
        comparison_summary = _comparison_summary(
            rows,
            reference_name=reference_name,
            candidate_name=candidate_name,
        )
        comparison_summary["tasks"] = {
            task_id: task_summary["comparison"]
            for task_id, task_summary in tasks.items()
        }
        comparison_summary["eligible"] = bool(
            comparison_summary["eligible"]
            and all(
                task_comparison["eligible"]
                for task_comparison in comparison_summary["tasks"].values()
            )
        )
        if not comparison_summary["eligible"]:
            comparison_summary["improvement_pct"] = None
            comparison_summary["speedup"] = None
            empty_stats = _runtime_stats([])
            comparison_summary["paired_improvement_pct"] = empty_stats
            comparison_summary["paired_improvement_mean"] = None
            comparison_summary["paired_improvement_stderr"] = None
            comparison_summary["paired_speedup"] = empty_stats
            comparison_summary["speedup_mean"] = None
            comparison_summary["speedup_stderr"] = None
        comparison_summary["pooled_runtime_ratio_reported"] = len(tasks) == 1
        comparison_summary["pooled_pair_metrics_reported"] = len(tasks) == 1
        if len(tasks) != 1:
            comparison_summary["improvement_pct"] = None
            comparison_summary["speedup"] = None
            empty_stats = _runtime_stats([])
            comparison_summary["paired_improvement_pct"] = empty_stats
            comparison_summary["paired_improvement_mean"] = None
            comparison_summary["paired_improvement_stderr"] = None
            comparison_summary["paired_speedup"] = empty_stats
            comparison_summary["speedup_mean"] = None
            comparison_summary["speedup_stderr"] = None
        summary["comparison"] = comparison_summary
    return summary


def output_paths(root: Path, requested: Path | None) -> tuple[Path, Path, Path]:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory = root / "results" / stamp
        return directory / "results.json", directory / "summary.json", directory / "logs"

    requested = requested.expanduser()
    if not requested.is_absolute():
        requested = (Path.cwd() / requested).resolve()
    if requested.suffix.lower() == ".json":
        report = requested
        summary = requested.with_name(f"{requested.stem}.summary.json")
        logs = requested.with_name(f"{requested.stem}.logs")
        return report, summary, logs
    return requested / "results.json", requested / "summary.json", requested / "logs"


def ensure_output_paths_available(
    report_path: Path,
    summary_path: Path,
    logs_dir: Path,
) -> None:
    evidence_paths = (report_path, summary_path, logs_dir)
    conflicts = [path for path in evidence_paths if os.path.lexists(path)]
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise BenchmarkError(
            "Output evidence already exists; choose a new --output path: "
            f"{rendered}"
        )
    for path in evidence_paths:
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if parent.exists() and not parent.is_dir():
            raise BenchmarkError(
                f"Output parent is not a directory: {parent}"
            )


def write_report(
    *,
    report_path: Path,
    summary_path: Path,
    results: Sequence[Mapping[str, Any]],
    root: Path,
    invocation: Sequence[str],
    comparison: tuple[str, str] | None = None,
) -> dict[str, Any]:
    summary = aggregate_results(results, comparison=comparison)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "benchmark_root": str(root.resolve()),
        "invocation": list(invocation),
        "host": host_provenance(root),
        "results": [dict(result) for result in results],
        "summary": summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


_MEMORY_RE = re.compile(
    r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>b|k|kb|ki|kib|m|mb|mi|mib|g|gb|gi|gib|t|tb|ti|tib)?\s*$",
    re.IGNORECASE,
)
_MEMORY_MULTIPLIERS = {
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "ki": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gi": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "ti": 1024**4,
    "tib": 1024**4,
}


def requested_memory_bytes(value: object) -> int | None:
    """Normalize manifest memory using the runner's numeric-as-MiB convention."""

    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("memory must be a size, not a boolean")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("memory must be positive")
        return math.ceil(float(value) * 1024**2)
    match = _MEMORY_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"unsupported memory size {value!r}")
    number = float(match.group("value"))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("memory must be positive")
    unit = (match.group("unit") or "b").lower()
    return math.ceil(number * _MEMORY_MULTIPLIERS[unit])


def requested_cpus(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("cpus must be a number, not a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported CPU limit {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("cpus must be positive")
    return parsed


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    for unit, divisor in (
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value} B"


def _docker_resource_info() -> tuple[dict[str, Any] | None, str | None]:
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip() or "docker info failed"
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return None, f"docker info returned invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "docker info did not return an object"
    try:
        ncpu = float(payload["NCPU"])
        memory_bytes = int(payload["MemTotal"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"docker info omitted NCPU or MemTotal: {exc}"
    if not math.isfinite(ncpu) or ncpu <= 0 or memory_bytes <= 0:
        return None, "docker info reported non-positive NCPU or MemTotal"
    return {
        "cpus": ncpu,
        "memory_bytes": memory_bytes,
        "memory_human": _format_bytes(memory_bytes),
    }, None


def doctor_environment(config: BenchConfig) -> tuple[bool, dict[str, Any]]:
    docker_path = shutil.which("docker")
    report: dict[str, Any] = {
        "docker_executable": docker_path,
        "daemon_available": False,
        "docker_resources": None,
        "resource_errors": [],
        "environments": {},
    }
    if docker_path:
        try:
            completed = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
            report["daemon_available"] = completed.returncode == 0
            report["server_version"] = completed.stdout.strip() or None
            if completed.returncode != 0:
                report["docker_error"] = completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["docker_error"] = str(exc)

    resources: dict[str, Any] | None = None
    if report["daemon_available"]:
        resources, resource_probe_error = _docker_resource_info()
        report["docker_resources"] = resources
        if resource_probe_error is not None:
            report["resource_probe_error"] = resource_probe_error
            report["resource_errors"].append(
                f"Could not inspect Docker CPU and memory: {resource_probe_error}"
            )

    files_ok = True
    for name, environment in config.environments.items():
        dockerfile_exists = environment.dockerfile.is_file()
        requirements_exists = environment.requirements.is_file()
        files_ok = files_ok and dockerfile_exists and requirements_exists
        environment_report = {
            "image": environment.image,
            "image_present": inspect_docker_image(environment.image) is not None
            if report["daemon_available"]
            else False,
            "dockerfile": str(environment.dockerfile),
            "dockerfile_exists": dockerfile_exists,
            "requirements": str(environment.requirements),
            "requirements_exists": requirements_exists,
            "requested_cpus": environment.cpus,
            "requested_memory": environment.memory,
            "requested_memory_bytes": None,
            "requested_memory_human": None,
            "resource_ok": True,
            "resource_errors": [],
        }
        environment_errors: list[str] = environment_report["resource_errors"]
        try:
            cpu_request = requested_cpus(environment.cpus)
        except ValueError as exc:
            cpu_request = None
            environment_errors.append(f"invalid CPU request: {exc}")
        try:
            memory_request = requested_memory_bytes(environment.memory)
        except ValueError as exc:
            memory_request = None
            environment_errors.append(f"invalid memory request: {exc}")
        environment_report["requested_cpus_normalized"] = cpu_request
        environment_report["requested_memory_bytes"] = memory_request
        environment_report["requested_memory_human"] = _format_bytes(memory_request)
        if resources is not None:
            if cpu_request is not None and cpu_request > resources["cpus"]:
                environment_errors.append(
                    f"requests {cpu_request:g} CPUs but Docker has "
                    f"{resources['cpus']:g}"
                )
            if (
                memory_request is not None
                and memory_request > resources["memory_bytes"]
            ):
                environment_errors.append(
                    f"requests {_format_bytes(memory_request)} memory but Docker "
                    f"has {resources['memory_human']}"
                )
        elif report["daemon_available"]:
            environment_errors.append(
                "Docker resource availability could not be determined"
            )
        environment_report["resource_ok"] = not environment_errors
        for error in environment_errors:
            report["resource_errors"].append(f"Environment {name}: {error}")
        report["environments"][name] = environment_report
    ok = bool(
        docker_path
        and report["daemon_available"]
        and resources is not None
        and files_ok
        and not report["resource_errors"]
    )
    return ok, report
