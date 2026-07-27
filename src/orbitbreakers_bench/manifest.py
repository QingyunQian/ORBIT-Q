from __future__ import annotations

import ast
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


TASK_IDS = tuple(f"{number:02d}" for number in range(1, 13))
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class ManifestError(ValueError):
    """Raised when benchmark metadata is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    image: str
    dockerfile: Path
    requirements: Path
    cpus: str | int | float | None
    memory: str | int | float | None


@dataclass(frozen=True)
class BenchConfig:
    root: Path
    default_environment: str
    default_timeout_sec: float
    default_repeats: int
    environments: Mapping[str, EnvironmentConfig]
    path: Path


@dataclass(frozen=True)
class SolutionConfig:
    name: str
    path: Path
    kind: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TaskConfig:
    id: str
    title: str
    environment: str
    evaluator: Path
    module: str
    timeout_sec: float
    solutions: tuple[SolutionConfig, ...]
    directory: Path
    manifest_path: Path
    raw: Mapping[str, Any]

    def solution(self, name: str) -> SolutionConfig:
        for solution in self.solutions:
            if solution.name == name:
                return solution
        names = ", ".join(solution.name for solution in self.solutions) or "none"
        raise ManifestError(
            f"Task {self.id} has no solution named {name!r}; available: {names}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"Invalid TOML in {path}: {exc}") from exc


def _normalize_task_id(value: Any, path: Path) -> str:
    text = str(value).strip()
    if text.startswith("challenge-"):
        text = text.removeprefix("challenge-")
    try:
        number = int(text)
    except ValueError as exc:
        raise ManifestError(f"Invalid task id {value!r} in {path}") from exc
    if number < 1 or number > 99:
        raise ManifestError(f"Invalid task id {value!r} in {path}")
    return f"{number:02d}"


def resolve_under_root(
    value: str | Path,
    *,
    root: Path,
    relative_to: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a manifest-controlled path and reject traversal outside ``root``."""

    root = root.resolve()
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        bases = [relative_to.resolve()] if relative_to is not None else []
        bases.append(root)
        candidates = [(base / raw).resolve() for base in bases]
        if must_exist:
            existing = next((path for path in candidates if path.exists()), None)
            candidate = existing if existing is not None else candidates[0]
        else:
            candidate = candidates[0]
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"Path escapes benchmark root: {value}") from exc
    if must_exist and not candidate.exists():
        raise ManifestError(f"Path does not exist: {candidate}")
    return candidate


def load_bench_config(root: Path) -> BenchConfig:
    root = root.expanduser().resolve()
    path = root / "bench.toml"
    raw = _read_toml(path)

    default_environment = str(raw.get("default_environment", "")).strip()
    if not default_environment:
        raise ManifestError(f"Missing default_environment in {path}")
    try:
        default_timeout_sec = float(raw.get("default_timeout_sec", 300))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Invalid default_timeout_sec in {path}") from exc
    if default_timeout_sec <= 0:
        raise ManifestError(f"default_timeout_sec must be positive in {path}")
    try:
        default_repeats = int(raw.get("default_repeats", 3))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Invalid default_repeats in {path}") from exc
    if default_repeats <= 0:
        raise ManifestError(f"default_repeats must be positive in {path}")

    environment_rows = raw.get("environments")
    if not isinstance(environment_rows, dict) or not environment_rows:
        raise ManifestError(f"Missing [environments.*] entries in {path}")

    environments: dict[str, EnvironmentConfig] = {}
    for name, row in environment_rows.items():
        if not isinstance(row, dict):
            raise ManifestError(f"Environment {name!r} must be a table in {path}")
        image = str(row.get("image", "")).strip()
        dockerfile_value = row.get("dockerfile")
        requirements_value = row.get("requirements")
        if not image or not dockerfile_value or not requirements_value:
            raise ManifestError(
                f"Environment {name!r} requires image, dockerfile, and requirements"
            )
        environments[str(name)] = EnvironmentConfig(
            name=str(name),
            image=image,
            dockerfile=resolve_under_root(
                str(dockerfile_value), root=root, relative_to=root
            ),
            requirements=resolve_under_root(
                str(requirements_value), root=root, relative_to=root
            ),
            cpus=row.get("cpus"),
            memory=row.get("memory"),
        )

    if default_environment not in environments:
        raise ManifestError(
            f"default_environment {default_environment!r} is not configured"
        )
    return BenchConfig(
        root=root,
        default_environment=default_environment,
        default_timeout_sec=default_timeout_sec,
        default_repeats=default_repeats,
        environments=environments,
        path=path,
    )


def load_task_manifest(
    path: Path,
    *,
    root: Path,
    bench_config: BenchConfig,
) -> TaskConfig:
    path = path.resolve()
    raw = _read_toml(path)
    directory = path.parent

    required = ("id", "title", "evaluator", "module")
    missing = [key for key in required if raw.get(key) in (None, "")]
    if missing:
        raise ManifestError(f"Missing {', '.join(missing)} in {path}")

    task_id = _normalize_task_id(raw["id"], path)
    title = str(raw["title"]).strip()
    environment = str(
        raw.get("environment") or bench_config.default_environment
    ).strip()
    if environment not in bench_config.environments:
        raise ManifestError(
            f"Task {task_id} references unknown environment {environment!r}"
        )
    module = str(raw["module"]).strip()
    if not MODULE_RE.fullmatch(module):
        raise ManifestError(f"Invalid Python module {module!r} in {path}")
    try:
        timeout_sec = float(
            raw.get("timeout_sec", bench_config.default_timeout_sec)
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Invalid timeout_sec in {path}") from exc
    if timeout_sec <= 0:
        raise ManifestError(f"timeout_sec must be positive in {path}")

    evaluator = resolve_under_root(
        str(raw["evaluator"]), root=root, relative_to=directory
    )
    rows = raw.get("solutions", [])
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"Task {task_id} has no [[solutions]] entries")

    solutions: list[SolutionConfig] = []
    seen_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ManifestError(f"Invalid [[solutions]] entry in {path}")
        name = str(row.get("name", "")).strip()
        solution_path = row.get("path")
        kind = str(row.get("kind", "")).strip()
        if not name or not solution_path or not kind:
            raise ManifestError(
                f"Every solution for task {task_id} requires name, path, and kind"
            )
        if name in seen_names:
            raise ManifestError(f"Duplicate solution {name!r} for task {task_id}")
        seen_names.add(name)
        solutions.append(
            SolutionConfig(
                name=name,
                path=resolve_under_root(
                    str(solution_path), root=root, relative_to=directory
                ),
                kind=kind,
                metadata=dict(row),
            )
        )

    return TaskConfig(
        id=task_id,
        title=title,
        environment=environment,
        evaluator=evaluator,
        module=module,
        timeout_sec=timeout_sec,
        solutions=tuple(solutions),
        directory=directory,
        manifest_path=path,
        raw=raw,
    )


def discover_tasks(root: Path, bench_config: BenchConfig | None = None) -> list[TaskConfig]:
    root = root.expanduser().resolve()
    config = bench_config or load_bench_config(root)
    manifest_paths = sorted((root / "tasks").glob("challenge-*/task.toml"))
    tasks = [
        load_task_manifest(path, root=root, bench_config=config)
        for path in manifest_paths
    ]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ManifestError(f"Duplicate task ids discovered: {ids}")
    return sorted(tasks, key=lambda task: task.id)


def defines_run_solution(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_solution"
        for node in tree.body
    )


def _declared_sha256(metadata: Mapping[str, Any]) -> str | None:
    for key in ("sha256", "provenance_sha256", "source_sha256"):
        value = metadata.get(key)
        if value:
            return str(value).strip().lower()
    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        for key in ("sha256", "source_sha256"):
            value = provenance.get(key)
            if value:
                return str(value).strip().lower()
    return None


def verify_benchmark(
    root: Path,
    *,
    expected_task_ids: Iterable[str] = TASK_IDS,
) -> list[str]:
    """Return all benchmark integrity errors without stopping at the first one."""

    root = root.expanduser().resolve()
    errors: list[str] = []
    try:
        config = load_bench_config(root)
    except ManifestError as exc:
        return [str(exc)]

    for environment in config.environments.values():
        if not environment.dockerfile.is_file():
            errors.append(
                f"Environment {environment.name}: missing Dockerfile "
                f"{environment.dockerfile}"
            )
        if not environment.requirements.is_file():
            errors.append(
                f"Environment {environment.name}: missing requirements "
                f"{environment.requirements}"
            )

    try:
        tasks = discover_tasks(root, config)
    except ManifestError as exc:
        return errors + [str(exc)]

    expected = tuple(sorted(expected_task_ids))
    actual = tuple(task.id for task in tasks)
    if actual != expected:
        errors.append(
            "Expected exactly tasks "
            f"{', '.join(expected)}; discovered {', '.join(actual) or 'none'}"
        )

    for task in tasks:
        if not task.evaluator.is_file():
            errors.append(f"Task {task.id}: missing evaluator {task.evaluator}")
        else:
            declared_evaluator_hash = task.raw.get("evaluator_sha256")
            if declared_evaluator_hash:
                declared_evaluator_hash = str(declared_evaluator_hash).strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", declared_evaluator_hash):
                    errors.append(
                        f"Task {task.id}: invalid evaluator_sha256 "
                        f"{declared_evaluator_hash!r}"
                    )
                else:
                    actual_evaluator_hash = file_sha256(task.evaluator)
                    if actual_evaluator_hash != declared_evaluator_hash:
                        errors.append(
                            f"Task {task.id}: evaluator sha256 mismatch "
                            f"(declared {declared_evaluator_hash}, "
                            f"actual {actual_evaluator_hash})"
                        )
        if task.environment not in config.environments:
            errors.append(
                f"Task {task.id}: unknown environment {task.environment!r}"
            )
        for solution in task.solutions:
            label = f"Task {task.id} solution {solution.name!r}"
            if not solution.path.is_file():
                errors.append(f"{label}: missing file {solution.path}")
                continue
            if not defines_run_solution(solution.path):
                errors.append(f"{label}: does not define run_solution")
            declared = _declared_sha256(solution.metadata)
            if declared is not None:
                if not re.fullmatch(r"[0-9a-f]{64}", declared):
                    errors.append(f"{label}: invalid declared sha256 {declared!r}")
                else:
                    actual_hash = file_sha256(solution.path)
                    if actual_hash != declared:
                        errors.append(
                            f"{label}: sha256 mismatch "
                            f"(declared {declared}, actual {actual_hash})"
                        )
    return errors
