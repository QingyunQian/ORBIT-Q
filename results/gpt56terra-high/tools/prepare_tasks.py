#!/usr/bin/env python3
"""Prepare isolated, resource-normalized Harbor task copies for one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


RESOURCE_VALUES = {"cpus": 6, "memory_mb": 10240, "storage_mb": 16384}


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def adapt_task_toml(path: Path) -> None:
    text = path.read_text()
    for name, value in RESOURCE_VALUES.items():
        pattern = rf"(?m)^{name} = \d+$"
        text, n_replacements = re.subn(pattern, f"{name} = {value}", text)
        if n_replacements != 1:
            raise ValueError(f"expected one {name!r} declaration in {path}")
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()

    tasks_root = args.repo_root / "tasks"
    if args.run_root.exists():
        if any(args.run_root.iterdir()):
            raise FileExistsError(f"run root must be empty: {args.run_root}")
    else:
        args.run_root.mkdir(parents=True)
    records = []
    for task_number in range(1, 13):
        challenge = f"challenge-{task_number:02d}"
        source = tasks_root / challenge
        destination = args.run_root / challenge
        shutil.copytree(source, destination)
        adapt_task_toml(destination / "task.toml")
        records.append(
            {
                "challenge": challenge,
                "canonical_tree_sha256": tree_sha256(source),
                "execution_copy_tree_sha256": tree_sha256(destination),
                "changed_files": ["task.toml"],
            }
        )

    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "base_commit": args.base_commit,
        "canonical_tasks_root": str(tasks_root),
        "task_copy_root": str(args.run_root),
        "resource_adapter": RESOURCE_VALUES,
        "aggregate_task_copy_sha256": aggregate,
        "tasks": records,
    }
    (args.run_root / "task-copy-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
