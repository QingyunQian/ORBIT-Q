"""Archive the selected valid Grok 4.5/high campaign outcomes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "grok-4.5-high"
CAMPAIGN = Path(
    "/Users/qqy/Desktop/2026Project/ORBIT-Q/jobs/"
    "grok-4.5-high-solaudit-20260806-valid"
)
ATTEMPTS = {
    1: "r2", 2: "r2", 3: "r2", 4: "r2", 5: "r2", 6: "r2",
    7: "r3", 8: "r1", 9: "r1", 10: "r1", 11: "r1", 12: "r1",
}
MODEL_FAILURES = {
    1: {
        "classification": "valid_model_timeout_failure",
        "signal": "AgentTimeoutError / 1,800-second limit",
        "evidence": "Grok reached the 1,800-second Agent limit before creating solution_1.py.",
    },
    4: {
        "classification": "valid_model_timeout_failure",
        "signal": "AgentTimeoutError / 1,800-second limit",
        "evidence": "Grok reached the 1,800-second Agent limit before creating solution_4.py.",
    },
    8: {
        "classification": "valid_model_tool_use_failure",
        "signal": "SIGTERM / exit 143",
        "evidence": "Grok issued pkill -f 'python', matching its own agent wrapper/proxy chain.",
    },
    11: {
        "classification": "valid_model_tool_use_failure",
        "signal": "SIGTERM / exit 143",
        "evidence": (
            "Grok issued pkill -f \"train\\(params0\\)\" followed by "
            "pkill -f \"jax\"; the broad JAX match killed its own agent chain."
        ),
    },
}


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("[REDACTED]", "null"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
    finish = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (finish - begin).total_seconds()


def selected_trial(task: int) -> tuple[Path, Path]:
    attempt = ATTEMPTS[task]
    job = CAMPAIGN / "jobs" / (
        f"challenge-{task:02d}-tensorcircuit-grok-4.5-high-20260806-{attempt}"
    )
    trials = sorted(job.glob(f"challenge-{task:02d}__*"))
    if len(trials) != 1:
        raise RuntimeError(f"Expected one trial for task {task:02d}, found {trials}")
    return job, trials[0]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def main() -> None:
    task_rows = []
    for task in range(1, 13):
        job, trial = selected_trial(task)
        challenge_out = OUT / f"challenge-{task:02d}"
        if challenge_out.exists():
            shutil.rmtree(challenge_out)
        challenge_out.mkdir(parents=True, exist_ok=True)
        solution = trial / "artifacts" / "root" / f"solution_{task}.py"

        copies = {
            solution: challenge_out / f"solution_{task}.py",
            trial / "verifier" / "functional-stdout.txt": challenge_out / "functional-stdout-official.txt",
            trial / "verifier" / "test-stdout.txt": challenge_out / "verifier-test-stdout.txt",
            trial / "verifier" / "audit-details.json": challenge_out / "audit-details.json",
            trial / "verifier" / "reward.json": challenge_out / "reward.json",
            trial / "result.json": challenge_out / "job-result.json",
            trial / "config.json": challenge_out / "config.json",
            trial / "lock.json": challenge_out / "lock.json",
            trial / "agent" / "codex.txt": challenge_out / "agent-codex.log",
            trial / "agent" / "trajectory.json": challenge_out / "agent-trajectory.json",
            trial / "agent" / "xai-responses-proxy.jsonl": challenge_out / "xai-responses-proxy.jsonl",
            trial / "agent" / "sessions": challenge_out / "agent-sessions",
            trial / "trial.log": challenge_out / "trial.log",
            trial / "exception.txt": challenge_out / "exception.txt",
            trial / "artifacts" / "manifest.json": challenge_out / "artifact-manifest.json",
            CAMPAIGN / f"host-challenge-{task:02d}-{ATTEMPTS[task]}.log": challenge_out / "host-run.log",
        }
        for source, target in copies.items():
            copy_if_exists(source, target)

        result = load(trial / "result.json")
        reward = load(trial / "verifier" / "reward.json")
        config = result["config"]
        agent_result = result.get("agent_result") or {}
        agent_interval = result.get("agent_execution") or {}
        exception = result.get("exception_info") or {}
        failure = MODEL_FAILURES.get(task)
        stamp = {
            "challenge": task,
            "selected_attempt": ATTEMPTS[task],
            "job_name": job.name,
            "trial_name": trial.name,
            "outcome": "pass" if reward["reward"] == 1 else "fail",
            "outcome_classification": (
                failure["classification"] if failure else "verifier_scored_outcome"
            ),
            "solver_model": config["agent"]["model_name"],
            "solver_reasoning_effort": config["agent"]["kwargs"]["reasoning_effort"],
            "audit_model": config["verifier"]["kwargs"]["audit_model"],
            "audit_reasoning_effort": "high",
            "framework": config["environment"]["kwargs"]["framework"],
            "docker_image": config["environment"]["kwargs"]["docker_image"],
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "agent_wall_time_sec": seconds_between(
                agent_interval.get("started_at"), agent_interval.get("finished_at")
            ),
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_cache_tokens": agent_result.get("n_cache_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
            "solution_sha256": sha256(solution),
            "reward": reward["reward"],
            "functional_score": reward["functional_score"],
            "runtime_score": reward["runtime_score"],
            "static_policy_score": reward["static_policy_score"],
            "llm_audit_score": reward["llm_audit_score"],
            "runtime_sec": reward["runtime_sec"],
            "exception_type": exception.get("exception_type"),
            "failure_provenance": failure,
        }
        dump(challenge_out / "stamp-info.json", stamp)
        if failure:
            dump(challenge_out / "model-failure.json", failure)
        task_rows.append(stamp)

    manifest = load(CAMPAIGN / "task-copy-manifest.json")
    copy_if_exists(CAMPAIGN / "task-copy-manifest.json", OUT / "task-copy-manifest.json")
    measured = [row["runtime_sec"] for row in task_rows if row["runtime_sec"] >= 0]
    summary = {
        "schema_version": 1,
        "protocol": {
            "base_commit": manifest["base_commit"],
            "solver_model": "grok-4.5",
            "solver_reasoning_effort": "high",
            "audit_model": "gpt-5.6-sol",
            "audit_reasoning_effort": "high",
            "framework": "tensorcircuit",
            "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
            "execution_order": "sequential",
            "valid_outcomes_per_task": 1,
            "resource_adapter": manifest["resource_adapter"],
            "aggregate_task_copy_sha256": manifest["aggregate_task_copy_sha256"],
        },
        "counts": {
            "valid_outcomes": 12,
            "passes": sum(row["reward"] == 1 for row in task_rows),
            "functional_passes": sum(row["functional_score"] == 1 for row in task_rows),
            "static_policy_passes": sum(row["static_policy_score"] == 1 for row in task_rows),
            "audit_passes": sum(row["llm_audit_score"] == 1 for row in task_rows),
        },
        "runtime": {
            "measured_tasks": len(measured),
            "total_sec": round(sum(measured), 2),
            "mean_sec": round(sum(measured) / len(measured), 2),
        },
        "agent_usage": {
            "wall_time_sec": round(sum(row["agent_wall_time_sec"] or 0 for row in task_rows), 3),
            "input_tokens": sum(row["n_input_tokens"] or 0 for row in task_rows),
            "cache_tokens": sum(row["n_cache_tokens"] or 0 for row in task_rows),
            "output_tokens": sum(row["n_output_tokens"] or 0 for row in task_rows),
            "cost_usd": None,
            "cost_note": "The local xAI/Codex integration did not report provider cost.",
        },
        "tasks": task_rows,
        "excluded_attempts": {
            "challenge-01-r1": "pre-repair compatibility-protocol outcome; excluded from normalized final selection",
            "challenge-02-r1": "pre-repair compatibility-protocol outcome; excluded from normalized final selection",
            "challenge-03-r1": "pre-repair compatibility-protocol outcome; excluded from normalized final selection",
            "challenge-04-r1": "pre-repair compatibility-protocol outcome; excluded from normalized final selection",
            "challenge-05-r1": "pre-repair compatibility-protocol outcome; excluded from normalized final selection",
            "challenge-06-r1": "xAI/Codex integer tool-argument compatibility failure",
            "challenge-07-r1": "interrupted during compatibility diagnosis",
            "challenge-07-r2": "interrupted during compatibility diagnosis",
        },
        "notes": {
            "challenge_01": MODEL_FAILURES[1]["evidence"],
            "challenge_04": MODEL_FAILURES[4]["evidence"],
            "challenge_08": MODEL_FAILURES[8]["evidence"],
            "challenge_11": MODEL_FAILURES[11]["evidence"],
            "runtime": "runtime_score is retained for reporting and does not multiply the pass reward.",
        },
    }
    dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
