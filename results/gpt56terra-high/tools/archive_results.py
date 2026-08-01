"""Archive the selected valid GPT-5.6 Terra/high campaign outcomes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "gpt56terra-high"
CAMPAIGN = ROOT / "jobs" / "gpt56terra-high-solaudit-20260731-valid"
ATTEMPTS = {task: ("r1" if task == 12 else "r2") for task in range(1, 13)}


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Harbor's sanitizer can emit bare [REDACTED] tokens in otherwise JSON
        # metadata. Treat those unavailable values as null for summary creation.
        return json.loads(text.replace("[REDACTED]", "null"))


def dump(path: Path, value: object) -> None:
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
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()


def selected_trial(task: int) -> tuple[Path, Path]:
    attempt = ATTEMPTS[task]
    job = CAMPAIGN / "jobs" / f"challenge-{task:02d}-tensorcircuit-gpt-5.6-terra-high-20260731-{attempt}"
    trials = sorted(job.glob(f"challenge-{task:02d}__*"))
    if len(trials) != 1:
        raise RuntimeError(f"Expected one trial for task {task:02d}, found {trials}")
    return job, trials[0]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    task_rows = []
    for task in range(1, 13):
        job, trial = selected_trial(task)
        challenge_out = OUT / f"challenge-{task:02d}"
        challenge_out.mkdir(parents=True, exist_ok=True)

        solution = trial / "artifacts" / "root" / f"solution_{task}.py"
        copy_if_exists(solution, challenge_out / f"solution_{task}.py")
        copy_if_exists(trial / "verifier" / "functional-stdout.txt", challenge_out / "functional-stdout-official.txt")
        copy_if_exists(trial / "result.json", challenge_out / "job-result.json")
        copy_if_exists(trial / "config.json", challenge_out / "config.json")
        copy_if_exists(trial / "lock.json", challenge_out / "lock.json")
        copy_if_exists(trial / "agent" / "codex.txt", challenge_out / "agent-codex.log")
        copy_if_exists(trial / "trial.log", challenge_out / "trial.log")
        copy_if_exists(trial / "artifacts" / "manifest.json", challenge_out / "artifact-manifest.json")

        result = load(trial / "result.json")
        original_reward = load(trial / "verifier" / "reward.json")
        reward = original_reward
        audit_source = trial / "verifier" / "audit-details.json"

        if task == 6:
            re_job = CAMPAIGN / "jobs" / "challenge-06-tensorcircuit-gpt-5.6-terra-high-20260731-r2-reaudit2"
            re_trials = sorted(re_job.glob("challenge-06-r2-reaudit2-task__*"))
            if len(re_trials) != 1:
                raise RuntimeError(f"Expected one Task 06 re-audit trial, found {re_trials}")
            re_trial = re_trials[0]
            re_reward = load(re_trial / "verifier" / "reward.json")
            re_solution = re_trial / "artifacts" / "root" / "solution_6.py"
            if sha256(solution) != sha256(re_solution):
                raise RuntimeError("Task 06 re-audit candidate is not byte-identical to r2")
            reward = dict(re_reward)
            reward["runtime_sec"] = original_reward["runtime_sec"]
            reward["provenance"] = {
                "combined_result": True,
                "solver_and_functional_source": "challenge-06 r2",
                "audit_and_static_source": "challenge-06 r2 re-audit2",
                "reason": "The original audit ended during a host network outage after the candidate had passed the functional evaluator.",
                "original_runtime_sec": original_reward["runtime_sec"],
                "reaudit_runtime_sec": re_reward["runtime_sec"],
                "candidate_sha256": sha256(solution),
            }
            dump(challenge_out / "reward-original-network-interrupted.json", original_reward)
            dump(challenge_out / "reward-reaudit2.json", re_reward)
            copy_if_exists(re_trial / "verifier" / "audit-details.json", challenge_out / "audit-details-reaudit2.json")
            copy_if_exists(re_trial / "verifier" / "functional-stdout.txt", challenge_out / "functional-stdout-reaudit2.txt")
            copy_if_exists(re_trial / "result.json", challenge_out / "job-result-reaudit2.json")
            copy_if_exists(re_trial / "agent" / "codex.txt", challenge_out / "reaudit2-codex.log")
            copy_if_exists(re_solution, challenge_out / "solution_6-reaudit2.py")
            audit_source = re_trial / "verifier" / "audit-details.json"

        dump(challenge_out / "reward.json", reward)
        copy_if_exists(audit_source, challenge_out / "audit-details.json")

        config = result["config"]
        agent_result = result.get("agent_result") or {}
        agent_interval = result.get("agent_execution") or {}
        stamp = {
            "challenge": task,
            "selected_attempt": ATTEMPTS[task],
            "job_name": job.name,
            "trial_name": trial.name,
            "solver_model": config["agent"]["model_name"],
            "solver_reasoning_effort": config["agent"]["kwargs"]["reasoning_effort"],
            "audit_model": config["verifier"]["kwargs"]["audit_model"],
            "audit_reasoning_effort": "high",
            "framework": config["environment"]["kwargs"]["framework"],
            "docker_image": config["environment"]["kwargs"]["docker_image"],
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "agent_wall_time_sec": seconds_between(agent_interval.get("started_at"), agent_interval.get("finished_at")),
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_cache_tokens": agent_result.get("n_cache_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
            "solution_sha256": sha256(solution),
            "reward": reward["reward"],
            "functional_score": reward["functional_score"],
            "static_policy_score": reward["static_policy_score"],
            "llm_audit_score": reward["llm_audit_score"],
            "runtime_sec": reward["runtime_sec"],
            "exception_type": (result.get("exception_info") or {}).get("exception_type"),
        }
        dump(challenge_out / "stamp-info.json", stamp)
        task_rows.append(stamp)

    manifest = load(CAMPAIGN / "task-copy-manifest.json")
    copy_if_exists(CAMPAIGN / "task-copy-manifest.json", OUT / "task-copy-manifest.json")

    passes = sum(row["reward"] == 1 for row in task_rows)
    functional = sum(row["functional_score"] == 1 for row in task_rows)
    static = sum(row["static_policy_score"] == 1 for row in task_rows)
    audit = sum(row["llm_audit_score"] == 1 for row in task_rows)
    measured = [row["runtime_sec"] for row in task_rows if row["runtime_sec"] >= 0]
    summary = {
        "schema_version": 1,
        "protocol": {
            "base_commit": manifest["base_commit"],
            "solver_model": "gpt-5.6-terra",
            "solver_reasoning_effort": "high",
            "audit_model": "gpt-5.6-sol",
            "audit_reasoning_effort": "high",
            "framework": "tensorcircuit",
            "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
            "execution_order": "sequential",
            "trials_per_valid_outcome": 1,
            "resource_adapter": manifest["resource_adapter"],
            "aggregate_task_copy_sha256": manifest["aggregate_task_copy_sha256"],
        },
        "counts": {
            "valid_outcomes": 12,
            "passes": passes,
            "functional_passes": functional,
            "static_policy_passes": static,
            "audit_passes": audit,
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
            "cost_usd": round(sum(row["cost_usd"] or 0 for row in task_rows), 6),
        },
        "tasks": task_rows,
        "excluded_attempts": {
            "challenge-01..11-r1": "terminal solver TLS/network failures",
            "challenge-05-r3": "unnecessary duplicate stopped after r2 had already passed",
            "challenge-06-r3,r4": "terminal solver TLS/network failures",
            "challenge-06-r2-reaudit1": "aborted malformed verifier-only task copy",
            "earlier_private_tmp_campaigns": "excluded because the container bind mount did not expose the intended host task tree",
        },
        "task06_reaudit": {
            "candidate_sha256": task_rows[5]["solution_sha256"],
            "original_runtime_sec": task_rows[5]["runtime_sec"],
            "reaudit_runtime_sec": 86.49,
            "final_reward": task_rows[5]["reward"],
        },
    }
    dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
