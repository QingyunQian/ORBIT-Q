"""Verify Grok 4.5/high protocol stamps, scores, hashes, failures, and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASE = "0201238ec2983907e2891f5319f5fff2d00844d5"
EXPECTED_TASK_SHA = "19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895"
EXPECTED_ATTEMPTS = {
    1: "r1", 2: "r1", 3: "r1", 4: "r1", 5: "r1", 6: "r2",
    7: "r3", 8: "r1", 9: "r1", 10: "r1", 11: "r1", 12: "r1",
}


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("[REDACTED]", "null"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_strings(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_strings(child)
    elif isinstance(value, str):
        yield value
        if value.startswith(("{", "[")):
            try:
                yield from nested_strings(json.loads(value))
            except json.JSONDecodeError:
                pass


def main() -> None:
    summary = load(ROOT / "summary.json")
    rows = summary["tasks"]
    assert len(rows) == 12
    assert summary["counts"] == {
        "valid_outcomes": 12,
        "passes": 9,
        "functional_passes": 10,
        "static_policy_passes": 9,
        "audit_passes": 9,
    }
    assert [row["challenge"] for row in rows if row["reward"] != 1] == [4, 8, 11]
    assert summary["agent_usage"]["cost_usd"] is None

    for row in rows:
        task = row["challenge"]
        folder = ROOT / f"challenge-{task:02d}"
        reward = load(folder / "reward.json")
        stamp = load(folder / "stamp-info.json")
        config = load(folder / "config.json")
        assert row["selected_attempt"] == EXPECTED_ATTEMPTS[task]
        assert reward["reward"] == row["reward"] == stamp["reward"]
        assert config["agent"]["model_name"] == "grok-4.5"
        assert config["agent"]["kwargs"]["reasoning_effort"] == "high"
        assert config["verifier"]["kwargs"]["audit_model"] == "gpt-5.6-sol"
        assert config["verifier"]["kwargs"]["profile_config_path"].endswith(
            "audit-high.config.toml"
        )
        solution = folder / f"solution_{task}.py"
        if row["solution_sha256"] is None:
            assert not solution.exists()
        else:
            assert solution.exists()
            assert sha256(solution) == row["solution_sha256"]
        assert (folder / "agent-codex.log").stat().st_size > 0
        assert (folder / "xai-responses-proxy.jsonl").exists()

    for task, pattern in ((8, "pkill -f 'python'"), (11, 'pkill -f "jax"')):
        folder = ROOT / f"challenge-{task:02d}"
        failure = load(folder / "model-failure.json")
        assert failure["classification"] == "valid_model_tool_use_failure"
        assert "exit 143" in failure["signal"]
        evidence = (folder / "agent-codex.log").read_text()
        for path in (folder / "agent-sessions").rglob("*.jsonl"):
            for line in path.read_text().splitlines():
                evidence += "\n".join(nested_strings(json.loads(line)))
        assert pattern in evidence

    task4 = load(ROOT / "challenge-04" / "audit-details.json")
    assert task4["static"]["line_count"] == 281
    assert task4["audit"]["raw_numpy_jax_quantum_simulator_bypass"] is True
    assert (ROOT / "audit-high.config.toml").read_text().strip() == (
        'model_reasoning_effort = "high"'
    )
    manifest = load(ROOT / "task-copy-manifest.json")
    assert manifest["base_commit"] == EXPECTED_BASE
    assert manifest["aggregate_task_copy_sha256"] == EXPECTED_TASK_SHA
    assert manifest["resource_adapter"] == {
        "cpus": 6,
        "memory_mb": 10240,
        "storage_mb": 16384,
    }

    for stem in ("grok-4.5-high-outcomes", "grok-4.5-high-agent-resource-use"):
        for suffix in ("png", "svg", "pdf"):
            path = ROOT / "figs" / f"{stem}.{suffix}"
            assert path.stat().st_size > 10_000
        with Image.open(ROOT / "figs" / f"{stem}.png") as image:
            assert image.width >= 2500 and image.height >= 800

    print(
        "Archive verification passed: 12 outcomes, 9 passes, Grok 4.5/high solver, "
        "Sol/high audit, fixed task hashes, self-termination provenance, and figures valid."
    )


if __name__ == "__main__":
    main()
