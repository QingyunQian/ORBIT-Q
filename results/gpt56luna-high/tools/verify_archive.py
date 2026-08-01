"""Verify Luna/high protocol stamps, scores, hashes, and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("[REDACTED]", "null"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = load(ROOT / "summary.json")
    rows = summary["tasks"]
    assert len(rows) == 12
    assert summary["counts"] == {
        "valid_outcomes": 12,
        "passes": 10,
        "functional_passes": 12,
        "static_policy_passes": 12,
        "audit_passes": 10,
    }
    assert [row["challenge"] for row in rows if row["reward"] != 1] == [1, 4]

    for row in rows:
        task = row["challenge"]
        folder = ROOT / f"challenge-{task:02d}"
        reward = load(folder / "reward.json")
        stamp = load(folder / "stamp-info.json")
        config = load(folder / "config.json")
        assert reward["reward"] == row["reward"] == stamp["reward"]
        assert config["agent"]["model_name"] == "gpt-5.6-luna"
        assert config["agent"]["kwargs"]["reasoning_effort"] == "high"
        assert config["verifier"]["kwargs"]["audit_model"] == "gpt-5.6-sol"
        assert config["verifier"]["kwargs"]["profile_config_path"].endswith(
            "audit-high.config.toml"
        )
        solution = folder / f"solution_{task}.py"
        assert solution.exists()
        assert sha256(solution) == row["solution_sha256"]

    assert rows[10]["exception_type"] == "AgentTimeoutError"
    assert rows[10]["reward"] == 1
    assert (ROOT / "audit-high.config.toml").read_text().strip() == (
        'model_reasoning_effort = "high"'
    )
    manifest = load(ROOT / "task-copy-manifest.json")
    assert manifest["base_commit"] == "0201238ec2983907e2891f5319f5fff2d00844d5"
    assert manifest["aggregate_task_copy_sha256"] == (
        "19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895"
    )

    for stem in ("gpt56luna-high-outcomes", "gpt56luna-high-agent-resource-use"):
        png = ROOT / "figs" / f"{stem}.png"
        assert png.stat().st_size > 50_000
        with Image.open(png) as image:
            assert image.width >= 2500 and image.height >= 800

    print(
        "Archive verification passed: 12 outcomes, Luna/high solver, "
        "Sol/high audit, fixed task hashes, and figures valid."
    )


if __name__ == "__main__":
    main()
