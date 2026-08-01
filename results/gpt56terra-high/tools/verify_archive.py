"""Verify protocol stamps, scores, hashes, and figure outputs."""

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
        "passes": 9,
        "functional_passes": 9,
        "static_policy_passes": 11,
        "audit_passes": 9,
    }

    for row in rows:
        task = row["challenge"]
        folder = ROOT / f"challenge-{task:02d}"
        reward = load(folder / "reward.json")
        stamp = load(folder / "stamp-info.json")
        config = load(folder / "config.json")
        assert reward["reward"] == row["reward"] == stamp["reward"]
        assert config["agent"]["model_name"] == "gpt-5.6-terra"
        assert config["agent"]["kwargs"]["reasoning_effort"] == "high"
        assert config["verifier"]["kwargs"]["audit_model"] == "gpt-5.6-sol"
        assert config["verifier"]["kwargs"]["profile_config_path"].endswith("audit-high.config.toml")
        solution = folder / f"solution_{task}.py"
        assert (sha256(solution) if solution.exists() else None) == row["solution_sha256"]

    task6 = ROOT / "challenge-06"
    assert sha256(task6 / "solution_6.py") == sha256(task6 / "solution_6-reaudit2.py")
    assert load(task6 / "reward.json")["runtime_sec"] == 116.18
    assert load(task6 / "reward-reaudit2.json")["runtime_sec"] == 86.49
    assert (ROOT / "audit-high.config.toml").read_text().strip() == 'model_reasoning_effort = "high"'

    for stem in ("gpt56terra-high-outcomes", "gpt56terra-high-agent-resource-use"):
        png = ROOT / "figs" / f"{stem}.png"
        assert png.stat().st_size > 50_000
        with Image.open(png) as image:
            assert image.width >= 2500 and image.height >= 800

    print("Archive verification passed: 12 outcomes, fixed model/effort protocol, Task 06 hash match, figures valid.")


if __name__ == "__main__":
    main()
