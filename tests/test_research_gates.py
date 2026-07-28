from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH_ROOT))

from check_gates import evaluate_gates  # noqa: E402


SHA = "a" * 64
SELECTED_TASK = "05"


def _write_ready_fixture(root: Path) -> Path:
    (root / "research").mkdir(parents=True)
    (root / "research" / "SURVEY.md").write_text(
        f"# Survey\n\n**Status: READY**\n\n## Task {SELECTED_TASK}: complete\n",
        encoding="utf-8",
    )

    cases = [
        {
            "task_id": SELECTED_TASK,
            "case_id": f"public-{SELECTED_TASK}",
            "sha256": SHA,
            "provenance": "generated from the canonical public task",
        }
    ]
    manifest = root / "datasets" / "public" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "status": "ready",
                "version": "orbitq-workloads-v20260727.1",
                "selected_task_id": SELECTED_TASK,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for repeat in range(1, 7):
        rows.append(
            {
                "task_id": SELECTED_TASK,
                "solution": "reference",
                "source_sha256": SHA,
                "passed": True,
                "timed_out": False,
                "terminal_status": "SUCCESS",
                "runtime_sec": float(repeat),
                "engine": "docker",
                "environment": "tensorcircuit-py311",
                "environment_image_provenance": {"id": f"sha256:{SHA}"},
                "timeout_sec": 300,
                "repeat": repeat,
                "planned_repeats": 6,
                "evaluator_sha256": SHA,
                "compatibility_sha256": SHA,
                "staging_snapshot_sha256": SHA,
                "shared_container_id": f"container-id-{SELECTED_TASK}",
                "shared_container_name": f"container-name-{SELECTED_TASK}",
                "command": [
                    "docker",
                    "exec",
                    f"container-name-{SELECTED_TASK}",
                    "python",
                    f"/session/evaluator/evaluate_{int(SELECTED_TASK)}.py",
                ],
                "shared_container_start_command": [
                    "docker",
                    "run",
                    "--cpus",
                    "8",
                    "--memory",
                    "9g",
                    "--network",
                    "none",
                    "--mount",
                    (
                        f"type=bind,src=/tmp/{SELECTED_TASK},"
                        "dst=/session,readonly"
                    ),
                ],
            }
        )
    baseline = root / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "host": {"fingerprint_sha256": SHA},
                "results": rows,
            }
        ),
        encoding="utf-8",
    )
    return baseline


class ResearchGateTests(unittest.TestCase):
    def test_ready_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = _write_ready_fixture(root)
            report = evaluate_gates(
                root,
                task_id=SELECTED_TASK,
                baseline_report=baseline,
            )
            self.assertTrue(report["ready"], report)
            self.assertTrue(report["research_ready"], report)
            self.assertTrue(report["promotion_ready"], report)

    def test_missing_artifacts_fail_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate_gates(
                Path(temp),
                task_id=SELECTED_TASK,
                baseline_report=None,
            )
            self.assertFalse(report["ready"])
            self.assertTrue(
                all(not check["ready"] for check in report["checks"].values())
            )

    def test_other_task_does_not_satisfy_public_dataset_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = _write_ready_fixture(root)
            manifest = root / "datasets" / "public" / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["cases"][0]["task_id"] = "04"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                task_id=SELECTED_TASK,
                baseline_report=baseline,
            )
            dataset = report["checks"]["public_dataset"]
            self.assertFalse(dataset["ready"])
            self.assertIn(SELECTED_TASK, " ".join(dataset["errors"]))

    def test_failing_reference_row_closes_baseline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = _write_ready_fixture(root)
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["results"][0]["passed"] = False
            payload["results"][0]["terminal_status"] = "FUNCTIONAL_FAILED"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                task_id=SELECTED_TASK,
                baseline_report=baseline,
            )
            self.assertFalse(report["checks"]["reference_baselines"]["ready"])
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])

    def test_multiple_containers_for_one_task_close_baseline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = _write_ready_fixture(root)
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["results"][1]["shared_container_id"] = "different-container"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                task_id=SELECTED_TASK,
                baseline_report=baseline,
            )
            baseline_check = report["checks"]["reference_baselines"]
            self.assertFalse(baseline_check["ready"])
            self.assertIn("one container ID", " ".join(baseline_check["errors"]))
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])

    def test_baseline_is_not_required_to_begin_research(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_ready_fixture(root)
            report = evaluate_gates(
                root,
                task_id=SELECTED_TASK,
                baseline_report=None,
            )
            self.assertTrue(report["ready"])
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])
            self.assertFalse(report["checks"]["reference_baselines"]["ready"])


if __name__ == "__main__":
    unittest.main()
