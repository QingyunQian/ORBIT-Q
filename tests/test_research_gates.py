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
CAMPAIGN_TASK_ID = "05"


def _write_ready_fixture(root: Path, external: Path) -> tuple[Path, Path]:
    (root / "research").mkdir(parents=True)
    sections = f"## Task {CAMPAIGN_TASK_ID}: complete"
    (root / "research" / "SURVEY.md").write_text(
        f"# Survey\n\n**Status: READY**\n\n{sections}\n",
        encoding="utf-8",
    )

    public_version = "orbitq-workloads-v20260727.1"
    cases = [
        {
            "task_id": CAMPAIGN_TASK_ID,
            "case_id": f"public-{CAMPAIGN_TASK_ID}",
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
                "version": public_version,
                "campaign_task_id": CAMPAIGN_TASK_ID,
                "required_task_ids": [CAMPAIGN_TASK_ID],
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for task in (CAMPAIGN_TASK_ID,):
        for repeat in range(1, 7):
            rows.append(
                {
                    "task_id": task,
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
                    "shared_container_id": f"container-id-{task}",
                    "shared_container_name": f"container-name-{task}",
                    "command": [
                        "docker",
                        "exec",
                        f"container-name-{task}",
                        "python",
                        f"/session/evaluator/evaluate_{int(task)}.py",
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
                            f"type=bind,src=/tmp/{task},"
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

    attestation = external / "controller-attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "public_dataset_version": public_version,
                "hidden_tuning_ready": True,
                "sealed_holdout_ready": True,
                "controller_protocol_version": "v1",
                "attested_at_utc": "2026-07-27T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return baseline, attestation


class ResearchGateTests(unittest.TestCase):
    def test_ready_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            baseline, attestation = _write_ready_fixture(root, Path(ext))
            report = evaluate_gates(
                root,
                baseline_report=baseline,
                controller_attestation=attestation,
            )
            self.assertTrue(report["ready"], report)
            self.assertTrue(report["research_ready"], report)
            self.assertTrue(report["promotion_ready"], report)

    def test_missing_artifacts_fail_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate_gates(
                Path(temp),
                baseline_report=None,
                controller_attestation=None,
            )
            self.assertFalse(report["ready"])
            self.assertTrue(
                all(not check["ready"] for check in report["checks"].values())
            )

    def test_controller_attestation_inside_checkout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            baseline, external_attestation = _write_ready_fixture(root, Path(ext))
            internal = root / "controller-attestation.json"
            internal.write_bytes(external_attestation.read_bytes())
            report = evaluate_gates(
                root,
                baseline_report=baseline,
                controller_attestation=internal,
            )
            controller = report["checks"]["trusted_controller"]
            self.assertFalse(controller["ready"])
            self.assertIn("outside", " ".join(controller["errors"]))

    def test_controller_attestation_rejects_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            baseline, attestation = _write_ready_fixture(root, Path(ext))
            payload = json.loads(attestation.read_text(encoding="utf-8"))
            payload["hidden_path"] = "/private/holdout"
            attestation.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                baseline_report=baseline,
                controller_attestation=attestation,
            )
            controller = report["checks"]["trusted_controller"]
            self.assertFalse(controller["ready"])
            self.assertIn("forbidden fields", " ".join(controller["errors"]))

    def test_failing_reference_row_closes_baseline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            baseline, attestation = _write_ready_fixture(root, Path(ext))
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["results"][0]["passed"] = False
            payload["results"][0]["terminal_status"] = "FUNCTIONAL_FAILED"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                baseline_report=baseline,
                controller_attestation=attestation,
            )
            self.assertFalse(report["checks"]["reference_baselines"]["ready"])
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])

    def test_multiple_containers_for_one_task_close_baseline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            baseline, attestation = _write_ready_fixture(root, Path(ext))
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["results"][1]["shared_container_id"] = "different-container"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_gates(
                root,
                baseline_report=baseline,
                controller_attestation=attestation,
            )
            baseline_check = report["checks"]["reference_baselines"]
            self.assertFalse(baseline_check["ready"])
            self.assertIn("one container ID", " ".join(baseline_check["errors"]))
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])

    def test_baseline_is_not_required_to_begin_research(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as ext:
            root = Path(temp)
            _, attestation = _write_ready_fixture(root, Path(ext))
            report = evaluate_gates(
                root,
                baseline_report=None,
                controller_attestation=attestation,
            )
            self.assertTrue(report["ready"])
            self.assertTrue(report["research_ready"])
            self.assertFalse(report["promotion_ready"])
            self.assertFalse(report["checks"]["reference_baselines"]["ready"])


if __name__ == "__main__":
    unittest.main()
