from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.validate_paper_outputs import validate_paper_outputs


class PaperOutputValidationTest(unittest.TestCase):
    def test_validate_outputs_reports_claim_and_suite_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            suite = root / "suite_a"
            (suite / "tables").mkdir(parents=True)
            (suite / "manifests").mkdir()
            pd.DataFrame([{"metric": 1.0}]).to_csv(suite / "tables" / "table.tsv", sep="\t", index=False)
            (suite / "manifests" / "suite.manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "step_name": "step_a",
                            "command_name": "known-benchmark",
                            "status": "success",
                            "returncode": 0,
                        }
                    ]
                )
            )
            expectations = [
                {
                    "claim_id": "figure_test",
                    "claim_label": "Test figure",
                    "suite": "suite_a",
                    "relative_path": "tables/table.tsv",
                    "artifact_type": "paper_table",
                }
            ]

            outputs = validate_paper_outputs(root, root / "readiness", expectations=expectations)

        self.assertEqual(outputs["claim_status"].loc[0, "status"], "ok")
        self.assertEqual(outputs["suite_status"].loc[0, "status"], "success")
        self.assertEqual(outputs["artifact_inventory"].loc[0, "row_count"], 1)

    def test_validate_outputs_marks_missing_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expectations = [
                {
                    "claim_id": "figure_test",
                    "claim_label": "Test figure",
                    "suite": "suite_a",
                    "relative_path": "missing.tsv",
                    "artifact_type": "paper_table",
                }
            ]

            outputs = validate_paper_outputs(root, root / "readiness", expectations=expectations)

        self.assertEqual(outputs["claim_status"].loc[0, "status"], "incomplete")
        self.assertEqual(outputs["artifact_inventory"].loc[0, "status"], "missing_or_empty")


if __name__ == "__main__":
    unittest.main()
