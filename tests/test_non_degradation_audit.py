from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.make_non_degradation_audit import make_non_degradation_audit


class NonDegradationAuditTest(unittest.TestCase):
    def test_audit_passes_primary_and_diagnostic_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "paper"
            for suite, contexts in {
                "paper_review_response_sbs96": ["SBS96"],
                "paper_review_response_dbs78_id83": ["DBS78", "ID83"],
            }.items():
                tables = root / suite / "tables"
                tables.mkdir(parents=True)
                known_rows = []
                insuff_rows = []
                for context in contexts:
                    known_rows.extend(
                        [
                            {
                                "mutation_type": context,
                                "expert_name": "plain_nnls",
                                "sample_f1_mean": 0.60,
                                "exposure_tvd_mean": 0.20,
                                "reconstruction_cosine_mean": 0.990,
                            },
                            {
                                "mutation_type": context,
                                "expert_name": "rule_fusion",
                                "sample_f1_mean": 0.61,
                                "exposure_tvd_mean": 0.19,
                                "reconstruction_cosine_mean": 0.986,
                            },
                        ]
                    )
                    insuff_rows.extend(
                        [
                            {
                                "mutation_type": context,
                                "expert_name": "plain_nnls",
                                "catalog_insufficiency_auroc_mean": 0.80,
                                "catalog_insufficiency_auprc_mean": 0.81,
                            },
                            {
                                "mutation_type": context,
                                "expert_name": "rule_fusion",
                                "catalog_insufficiency_auroc_mean": 0.82,
                                "catalog_insufficiency_auprc_mean": 0.83,
                            },
                        ]
                    )
                pd.DataFrame(known_rows).to_csv(
                    tables / "known_catalog_overall_with_uncertainty.tsv",
                    sep="\t",
                    index=False,
                )
                pd.DataFrame(insuff_rows).to_csv(
                    tables / "catalog_insufficiency_overall_with_uncertainty.tsv",
                    sep="\t",
                    index=False,
                )

            outputs = make_non_degradation_audit(root, root / "paper_non_degradation_audit" / "tables")
            audit = outputs["audit"]
            compact = outputs["compact"]

            self.assertEqual(len(audit), 15)
            self.assertTrue(audit["audit_status"].eq("pass").all())
            self.assertEqual(compact["primary_checks_passed"].sum(), 12)
            self.assertEqual(compact["diagnostic_checks_passed"].sum(), 3)


if __name__ == "__main__":
    unittest.main()
