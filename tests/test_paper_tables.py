from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_tables import make_tables


class PaperTablesTest(unittest.TestCase):
    def test_make_tables_explodes_removal_selection_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_dir = root / "metrics"
            metrics_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "table_name": "aggregate_metrics",
                        "step_name": "insuff_sbs96_manifest_smoke",
                        "benchmark_name": "catalog_insufficiency",
                        "mutation_type": "SBS96",
                        "expert_name": "rule_fusion",
                        "burden": 200,
                        "removal_selection_groups": "flat_signature,high_prevalence_active",
                        "catalog_insufficiency_auroc": 0.75,
                        "catalog_insufficiency_auprc": 0.8,
                        "catalog_insufficiency_probability_ece": 0.1,
                        "catalog_insufficiency_probability_brier": 0.2,
                        "sample_f1": 0.6,
                        "exposure_tvd": 0.3,
                        "reconstruction_cosine": 0.9,
                    }
                ]
            ).to_csv(metrics_dir / "aggregate_metrics.tsv", sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "table_name": "fusion_evidence_features",
                        "step_name": "insuff_sbs96_manifest_smoke",
                        "benchmark_name": "catalog_insufficiency",
                        "mutation_type": "SBS96",
                        "burden_group": "low",
                        "removal_selection_groups": "flat_signature,high_prevalence_active",
                        "catalog_insufficiency_proxy_score": 0.5,
                        "catalog_insufficiency_probability": 0.55,
                        "residual_structure_score": 0.2,
                        "agreement_score": 0.7,
                        "disagreement_score": 0.3,
                        "mean_reconstruction_cosine": 0.9,
                        "catalog_feature_missing_catalog_probability_mass": 0.1,
                        "catalog_feature_classifier_entropy": 0.4,
                    }
                ]
            ).to_csv(metrics_dir / "fusion_evidence_features.tsv", sep="\t", index=False)

            make_tables(root, root / "tables")
            insuff_summary = pd.read_csv(root / "tables" / "catalog_insufficiency_by_group.tsv", sep="\t")
            evidence_summary = pd.read_csv(root / "tables" / "fusion_evidence_by_group.tsv", sep="\t")

        self.assertEqual(
            sorted(insuff_summary["removal_selection_groups"].tolist()),
            ["flat_signature", "high_prevalence_active"],
        )
        self.assertEqual(
            sorted(evidence_summary["removal_selection_groups"].tolist()),
            ["flat_signature", "high_prevalence_active"],
        )


if __name__ == "__main__":
    unittest.main()
