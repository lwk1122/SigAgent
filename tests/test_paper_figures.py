from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_paper_figures import make_paper_figures


class PaperFiguresTest(unittest.TestCase):
    def test_make_paper_figures_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "paper"
            (root / "paper_known_catalog_smoke" / "metrics").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "expert_name": "plain_nnls",
                        "burden": 100,
                        "sample_f1": 0.2,
                        "exposure_tvd": 0.5,
                        "reconstruction_cosine": 0.9,
                    },
                    {
                        "expert_name": "rule_fusion",
                        "burden": 100,
                        "sample_f1": 0.3,
                        "exposure_tvd": 0.4,
                        "reconstruction_cosine": 0.92,
                    },
                ]
            ).to_csv(root / "paper_known_catalog_smoke" / "metrics" / "aggregate_metrics.tsv", sep="\t", index=False)

            (root / "paper_catalog_insufficiency_manifest_smoke" / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "expert_name": "rule_fusion",
                        "mutation_type": "SBS96",
                        "burden": 100,
                        "removal_selection_groups": "flat_signature",
                        "catalog_insufficiency_auroc": 0.8,
                        "catalog_insufficiency_auprc": 0.9,
                    }
                ]
            ).to_csv(
                root / "paper_catalog_insufficiency_manifest_smoke" / "tables" / "catalog_insufficiency_by_group.tsv",
                sep="\t",
                index=False,
            )

            (root / "paper_calibration_smoke" / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "group_dimension": "overall",
                        "group_value": "all",
                        "ece": 0.1,
                        "brier": 0.2,
                        "mean_predicted_probability": 0.5,
                        "observed_positive_fraction": 0.6,
                        "n_samples": 10,
                    }
                ]
            ).to_csv(root / "paper_calibration_smoke" / "tables" / "reliability_summary.tsv", sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "mean_predicted_probability": 0.5,
                        "observed_positive_fraction": 0.6,
                        "n_samples": 10,
                    }
                ]
            ).to_csv(root / "paper_calibration_smoke" / "tables" / "reliability_bins.tsv", sep="\t", index=False)

            (root / "paper_discovery_smoke" / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "n_extracted_components": 1,
                        "mean_delta_reconstruction_cosine_vs_current": 0.1,
                        "mean_delta_reconstruction_cosine_vs_known_only": 0.12,
                        "mean_delta_relative_l1_pct_vs_current": 10.0,
                        "mean_delta_relative_l1_pct_vs_known_only": 12.0,
                    }
                ]
            ).to_csv(root / "paper_discovery_smoke" / "tables" / "discovery_packet_summary.tsv", sep="\t", index=False)

            (root / "paper_real_data_stress_smoke" / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "stress_design": "data_driven_active_signature_removal",
                        "stress_step_name": "fixed",
                        "mean_catalog_insufficiency_probability_delta": 0.02,
                        "max_catalog_insufficiency_probability_delta": 0.1,
                        "n_catalog_insufficiency_delta_ge_0_10": 1,
                        "n_primary_recommendation_changed": 0,
                        "n_samples": 1,
                    }
                ]
            ).to_csv(
                root / "paper_real_data_stress_smoke" / "tables" / "real_data_stress_design_summary.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "stress_design": "data_driven_active_signature_removal",
                        "sample_id": "s1",
                        "source_tumor_type": "Synthetic",
                        "primary_recommendation_full": "accept",
                        "primary_recommendation_reduced": "accept",
                        "catalog_insufficiency_probability_delta_reduced_minus_full": 0.1,
                        "residual_structure_score_delta_reduced_minus_full": 0.02,
                    }
                ]
            ).to_csv(
                root / "paper_real_data_stress_smoke" / "tables" / "real_data_catalog_stress_delta.tsv",
                sep="\t",
                index=False,
            )

            manifest = make_paper_figures(root, Path(tmpdir) / "figures")

            self.assertEqual(len(manifest), 6)
            self.assertTrue((Path(tmpdir) / "figures" / "figure_manifest.tsv").exists())
            self.assertTrue((Path(tmpdir) / "figures" / "figure6_real_data_stress.png").exists())


if __name__ == "__main__":
    unittest.main()
