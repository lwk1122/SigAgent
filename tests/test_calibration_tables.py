from __future__ import annotations

import unittest

import pandas as pd

from experiments.make_calibration_tables import reliability_tables_from_per_sample


class CalibrationTablesTest(unittest.TestCase):
    def test_assignment_reliability_uses_active_set_f1_label(self) -> None:
        per_sample = pd.DataFrame(
            [
                {
                    "step_name": "known_sbs96_calibration_smoke",
                    "benchmark_name": "known_catalog",
                    "mutation_type": "SBS96",
                    "expert_name": "rule_fusion",
                    "active_set_f1": 0.9,
                    "assignment_confidence_probability": 0.8,
                    "burden_group": "200_499",
                },
                {
                    "step_name": "known_sbs96_calibration_smoke",
                    "benchmark_name": "known_catalog",
                    "mutation_type": "SBS96",
                    "expert_name": "rule_fusion",
                    "active_set_f1": 0.2,
                    "assignment_confidence_probability": 0.7,
                    "burden_group": "200_499",
                },
            ]
        )

        bins, summary = reliability_tables_from_per_sample(per_sample, n_bins=2)

        overall = summary.loc[summary["group_dimension"] == "overall"].iloc[0]
        self.assertEqual(int(overall["n_samples"]), 2)
        self.assertAlmostEqual(float(overall["observed_positive_fraction"]), 0.5)
        self.assertIn("assignment_confidence", set(bins["task_name"]))

    def test_catalog_reliability_explodes_removal_groups(self) -> None:
        per_sample = pd.DataFrame(
            [
                {
                    "step_name": "insuff_sbs96_manifest_smoke",
                    "benchmark_name": "catalog_insufficiency",
                    "mutation_type_x": "SBS96",
                    "expert_name": "rule_fusion",
                    "catalog_insufficient_label": 1,
                    "catalog_insufficiency_probability": 0.6,
                    "removal_selection_groups": "flat_signature,high_prevalence_active",
                }
            ]
        )

        _, summary = reliability_tables_from_per_sample(per_sample, n_bins=2)

        groups = set(summary["group_value"])
        self.assertIn("flat_signature", groups)
        self.assertIn("high_prevalence_active", groups)


if __name__ == "__main__":
    unittest.main()
