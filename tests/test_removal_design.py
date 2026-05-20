from __future__ import annotations

import unittest

import pandas as pd

from signature_decision.removal_design import (
    RemovalDesignConfig,
    build_catalog_removal_design,
    signature_property_frame,
)


class RemovalDesignTest(unittest.TestCase):
    def test_signature_properties_include_prevalence_flatness_and_similarity(self) -> None:
        signatures = pd.DataFrame(
            {
                "S1": [0.5, 0.5, 0.0],
                "S2": [0.49, 0.51, 0.0],
                "S3": [0.0, 0.0, 1.0],
            }
        )
        exposures = pd.DataFrame(
            {
                "sample_a": [10.0, 0.0, 0.0],
                "sample_b": [0.0, 3.0, 0.0],
                "sample_c": [0.0, 0.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

        properties = signature_property_frame(signatures, exposures)
        row = properties.set_index("signature_name").loc["S1"]

        self.assertEqual(int(row["prevalence_count"]), 1)
        self.assertGreater(float(row["flatness_score"]), 0.0)
        self.assertEqual(row["nearest_signature"], "S2")
        self.assertGreater(float(row["max_cosine_to_other"]), 0.99)

    def test_removal_design_marks_unbenchmarkable_controls(self) -> None:
        signatures = pd.DataFrame(
            {
                "S1": [0.5, 0.5, 0.0],
                "S2": [0.49, 0.51, 0.0],
                "S3": [0.0, 0.0, 1.0],
            }
        )
        exposures = pd.DataFrame(
            {
                "sample_a": [10.0, 0.0, 0.0],
                "sample_b": [0.0, 3.0, 0.0],
                "sample_c": [0.0, 0.0, 0.0],
            },
            index=["S1", "S2", "S3"],
        )

        design = build_catalog_removal_design(
            signatures,
            exposures,
            mutation_type="SBS96",
            config=RemovalDesignConfig(n_per_group=1),
        )

        self.assertIn("high_similarity", set(design["selection_group"]))
        self.assertIn("inactive_or_unbenchmarkable_control", set(design["selection_group"]))
        control = design.loc[design["selection_group"] == "inactive_or_unbenchmarkable_control"].iloc[0]
        self.assertFalse(bool(control["benchmarkable_with_active_labels"]))


if __name__ == "__main__":
    unittest.main()

