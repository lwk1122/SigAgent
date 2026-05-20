from __future__ import annotations

import unittest

import pandas as pd

from signature_decision.experts.schema import ExpertRequest, ExpertSampleResult
from signature_decision.fusion_features import extract_fusion_evidence


class FusionEvidenceTest(unittest.TestCase):
    def test_extracts_agreement_and_catalog_features(self) -> None:
        sample_matrix = pd.DataFrame({"sample_a": [8.0, 2.0]}, index=["A", "B"])
        signature_matrix = pd.DataFrame(
            {
                "S1": [0.8, 0.2],
                "S2": [0.2, 0.8],
            },
            index=["A", "B"],
        )
        request = ExpertRequest(
            mutation_type="SBS96",
            sample_matrix=sample_matrix,
            signature_matrix=signature_matrix,
        )
        sample_results = {
            "expert_one": ExpertSampleResult(
                sample_id="sample_a",
                active_signatures=["S1"],
                exposures={"S1": 10.0},
                residual_counts=[0.0, 0.0],
                metrics={"mutation_count": 10.0, "reconstruction_cosine": 0.99},
            ),
            "expert_two": ExpertSampleResult(
                sample_id="sample_a",
                active_signatures=["S1", "S2"],
                exposures={"S1": 8.0, "S2": 2.0},
                residual_counts=[1.0, 0.0],
                metrics={"mutation_count": 10.0, "reconstruction_cosine": 0.95},
            ),
        }

        evidence = extract_fusion_evidence(
            sample_id="sample_a",
            request=request,
            sample_results_by_expert=sample_results,
            failed_expert_names=["failed_expert"],
        )

        self.assertEqual(evidence.sample_id, "sample_a")
        self.assertEqual(evidence.expert_names, ["expert_one", "expert_two"])
        self.assertEqual(evidence.failed_expert_names, ["failed_expert"])
        self.assertAlmostEqual(evidence.agreement_score, 0.5)
        self.assertAlmostEqual(evidence.disagreement_score, 0.5)
        self.assertAlmostEqual(evidence.mean_reconstruction_cosine, 0.97)
        self.assertIn("burden_group", evidence.group_context)
        feature_row = evidence.to_feature_row()
        self.assertEqual(feature_row["expert_count"], 2)
        self.assertEqual(feature_row["failed_expert_count"], 1)
        self.assertIn("catalog_feature_disagreement_score", feature_row)


if __name__ == "__main__":
    unittest.main()

