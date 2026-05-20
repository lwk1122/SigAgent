from __future__ import annotations

import unittest

from signature_decision.catalog_insufficiency import catalog_insufficiency_score_from_sample_result
from signature_decision.experts.schema import ExpertSampleResult


class CatalogInsufficiencyScoreTest(unittest.TestCase):
    def test_uses_fused_proxy_score_when_available(self) -> None:
        sample_result = ExpertSampleResult(
            sample_id="sample_a",
            active_signatures=["S1"],
            exposures={"S1": 10.0},
            residual_counts=[0.0, 0.0],
            metrics={"mutation_count": 10.0, "reconstruction_cosine": 0.99},
            diagnostics={"catalog_insufficiency_proxy_score": 0.73},
        )

        self.assertAlmostEqual(catalog_insufficiency_score_from_sample_result(sample_result), 0.73)


if __name__ == "__main__":
    unittest.main()
