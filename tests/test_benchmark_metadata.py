from __future__ import annotations

import unittest

import pandas as pd

from signature_decision.benchmark import _evaluate_runs_for_slice
from signature_decision.experts.schema import ExpertRequest, ExpertRunResult, ExpertSampleResult


class BenchmarkMetadataTest(unittest.TestCase):
    def test_evaluated_slice_tables_are_self_describing(self) -> None:
        sample_matrix = pd.DataFrame({"sample_a": [10.0, 0.0]}, index=["A", "B"])
        signature_matrix = pd.DataFrame({"S1": [1.0, 0.0]}, index=["A", "B"])
        request = ExpertRequest(
            mutation_type="SBS96",
            sample_matrix=sample_matrix,
            signature_matrix=signature_matrix,
            request_id="test_request",
        )
        run = ExpertRunResult(
            expert_name="plain_nnls",
            mutation_type="SBS96",
            request_id="test_request",
            status="success",
            signature_names=["S1"],
            channel_ids=["A", "B"],
            sample_results=[
                ExpertSampleResult(
                    sample_id="sample_a",
                    active_signatures=["S1"],
                    exposures={"S1": 10.0},
                )
            ],
        )

        result = _evaluate_runs_for_slice(
            runs=[run],
            request=request,
            truth_exposures=pd.DataFrame({"sample_a": [10.0]}, index=["S1"]),
            benchmark_name="known_catalog",
            slice_parameters={"mutation_type": "SBS96", "burden": 200},
        )

        self.assertEqual(result.aggregate_metrics.loc[0, "benchmark_name"], "known_catalog")
        self.assertEqual(result.per_sample_metrics.loc[0, "benchmark_name"], "known_catalog")
        self.assertEqual(result.aggregate_metrics.loc[0, "mutation_type"], "SBS96")


if __name__ == "__main__":
    unittest.main()
