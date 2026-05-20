from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_ablation_tables import make_ablation_tables


class AblationTablesTest(unittest.TestCase):
    def test_make_ablation_tables_computes_directional_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_dir = root / "metrics"
            metrics_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "step_name": "known_sbs96_ablation_all_experts",
                        "benchmark_name": "known_catalog",
                        "mutation_type": "SBS96",
                        "expert_name": "rule_fusion",
                        "burden": 200,
                        "sample_f1": 0.8,
                        "exposure_tvd": 0.2,
                    },
                    {
                        "step_name": "known_sbs96_ablation_no_musical",
                        "benchmark_name": "known_catalog",
                        "mutation_type": "SBS96",
                        "expert_name": "rule_fusion",
                        "burden": 200,
                        "sample_f1": 0.7,
                        "exposure_tvd": 0.3,
                    },
                ]
            ).to_csv(metrics_dir / "aggregate_metrics.tsv", sep="\t", index=False)

            deltas, summary = make_ablation_tables(
                root,
                root / "tables",
                baseline_step="known_sbs96_ablation_all_experts",
                metrics=["sample_f1", "exposure_tvd"],
            )

        sample_delta = deltas.loc[deltas["metric_name"] == "sample_f1"].iloc[0]
        tvd_delta = deltas.loc[deltas["metric_name"] == "exposure_tvd"].iloc[0]
        self.assertAlmostEqual(float(sample_delta["delta_vs_baseline"]), -0.1)
        self.assertAlmostEqual(float(sample_delta["improvement_vs_baseline"]), -0.1)
        self.assertAlmostEqual(float(tvd_delta["delta_vs_baseline"]), 0.1)
        self.assertAlmostEqual(float(tvd_delta["improvement_vs_baseline"]), -0.1)
        self.assertEqual(summary.loc[0, "baseline_label"], "all_experts")


if __name__ == "__main__":
    unittest.main()
