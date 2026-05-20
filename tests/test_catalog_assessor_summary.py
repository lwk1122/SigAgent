from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.summarize_catalog_assessor import coefficient_frame, summarize_artifacts


class CatalogAssessorSummaryTest(unittest.TestCase):
    def test_writes_coefficients_and_training_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "assessor.json"
            artifact.write_text(
                """{
  "feature_names": ["mean_reconstruction_cosine", "disagreement_score"],
  "scaler_mean": [0.9, 0.1],
  "scaler_scale": [0.05, 0.2],
  "coefficients": [-1.5, 0.75],
  "intercept": 0.0,
  "probability_calibrator": null,
  "group_probability_calibrator": null,
  "metadata": {"mutation_type": "SBS96"}
}"""
            )
            pd.DataFrame(
                [
                    {
                        "mutation_type": "SBS96",
                        "burden": 200,
                        "removed_signature": "SBS1",
                        "split_partition": "train",
                        "burden_group": "200_499",
                        "flatness_group": "flat",
                        "disagreement_group": "low",
                        "label": 1,
                    },
                    {
                        "mutation_type": "SBS96",
                        "burden": 200,
                        "removed_signature": "SBS1",
                        "split_partition": "train",
                        "burden_group": "200_499",
                        "flatness_group": "flat",
                        "disagreement_group": "low",
                        "label": 0,
                    },
                ]
            ).to_csv(root / "assessor.training.tsv", sep="\t", index=False)

            coefficients = coefficient_frame(artifact)
            summarize_artifacts([artifact], root / "tables")

            balance = pd.read_csv(root / "tables" / "catalog_assessor_training_balance.tsv", sep="\t")

        self.assertEqual(coefficients.loc[0, "feature_name"], "mean_reconstruction_cosine")
        self.assertEqual(int(balance.loc[0, "n_samples"]), 2)
        self.assertEqual(int(balance.loc[0, "n_positive"]), 1)


if __name__ == "__main__":
    unittest.main()
