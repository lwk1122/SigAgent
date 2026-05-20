from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_real_data_stress_tables import make_real_data_stress_tables


class RealDataStressTablesTest(unittest.TestCase):
    def test_make_real_data_stress_tables_summarizes_decision_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_source = root / "samples.csv"
            pd.DataFrame(
                {
                    "Mutation.type": ["C>A", "C>A"],
                    "Trinucleotide": ["ACA", "ACC"],
                    "TumorA::S1": [10, 20],
                    "TumorB::S2": [5, 15],
                }
            ).to_csv(sample_source, index=False)
            source_manifest = root / "public_data_manifest.tsv"
            pd.DataFrame(
                [
                    {
                        "source_label": "fixture",
                        "source_url": "file://fixture",
                        "source_reference": "unit test",
                        "sample_source": str(sample_source),
                        "n_contexts": 2,
                        "n_selected_samples": 2,
                    }
                ]
            ).to_csv(source_manifest, sep="\t", index=False)

            step = root / "suite" / "raw" / "decision_full_catalog"
            (step / "fusion").mkdir(parents=True)
            (step / "cohort").mkdir()
            (step / "experts").mkdir()
            pd.DataFrame(
                [
                    {
                        "sample_id": "TumorA::S1",
                        "fusion_mode": "consensus",
                        "primary_recommendation": "direct_downstream_analysis",
                        "catalog_insufficiency_level": "low",
                        "catalog_insufficiency_probability": 0.1,
                        "catalog_insufficiency_proxy_score": 0.1,
                        "assignment_confidence_probability": 0.8,
                        "mean_reconstruction_cosine": 0.95,
                        "residual_structure_score": 0.2,
                        "mutation_count": 30,
                    }
                ]
            ).to_csv(step / "fusion" / "summary.tsv", sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "sample_id": "TumorA::S1",
                        "candidate_type": "manual_review",
                        "reason": "fixture",
                    }
                ]
            ).to_csv(step / "cohort" / "candidates.tsv", sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "expert_name": "plain_nnls",
                        "status": "success",
                        "sample_id": "TumorA::S1",
                        "active_signature_count": 2,
                        "reconstruction_cosine": 0.95,
                        "rss": 1.0,
                        "runtime_seconds": 0.01,
                    }
                ]
            ).to_csv(step / "experts" / "summary.tsv", sep="\t", index=False)
            reduced_step = root / "suite" / "raw" / "decision_reduced_catalog"
            (reduced_step / "fusion").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "sample_id": "TumorA::S1",
                        "fusion_mode": "fallback",
                        "primary_recommendation": "manual_review",
                        "catalog_insufficiency_level": "medium",
                        "catalog_insufficiency_probability": 0.4,
                        "catalog_insufficiency_proxy_score": 0.4,
                        "assignment_confidence_probability": 0.6,
                        "mean_reconstruction_cosine": 0.90,
                        "residual_structure_score": 0.5,
                        "mutation_count": 30,
                    }
                ]
            ).to_csv(reduced_step / "fusion" / "summary.tsv", sep="\t", index=False)
            active_reduced_step = root / "suite" / "raw" / "decision_active_reduced_catalog"
            (active_reduced_step / "fusion").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "sample_id": "TumorA::S1",
                        "fusion_mode": "fallback",
                        "primary_recommendation": "cohort_level_discovery",
                        "catalog_insufficiency_level": "high",
                        "catalog_insufficiency_probability": 0.8,
                        "catalog_insufficiency_proxy_score": 0.8,
                        "assignment_confidence_probability": 0.2,
                        "mean_reconstruction_cosine": 0.70,
                        "residual_structure_score": 0.9,
                        "mutation_count": 30,
                    }
                ]
            ).to_csv(active_reduced_step / "fusion" / "summary.tsv", sep="\t", index=False)

            make_real_data_stress_tables(
                root / "suite",
                root / "suite" / "tables",
                sample_source=sample_source,
                source_manifest=source_manifest,
            )
            samples = pd.read_csv(root / "suite" / "tables" / "real_data_sample_summary.tsv", sep="\t")
            counts = pd.read_csv(root / "suite" / "tables" / "real_data_recommendation_counts.tsv", sep="\t")
            public_manifest = pd.read_csv(root / "suite" / "tables" / "public_data_manifest.tsv", sep="\t")
            delta = pd.read_csv(root / "suite" / "tables" / "real_data_catalog_stress_delta.tsv", sep="\t")
            design_summary = pd.read_csv(root / "suite" / "tables" / "real_data_stress_design_summary.tsv", sep="\t")

        self.assertEqual(samples.loc[0, "source_tumor_type"], "TumorA")
        self.assertIn("primary_recommendation", counts["summary_dimension"].tolist())
        self.assertIn("decision_full_catalog", counts["step_name"].tolist())
        self.assertIn("analysis_sample_source", public_manifest["source_label"].tolist())
        self.assertEqual(len(delta), 2)
        self.assertIn("data_driven_active_signature_removal", delta["stress_design"].tolist())
        self.assertAlmostEqual(
            float(
                delta.loc[
                    delta["stress_design"].eq("fixed_sbs1_sbs5_removal"),
                    "catalog_insufficiency_probability_delta_reduced_minus_full",
                ].iloc[0]
            ),
            0.3,
        )
        active_summary = design_summary.loc[
            design_summary["stress_design"].eq("data_driven_active_signature_removal")
        ].iloc[0]
        self.assertEqual(int(active_summary["n_primary_recommendation_changed"]), 1)


if __name__ == "__main__":
    unittest.main()
