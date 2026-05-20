from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from signature_decision.benchmark import _candidate_removed_signatures, _removal_metadata_by_signature


class BenchmarkRemovalManifestTest(unittest.TestCase):
    def test_manifest_selects_benchmarkable_unique_signatures_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.tsv"
            pd.DataFrame(
                [
                    {
                        "mutation_type": "SBS96",
                        "selection_group": "high_prevalence_active",
                        "signature_name": "SBS1",
                        "benchmarkable_with_active_labels": True,
                        "prevalence_count": 10,
                        "flatness_score": 0.4,
                    },
                    {
                        "mutation_type": "SBS96",
                        "selection_group": "flat_signature",
                        "signature_name": "SBS1",
                        "benchmarkable_with_active_labels": True,
                        "prevalence_count": 10,
                        "flatness_score": 0.4,
                    },
                    {
                        "mutation_type": "SBS96",
                        "selection_group": "inactive_control",
                        "signature_name": "SBS2",
                        "benchmarkable_with_active_labels": False,
                        "prevalence_count": 0,
                        "flatness_score": 0.8,
                    },
                ]
            ).to_csv(path, sep="\t", index=False)

            candidates = _candidate_removed_signatures(
                ["SBS1", "SBS2"],
                removed_signatures=None,
                removal_manifest_source=path,
            )
            metadata = _removal_metadata_by_signature(path)

        self.assertEqual(candidates, ["SBS1"])
        self.assertEqual(metadata["SBS1"]["removal_selection_groups"], "flat_signature,high_prevalence_active")
        self.assertEqual(int(metadata["SBS1"]["removal_prevalence_count"]), 10)

    def test_explicit_removed_signatures_override_manifest_candidates(self) -> None:
        candidates = _candidate_removed_signatures(
            ["SBS1", "SBS2"],
            removed_signatures=["SBS2"],
            removal_manifest_source=None,
        )

        self.assertEqual(candidates, ["SBS2"])


if __name__ == "__main__":
    unittest.main()

