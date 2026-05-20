from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_discovery_tables import make_discovery_tables


class DiscoveryTablesTest(unittest.TestCase):
    def test_make_discovery_tables_flattens_catalog_hits_and_summarizes_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            step = root / "raw" / "discovery_step"
            step.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "record_id": "r1",
                        "sample_id": "s1",
                        "mutation_type": "SBS96",
                        "trigger_status": "ready",
                        "review_gate_status": "disabled",
                        "priority_score": 0.9,
                        "catalog_insufficiency_probability": 0.8,
                        "residual_structure_score": 0.7,
                        "mutation_count": 1000,
                        "recurrence_count": 2,
                    }
                ]
            ).to_csv(step / "trigger_candidates.tsv", sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "packet_id": "p1",
                        "mutation_type": "SBS96",
                        "packet_status": "ready_for_review",
                        "n_candidate_records": 1,
                    }
                ]
            ).to_csv(step / "packets.tsv", sep="\t", index=False)
            (step / "packets.json").write_text(
                json.dumps(
                    [
                        {
                            "packet_id": "p1",
                            "catalog_match_summary": [
                                {
                                    "component_id": "c1",
                                    "top_hits": [
                                        {"signature_name": "SBS1", "cosine": 0.9},
                                        {"signature_name": "SBS5", "cosine": 0.8},
                                    ],
                                }
                            ],
                        }
                    ]
                )
            )

            make_discovery_tables(root, root / "tables")
            hits = pd.read_csv(root / "tables" / "discovery_catalog_hits.tsv", sep="\t")
            summary = pd.read_csv(root / "tables" / "discovery_trigger_summary.tsv", sep="\t")

        self.assertEqual(hits["signature_name"].tolist(), ["SBS1", "SBS5"])
        self.assertEqual(int(summary.loc[0, "n_candidates"]), 1)
        self.assertAlmostEqual(float(summary.loc[0, "priority_score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
