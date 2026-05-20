from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_real_data_removal_design import (
    build_real_data_removal_design,
    write_real_data_removal_design,
)


class RealDataRemovalDesignTest(unittest.TestCase):
    def test_build_real_data_removal_design_ranks_total_fused_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            decision_dir = root / "decision"
            (decision_dir / "fusion").mkdir(parents=True)
            (decision_dir / "fusion" / "fused_run.json").write_text(
                json.dumps(
                    {
                        "sample_results": [
                            {"sample_id": "s1", "exposures": {"SBS1": 10.0, "SBS2": 5.0, "SBS3": 0.0}},
                            {"sample_id": "s2", "exposures": {"SBS1": 1.0, "SBS2": 20.0, "SBS3": 2.0}},
                        ]
                    }
                )
            )

            design = build_real_data_removal_design(decision_dir, top_n=2)

        self.assertEqual(design["signature_name"].head(2).tolist(), ["SBS2", "SBS1"])
        self.assertEqual(design["selected_for_removal"].head(2).tolist(), [True, True])

    def test_write_real_data_removal_design_creates_catalog_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            decision_dir = root / "decision"
            (decision_dir / "fusion").mkdir(parents=True)
            (decision_dir / "fusion" / "fused_run.json").write_text(
                json.dumps(
                    {
                        "sample_results": [
                            {"sample_id": "s1", "exposures": {"SBS1": 10.0, "SBS2": 5.0}},
                            {"sample_id": "s2", "exposures": {"SBS1": 1.0, "SBS2": 20.0}},
                        ]
                    }
                )
            )
            signature_source = root / "sigs.csv"
            pd.DataFrame(
                {
                    "Trinucleotide": ["ACA", "ACC"],
                    "SBS1": [0.1, 0.9],
                    "SBS2": [0.2, 0.8],
                    "SBS3": [0.3, 0.7],
                }
            ).to_csv(signature_source, index=False)

            write_real_data_removal_design(
                decision_dir=decision_dir,
                signature_source=signature_source,
                output_path=root / "variant.csv",
                manifest_path=root / "manifest.tsv",
                top_n=1,
            )
            variant = pd.read_csv(root / "variant.csv")
            manifest = pd.read_csv(root / "manifest.tsv", sep="\t")

        self.assertEqual(variant.columns.tolist(), ["Trinucleotide", "SBS1", "SBS3"])
        self.assertEqual(manifest.loc[0, "removed_signatures"], "SBS2")


if __name__ == "__main__":
    unittest.main()
