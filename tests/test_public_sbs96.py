from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.download_public_sbs96 import normalize_public_sbs96_catalog
from experiments.make_signature_catalog_variant import make_signature_catalog_variant


class PublicSBS96Test(unittest.TestCase):
    def test_normalize_public_sbs96_catalog_selects_balanced_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw.csv"
            pd.DataFrame(
                {
                    "Mutation type": ["C>A", "C>A", "C>G", "C>G"],
                    "Trinucleotide": ["ACA", "ACC", "ACA", "ACC"],
                    "TypeA::S1": [10, 1, 0, 0],
                    "TypeA::S2": [5, 1, 1, 0],
                    "TypeB::S3": [100, 0, 0, 1],
                    "TypeB::S4": [20, 0, 0, 0],
                }
            ).to_csv(raw, index=False)

            manifest = normalize_public_sbs96_catalog(
                raw,
                root / "out",
                source_url="file://raw.csv",
                max_samples=2,
                selection_mode="balanced-tumor-types",
                expected_context_count=4,
            )
            samples = pd.read_csv(root / "out" / "samples_sbs96.csv")
            sample_manifest = pd.read_csv(root / "out" / "sample_manifest.tsv", sep="\t")

        self.assertEqual(samples.columns[:2].tolist(), ["Mutation.type", "Trinucleotide"])
        self.assertEqual(set(samples.columns[2:]), {"TypeA::S1", "TypeB::S3"})
        self.assertEqual(manifest["n_selected_samples"], 2)
        self.assertEqual(sample_manifest["tumor_type"].tolist(), ["TypeB", "TypeA"])

    def test_make_signature_catalog_variant_removes_requested_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sigs.csv"
            pd.DataFrame(
                {
                    "Trinucleotide": ["ACA", "ACC"],
                    "SBS1": [0.1, 0.9],
                    "SBS5": [0.2, 0.8],
                    "SBS40": [0.3, 0.7],
                }
            ).to_csv(source, index=False)

            make_signature_catalog_variant(
                source,
                root / "variant.csv",
                remove_signatures=["SBS1", "SBS5"],
                manifest_path=root / "manifest.tsv",
            )
            variant = pd.read_csv(root / "variant.csv")
            manifest = pd.read_csv(root / "manifest.tsv", sep="\t")

        self.assertEqual(variant.columns.tolist(), ["Trinucleotide", "SBS40"])
        self.assertEqual(manifest.loc[0, "removed_signatures"], "SBS1,SBS5")
        self.assertEqual(int(manifest.loc[0, "n_remaining_signatures"]), 1)


if __name__ == "__main__":
    unittest.main()
