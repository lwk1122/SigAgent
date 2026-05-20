from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.make_release_smoke_fixtures import make_release_smoke_fixtures
from signature_decision.experts.io import load_expert_request


class ReleaseSmokeFixtureTest(unittest.TestCase):
    def test_generated_fixture_is_loadable_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = make_release_smoke_fixtures(output_dir, burden=250, random_seed=7)

            signature_path = output_dir / "toy_sbs96_signatures.csv"
            sample_path = output_dir / "toy_sbs96_samples.csv"
            exposure_path = output_dir / "toy_sbs96_exposures.csv"
            self.assertTrue(signature_path.exists())
            self.assertTrue(sample_path.exists())
            self.assertTrue(exposure_path.exists())
            self.assertEqual(manifest["n_channels"], 96)
            self.assertEqual(manifest["n_signatures"], 4)
            self.assertEqual(manifest["n_samples"], 6)

            request = load_expert_request(
                sample_source=sample_path,
                signature_source=signature_path,
                mutation_type="SBS96",
            )
            self.assertEqual(request.sample_matrix.shape, (96, 6))
            self.assertEqual(request.signature_matrix.shape, (96, 4))
            self.assertTrue((request.sample_matrix.sum(axis=0) == 250).all())

            exposures = pd.read_csv(exposure_path, index_col=0)
            self.assertEqual(exposures.shape, (4, 6))
            self.assertTrue((exposures.sum(axis=0).round(6) == 1.0).all())


if __name__ == "__main__":
    unittest.main()
