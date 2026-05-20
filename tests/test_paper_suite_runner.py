from __future__ import annotations

import unittest

from experiments.run_paper_suite import _build_command


class PaperSuiteRunnerTest(unittest.TestCase):
    def test_build_command_formats_flags_and_lists(self) -> None:
        command = _build_command(
            "known-benchmark",
            {
                "sample_source": "samples.csv",
                "burdens": [100, 200],
                "skip_rule_fusion": True,
                "confidence_artifact": None,
                "max_samples_per_burden": 5,
            },
        )

        self.assertIn("known-benchmark", command)
        self.assertIn("--sample-source", command)
        self.assertIn("samples.csv", command)
        self.assertIn("--burdens", command)
        self.assertIn("100,200", command)
        self.assertIn("--skip-rule-fusion", command)
        self.assertNotIn("--confidence-artifact", command)
        self.assertIn("--max-samples-per-burden", command)

    def test_build_command_uses_script_entrypoint_for_release_fixture_generation(self) -> None:
        command = _build_command(
            "release-smoke-fixtures",
            {
                "output_dir": "fixtures",
                "burden": 500,
            },
        )

        self.assertIn("make_release_smoke_fixtures.py", command[1])
        self.assertIn("--output-dir", command)
        self.assertIn("fixtures", command)
        self.assertIn("--burden", command)


if __name__ == "__main__":
    unittest.main()
