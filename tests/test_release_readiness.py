from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.make_release_smoke_fixtures import make_release_smoke_fixtures
from experiments.validate_release_readiness import validate_release_readiness


class ReleaseReadinessTest(unittest.TestCase):
    def test_release_readiness_reports_known_blockers_and_valid_smoke_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            results_root = repo_root / "results" / "paper"
            output_dir = results_root / "release_readiness"

            for relative in [
                "README.md",
                "REVIEWER_QUICKSTART.md",
                "requirements.txt",
                "requirements-optional.txt",
                "THIRD_PARTY_NOTICES.md",
                "paper/data_code_availability_statement.md",
                "paper/license_decision_needed.md",
                "experiments/make_release_smoke_fixtures.py",
                "experiments/configs/paper_release_smoke.json",
                "paper/figures/figure_manifest.tsv",
            ]:
                path = repo_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".tsv":
                    path.write_text("figure_id\tstatus\nfigure_1\tok\n")
                else:
                    path.write_text("placeholder\n")
            (repo_root / "paper/data_code_availability_statement.md").write_text("[Zenodo DOI]\n")

            fixture_dir = results_root / "paper_release_smoke" / "fixtures"
            make_release_smoke_fixtures(fixture_dir, burden=100, random_seed=1)
            suite_dir = results_root / "paper_release_smoke" / "manifests"
            suite_dir.mkdir(parents=True, exist_ok=True)
            (suite_dir / "suite.manifest.json").write_text(
                '[{"step_name": "make", "status": "success", "returncode": 0}]\n'
            )
            metrics_dir = results_root / "paper_release_smoke" / "raw" / "known_sbs96_toy_plain_nnls"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "aggregate_metrics.tsv").write_text("expert_name\tstatus\nplain_nnls\tsuccess\n")

            readiness_dir = results_root / "paper_readiness"
            readiness_dir.mkdir(parents=True, exist_ok=True)
            (readiness_dir / "paper_output_inventory.tsv").write_text("status\nok\n")
            (readiness_dir / "paper_suite_status.tsv").write_text("status\nsuccess\n")

            result = validate_release_readiness(repo_root, results_root, output_dir)
            inventory = result["inventory"]

            self.assertIn("top_level_license", set(inventory["check_id"]))
            self.assertIn("release_smoke_fixture_checksums", set(inventory["check_id"]))
            smoke_status = inventory.loc[
                inventory["check_id"].eq("release_smoke_fixture_checksums"),
                "status",
            ].iloc[0]
            license_status = inventory.loc[inventory["check_id"].eq("top_level_license"), "status"].iloc[0]
            archive_status = inventory.loc[inventory["check_id"].eq("archive_placeholders"), "status"].iloc[0]
            self.assertEqual(smoke_status, "ok")
            self.assertEqual(license_status, "blocker")
            self.assertEqual(archive_status, "blocker")
            self.assertTrue((output_dir / "release_readiness_report.md").exists())


if __name__ == "__main__":
    unittest.main()
