from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class OptionalExpertImportTest(unittest.TestCase):
    def test_top_level_import_does_not_eagerly_load_amusa_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            "import sys; "
            "import signature_decision; "
            "from signature_decision.experts import AMuSaExpert, PlainNNLSExpert; "
            "assert 'AMuSa.runtime' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert AMuSaExpert.expert_name == 'amusa'; "
            "assert PlainNNLSExpert.expert_name == 'plain_nnls'"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_registry_uses_release_core_only(self) -> None:
        from signature_decision.experts import build_default_registry

        registry = build_default_registry(Path(__file__).resolve().parents[1])

        self.assertEqual(registry.default_names(), ["plain_nnls"])
        self.assertIn("musical", registry.names())
        self.assertIn("sigprofiler_assignment", registry.names())
        self.assertIn("amusa", registry.names())


if __name__ == "__main__":
    unittest.main()
