# SigAgent minimal release

SigAgent is a decision-support framework for mutational signature assignment. This minimal public package keeps the self-contained release core: plain NNLS assignment, rule-based decision records, catalog-insufficiency diagnostics, review-queue outputs, and a deterministic smoke benchmark.

The minimal package intentionally does not vendor MuSiCal, SigProfilerAssignment, AMuSA, or legacy internal data fixtures. Those tools can still be used through optional adapters when users install or provide them separately under their own licenses.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

If editable installation is not needed, install the core scientific stack with:

```bash
python -m pip install -r requirements.txt
```

## Verify the core release

```bash
python -m unittest tests.test_optional_expert_imports tests.test_release_smoke_fixtures tests.test_paper_suite_runner
python experiments/run_paper_suite.py experiments/configs/paper_release_smoke.json
```

The smoke suite generates toy SBS96 fixtures and runs the release-core `plain_nnls` path. It is intended to verify installation and execution, not to support manuscript-scale performance claims.

## Run a decision workflow

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py decision \
  --sample-source path/to/samples.csv \
  --signature-source path/to/signatures.csv \
  --mutation-type SBS96 \
  --expert-names plain_nnls \
  --output-dir results/my_decision_run
```

Input matrices should follow the schema in `skills/mutational-signature-decision/references/io-schema.md`.

## Optional external adapters

Optional adapters are present in the code but disabled by default. Users must install or provide the external tools themselves.

- MuSiCal: install according to upstream instructions or set `SIGAGENT_MUSICAL_PATH=/path/to/MuSiCal`, then request `--expert-names plain_nnls,musical`.
- SigProfilerAssignment: install according to upstream instructions or set `SIGAGENT_SIGPROFILERASSIGNMENT_PATH=/path/to/SigProfilerAssignment`, then request `--expert-names plain_nnls,sigprofiler_assignment`.
- AMuSA-derived adapters: provide AMuSA on `PYTHONPATH` or set `SIGAGENT_AMUSA_PATH=/path/to/parent-or-AMuSa`, then request `amusa`, `amusa_support_only`, or `classifier_guided_refit`.

The default registry remains `plain_nnls` only, so a clean install does not require optional third-party tools.

## Release boundary

This package excludes:

- `MuSiCal/`
- `SigProfilerAssignment/`
- `AMuSa/`
- `Data/`
- generated `results/`, runtime caches, and paper build artifacts

Before a public manuscript submission, add a repository-level `LICENSE`, create a version tag, and archive that tag to obtain a DOI. See `LICENSE_DECISION_NEEDED.md`, `THIRD_PARTY_NOTICES.md`, and `paper/data_code_availability_statement.md`.
