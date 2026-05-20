# SigAgent Release Package Plan

Status date: 2026-05-14.

This file defines the reviewer-testable and public-submission release boundary for the BMC Bioinformatics Software article track. It is a release engineering document, not a replacement for the manuscript availability statement.

## Current Gate Status

Two release stages are maintained by `experiments/validate_release_readiness.py`.

| Stage | Current status | Evidence |
|---|---|---|
| Reviewer package | Clear | `results/paper/reviewer_package_readiness/release_readiness_report.md`: 16 OK, 5 warnings, 0 blockers |
| Public submission | Not ready | `results/paper/release_readiness/release_readiness_report.md`: 16 OK, 3 warnings, 2 blockers |

The public-submission blockers are:

1. No author- or institution-approved top-level `LICENSE` is present for original SigAgent code.
2. Public repository and archive placeholders remain in the availability statement because no public release/tag/DOI has been created yet.

## Reviewer-Testable Core

The reviewer package should include an always-runnable core path that does not depend on vendored third-party software with unresolved redistribution terms. The default expert set for this release core is `plain_nnls`; all other assignment-tool integrations are optional adapters requested explicitly with `--expert-names`.

Include:

- `signature_decision/`
- `skills/mutational-signature-decision/`
- `experiments/`
- `tests/`
- `requirements.txt`, `requirements-optional.txt`, and `environment.yml`
- `README.md`, `REVIEWER_QUICKSTART.md`, `THIRD_PARTY_NOTICES.md`, and this file
- `paper/data_code_availability_statement.md`
- `paper/software_availability_checklist.md`
- `results/paper/paper_release_smoke/fixtures/manifest.json` and generated smoke-suite outputs when the archive includes result artifacts

The reviewer smoke suite uses deterministic toy SBS96 fixtures and `plain_nnls`. It checks installation, input/output contracts, decision-record generation, manifests, and checksums. It is not biological performance evidence.

## Public Archive Exclusion Policy

Do not include these assets in a public release unless their source, citation, license, and redistribution terms are resolved:

- `AMuSa/`: local artifact lacks visible README, LICENSE, and citation metadata. The exclusion policy is recorded in `paper/amusa_release_exclusion.md`.
- `MuSiCal/`: local license restricts redistribution/use; keep as an optional external integration unless permission is obtained.
- `SigProfilerAssignment/`: local license file is BSD 2-Clause, but local README contains GPL wording. Prefer optional external installation until the upstream release-tag license is verified and documented.
- Legacy `Data/` fixtures: include only if their provenance is documented and acceptable for the public archive; otherwise regenerate public fixtures from scripts or exclude them.

## Commands To Reproduce Release Checks

```bash
python -m unittest discover -s tests
python experiments/run_paper_suite.py experiments/configs/paper_release_smoke.json
python experiments/make_paper_figures.py --root results/paper --output-dir paper/figures
python experiments/validate_paper_outputs.py --root results/paper --output-dir results/paper/paper_readiness
python experiments/validate_release_readiness.py --repo-root . --results-root results/paper --output-dir results/paper/reviewer_package_readiness --release-stage review_package
python experiments/validate_release_readiness.py --repo-root . --results-root results/paper --output-dir results/paper/release_readiness --release-stage public_submission
```

Current local verification on 2026-05-14:

- Unit tests: 28 tests passed.
- Release smoke suite: 2 successful steps.
- Default-core CLI smoke: an invocation without `--expert-names` records `expert_names = ["plain_nnls"]`.
- Paper readiness: 39 required artifacts checked, 0 missing, 61 suite steps successful.

## Minimum Public-Submission Actions

Before journal submission, complete these actions outside the local code edits:

1. Choose and add an author- and institution-approved top-level license for original SigAgent code.
2. Create a public repository release with a stable version tag.
3. Archive the release on a persistent archive such as Zenodo and record the DOI.
4. Replace manuscript and availability-statement placeholders with the public URL, version tag, DOI, and license.
5. Re-run the `public_submission` release gate and keep the generated report in `results/paper/release_readiness/`.

## Relation To Manuscript Claims

The manuscript should frame SigAgent as a decision-support and benchmark framework built around the plain-NNLS evidence path plus transparent rule fusion. MuSiCal, SigProfilerAssignment, and AMuSA adapters demonstrate extensibility, but they should be described as optional external integrations unless their installation, provenance, and license terms are fully resolved for the submitted archive. Claims about multi-expert superiority, deployment-ready calibrated probabilities, or validated discovery of new signatures should remain out of scope.
