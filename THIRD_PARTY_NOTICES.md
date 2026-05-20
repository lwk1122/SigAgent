# Third-Party Notices

Status: draft for manuscript and release preparation. This file is not legal advice; it records the current evidence visible in this workspace and the actions needed before a public software release.

## Release Policy

SigAgent original code should be released under a top-level license selected by the authors or institution. Third-party mutational-signature toolkits should be treated as optional integrations unless redistribution is explicitly permitted.

## Components

| Component | Local path | Current evidence | Release action |
|---|---|---|---|
| SigAgent original code | `signature_decision/`, `experiments/`, `skills/mutational-signature-decision/` | No top-level license currently exists. | Add an author-approved top-level `LICENSE` before public release. |
| AMuSa local runtime/checkpoints | `AMuSa/` | No local `LICENSE`, `README`, or `CITATION` file was found. A quick public web search on 2026-05-09 did not identify a stable citation/license for the exact local artifact. A release-exclusion note is maintained in `paper/amusa_release_exclusion.md`. | Do not redistribute in a public release until provenance, citation, and permission are resolved. If unresolved, remove from the release package and omit AMuSA-dependent submission claims. |
| MuSiCal | `MuSiCal/` | Local `MuSiCal/LICENSE` grants internal academic/non-commercial use and restricts third-party distribution of derivative works without Harvard consent. The citable paper is Jin et al., Nature Genetics 2024, DOI `10.1038/s41588-024-01659-0`. | Do not vendor in a public SigAgent release without permission. Keep as an optional external dependency/integration and cite MuSiCal. |
| SigProfilerAssignment | `SigProfilerAssignment/` | Local `LICENSE.txt` is BSD 2-Clause. Local `README.md` contains GPL wording, while the Bioinformatics article states SigProfilerAssignment is available under BSD 2-Clause. | Prefer optional external installation. Do not vendor unless the upstream release-tag license is verified and the README wording discrepancy is documented. |
| PCAWG7 SBS96 public matrix | `results/paper/public_data/pcawg_sbs96_smoke/` | Pinned public URL and checksum are recorded in `public_data_manifest.tsv` and `source_manifest.json`. | Preserve URL, retrieval date, checksum, and sample-selection manifest in the reproducibility package. |
| Local synthetic fixtures | `Data/` | Synthetic signature/exposure/catalog fixtures are present locally. `Data/README.md` and `Data/manifest.json` document them as legacy internal fixtures with checksums, but the original generation script is not available in the current workspace. | Regenerate from documented scripts before public archive, or exclude these legacy files from the public release package and use deterministic release-smoke fixtures instead. |

## External Source Notes

- MuSiCal paper: Jin H, Gulhan DC, Geiger B, et al. Accurate and sensitive mutational signature analysis with MuSiCal. Nature Genetics 56, 541-552 (2024). https://doi.org/10.1038/s41588-024-01659-0
- SigProfilerAssignment paper: Assigning mutational signatures to individual samples and individual somatic mutations with SigProfilerAssignment. Bioinformatics 39(12), btad756 (2023). https://doi.org/10.1093/bioinformatics/btad756
- SigProfilerAssignment code availability in the article states BSD 2-Clause availability at the AlexandrovLab GitHub repository and PyPI.

## Current Release Recommendation

For a first BMC Bioinformatics-style submission package:

1. Release SigAgent original code, paper-suite scripts, and deterministic release-smoke fixtures under an approved license.
2. Exclude vendored MuSiCal, SigProfilerAssignment, and unresolved AMuSa assets from the public archive unless permission/provenance is resolved.
3. Provide optional adapter instructions for users who install or provide MuSiCal, SigProfilerAssignment, or AMuSa separately.
4. Ensure at least one smoke suite runs with the always-available `plain_nnls` core path and `requirements.txt`.
5. Archive paper result tables, figure scripts, manifests, and source checksums.

## Optional Adapter Paths

The minimal SigAgent release does not vendor third-party source trees. Users who have installed or downloaded optional tools can make them visible to SigAgent with these environment variables:

- `SIGAGENT_MUSICAL_PATH`: path to a MuSiCal checkout root or to the importable `musical` package directory.
- `SIGAGENT_SIGPROFILERASSIGNMENT_PATH`: path to a SigProfilerAssignment checkout root or to the importable `SigProfilerAssignment` package directory.
- `SIGAGENT_AMUSA_PATH`: path to an AMuSA checkout root, to the importable `AMuSa` package directory, or to a parent directory already suitable for `PYTHONPATH`.

If these variables are not set, SigAgent still supports the default `plain_nnls` release-core path.
