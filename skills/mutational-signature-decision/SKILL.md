---
name: mutational-signature-decision
description: Use this skill when you need to run, compare, benchmark, or rule-fuse mutational signature experts in this repository. It covers SBS96/DBS78/ID83 inputs, the release-core plain-NNLS path, optional AMuSa/MuSiCal/SigProfilerAssignment/classifier-guided adapters, known-catalog and catalog-insufficiency benchmarks, rule-based decision reporting for sample or cohort analysis, the V2 experience loop, and conservative local constrained discovery packets.
---

# Mutational Signature Decision

Use this skill for repository-local mutational signature work. It assumes the code in `signature_decision/` is the execution backbone and that the caller wants one of these workflows:

1. Run experts and produce sample/cohort decision reports.
2. Run `known-catalog` benchmark slices.
3. Run `catalog-insufficiency` benchmark slices.
4. Fit confidence calibration artifacts from known-catalog synthetic data.
5. Fit a trainable catalog-insufficiency assessor from incomplete-catalog synthetic data.
6. Persist decision outputs into an experience store, append review outcomes, and export reviewed datasets.
7. Run conservative local constrained discovery on the experience store and generate discovery packets.

## Default workflow

1. Pick the mode:
   - `decision`: run experts, then apply rule fusion.
   - `known-benchmark`: compare experts and optional rule fusion when the reference catalog is complete.
   - `catalog-insufficiency-benchmark`: remove one or more signatures from the catalog and measure whether the system raises insufficiency suspicion.
   - `fit-confidence`: fit AMuSa probability calibration, final assignment calibration, and conformal margin.
   - `fit-catalog-assessor`: fit a low-dimensional logistic catalog-insufficiency assessor.
   - `append-review`: append one review decision into the experience store and refresh queues/datasets.
   - `export-experience-dataset`: export reviewed records into a reusable TSV.
   - `discovery-run`: trigger secondary residual-only constrained discovery and emit discovery packets without updating the reference catalog.
2. Prefer the bundled script instead of ad hoc Python snippets:
   - `skills/mutational-signature-decision/scripts/run_signature_decision.py`
3. Only drop to direct Python when you are debugging internals or patching behavior.

Default expert policy:

- The release-core default is `plain_nnls`.
- `AMuSa`, `classifier_guided_refit`, `MuSiCal`, `SigProfilerAssignment`, and `amusa_support_only` are optional adapters and must be requested with `--expert-names`.
- Public/reviewer smoke paths should not require vendored third-party tool directories.

## Required inputs

- `sample_source`: mutation count matrix CSV.
- `signature_source`: reference signature matrix CSV.
- `mutation_type`: one of `SBS96`, `DBS78`, `ID83`.

Additional benchmark input:

- `exposure_source`: truth exposure CSV aligned to the same sample universe.

## Command patterns

Run a decision report for one or more samples:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py decision \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --mutation-type SBS96 \
  --sample-ids SP.Syn.Biliary..S.122,SP.Syn.Biliary..S.263 \
  --confidence-artifact out/confidence.json \
  --catalog-assessor-artifact out/catalog_assessor.json \
  --bootstrap-replicates 100 \
  --output-dir out/decision_sbs
```

Run optional external adapters only after the user has installed or provided them under their own licenses:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py decision \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --mutation-type SBS96 \
  --expert-names plain_nnls,musical,sigprofiler_assignment \
  --output-dir out/decision_sbs_optional_adapters
```

Fit confidence artifacts:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py fit-confidence \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --exposure-source Data/test_sbs_exposures.csv \
  --mutation-type SBS96 \
  --output-artifact out/confidence.json
```

Fit a catalog-insufficiency assessor:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py fit-catalog-assessor \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --exposure-source Data/test_sbs_exposures.csv \
  --mutation-type SBS96 \
  --confidence-artifact out/confidence.json \
  --output-artifact out/catalog_assessor.json
```

Run the known-catalog benchmark:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py known-benchmark \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --exposure-source Data/test_sbs_exposures.csv \
  --mutation-type SBS96 \
  --burdens 100,200,500,2000,50000 \
  --max-samples-per-burden 100 \
  --output-dir out/known_benchmark_sbs
```

Run the catalog-insufficiency benchmark:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py catalog-insufficiency-benchmark \
  --sample-source Data/test_sbs_catalog.csv \
  --signature-source Data/ground.truth.syn.sigs.SBS96.csv \
  --exposure-source Data/test_sbs_exposures.csv \
  --mutation-type SBS96 \
  --removed-signatures SBS21,SBS40 \
  --burdens 200,2000 \
  --output-dir out/catalog_insufficiency_sbs
```

## Output contract

Decision mode writes:

- `experts/*.json`
- `experts/summary.tsv`
- `experts/exposures.tsv`
- `fusion/fused_run.json`
- `fusion/reports.json`
- `fusion/reports.tsv`
- `fusion/summary.tsv`
- `cohort/summary.json`
- `cohort/summary.tsv`
- `cohort/candidates.tsv`
- `experience/records/*.json`
- `experience/reviews/*.json`
- `experience/index/*.tsv`
- `experience/queues/*.tsv`
- `experience/datasets/*.tsv`

Benchmark modes write:

- `result.json`
- `aggregate_metrics.tsv`
- `per_sample_metrics.tsv`

Read [references/io-schema.md](references/io-schema.md) when exact JSON and TSV field names matter.

## Version-1 fusion policy

This skill intentionally uses rule fusion, not a trained router.

- If selected experts have high active-set agreement and strong reconstruction, prefer consensus.
- If optional `AMuSa` and `MuSiCal` agree while `SigProfilerAssignment` diverges, prefer the first two.
- If disagreement is large and residual signal stays structured, raise catalog-insufficiency suspicion.
- If mutation count is low, lower exposure confidence and widen intervals automatically.
- When confidence and assessor artifacts are provided, keep `raw_score` and `calibrated probability` distinct in outputs.
- When bootstrap is enabled, replace heuristic exposure intervals with bootstrap or bootstrap-conformal intervals.

If the user asks for a learned router, treat that as a later-stage extension, not the default path.

## V2 Experience Loop

`decision` now also writes an append-only experience store by default under `<output-dir>/experience`.

Append one review outcome:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py append-review \
  --experience-dir out/decision_sbs/experience \
  --record-id <record-id> \
  --reviewer wk \
  --review-outcome confirmed \
  --evidence-type manual_review \
  --validated-recommendation cohort_level_discovery
```

Export reviewed records:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py export-experience-dataset \
  --experience-dir out/decision_sbs/experience \
  --output-path out/decision_sbs/experience/datasets/high_quality_dataset.tsv
```

Run conservative local constrained discovery:

```bash
python skills/mutational-signature-decision/scripts/run_signature_decision.py discovery-run \
  --experience-dir out/decision_sbs/experience \
  --output-dir out/decision_sbs/discovery
```

This command is intentionally conservative:

- it consumes calibrated `catalog_insufficiency_probability`, residual structure, cohort recurrence, burden thresholds, and review gating;
- it performs residual-only local extraction with at most `1-2` components;
- it compares `known-only refit` against `known + local component refit`;
- it generates `discovery packets` as evidence artifacts only;
- it does **not** modify the main catalog automatically.

Use `--disable-review-confirmation` only for debugging or smoke tests. The default workflow expects review-gated discovery.

## File ownership

When updating behavior, patch the relevant layer directly:

- Wrapper layer: `signature_decision/experts/`
- Metrics: `signature_decision/metrics.py`
- Benchmarks: `signature_decision/benchmark.py`
- Rule fusion: `signature_decision/fusion.py`
- Skill entrypoint: `skills/mutational-signature-decision/scripts/run_signature_decision.py`

## Notes

- The skill is repository-local. It is not installed into global Codex skill search automatically.
- For sample mode, pass one sample id. For cohort mode, pass many sample ids or omit `--sample-ids`.
- Benchmarks default to including `rule_fusion`. Disable it with `--skip-rule-fusion` if the user wants expert-only baselines.
