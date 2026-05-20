# I/O Schema

Use this file only when exact field names or output shapes matter.

## Decision mode inputs

- `sample_source`: mutation catalog CSV
- `signature_source`: signature matrix CSV
- `mutation_type`: `SBS96` | `DBS78` | `ID83`
- `sample_ids` optional: comma-separated subset of samples
- `expert_names` optional: defaults to `amusa,classifier_guided_refit,musical,plain_nnls,sigprofiler_assignment`
- `confidence_artifact` optional: calibration artifact JSON from `fit-confidence`
- `catalog_assessor_artifact` optional: trained assessor JSON from `fit-catalog-assessor`
- `bootstrap_replicates` optional: enables bootstrap exposure intervals

## Decision mode outputs

### `fusion/reports.json`

List of `FinalSampleReport` objects. Each report contains:

```json
{
  "sample_id": "sample_x",
  "mutation_type": "SBS96",
  "known_signatures": [
    {
      "name": "SBS1",
      "active_proxy_score": 0.94,
      "active_probability": 0.91,
      "stability": "high",
      "exposure": 0.31,
      "exposure_interval": [0.22, 0.39],
      "exposure_interval_source": "bootstrap_conformal",
      "supporting_experts": ["amusa", "musical"],
      "dissenting_experts": ["sigprofiler_assignment"],
      "trusted_experts": ["amusa", "musical"]
    }
  ],
  "unstable_conclusions": [
    {
      "name": "SBS5",
      "active_probability": 0.42,
      "exposure_interval": [0.0, 0.08],
      "reason": "Mixed expert support or wide interval after rule fusion."
    }
  ],
  "catalog_insufficiency_proxy_score": 0.63,
  "catalog_insufficiency_probability": 0.63,
  "catalog_insufficiency_level": "medium",
  "assignment_confidence_raw_score": 0.48,
  "assignment_confidence_probability": 0.55,
  "assignment_confidence": 0.55,
  "primary_recommendation": "manual_review",
  "secondary_recommendations": ["reassess_reference_catalog"],
  "recommendation_rationale": [
    "Large expert disagreement coincides with structured residual signal."
  ],
  "metadata": {
    "fusion_mode": "disagreement_review",
    "agreement_score": 0.41,
    "mean_reconstruction_cosine": 0.91,
    "residual_structure_score": 0.58,
    "mutation_count": 243.0
  }
}
```

### `cohort/summary.json`

`CohortSummaryReport` object. Key fields:

- `mutation_type`
- `sample_ids`
- `n_samples`
- `recommendation_counts`
- `catalog_insufficiency_level_counts`
- `manual_review_candidates`
- `cohort_discovery_candidates`
- `reference_reassessment_candidates`
- `direct_downstream_candidates`

### `cohort/candidates.tsv`

Flat candidate table with:

- `sample_id`
- `candidate_type`
- `reason`

### `fusion/fused_run.json`

Same high-level shape as each expert output:

- `expert_name`
- `status`
- `sample_results`
- `parameters`
- `artifacts`
- `warnings`
- `error`

### `experts/*.json`

Each expert run contains:

- `sample_results[].active_signatures`
- `sample_results[].exposures`
- `sample_results[].signature_scores`
- `sample_results[].signature_probabilities`
- `sample_results[].metrics`
- `sample_results[].diagnostics`

## Benchmark mode inputs

Known benchmark:

- `sample_source`
- `signature_source`
- `exposure_source`
- `mutation_type`
- `burdens`
- `max_samples_per_burden`

Catalog-insufficiency benchmark:

- same as above
- `removed_signatures`
- `max_positive_per_signature`
- `max_negative_per_signature`
- `active_threshold`

## Discovery mode inputs

- `experience_dir`: append-only experience store emitted by `decision`
- `sample_source` optional: override sample catalog source used to recover channel layout
- `signature_source` optional: override reference catalog source used for constrained refit and matching
- `probability_threshold`
- `residual_structure_threshold`
- `min_recurrence_count`
- `mutation_count_threshold`
- `similarity_threshold`
- `min_cluster_size`
- `max_components`
- `min_records`
- `min_mean_residual_mass`
- `min_error_gain`
- `min_component_weight_fraction`
- `bootstrap_repeats`
- `disable_review_confirmation` optional: intended for debugging, not the default workflow

## Benchmark mode outputs

### `aggregate_metrics.tsv`

Typical columns:

- `expert_name`
- `status`
- `burden`
- `sample_precision`
- `sample_recall`
- `sample_f1`
- `signature_precision`
- `signature_recall`
- `signature_f1`
- `exposure_tvd`
- `exposure_mae`
- `exposure_rmse`
- `exposure_cosine`
- `reconstruction_cosine`
- `reconstruction_tvd`
- `calibration_brier`
- `calibration_ece`
- `calibration_auroc`
- `calibration_auprc`

Catalog-insufficiency benchmark adds:

- `removed_signature`
- `catalog_insufficiency_auroc`
- `catalog_insufficiency_auprc`
- `catalog_insufficiency_probability_brier`
- `catalog_insufficiency_probability_ece`
- `catalog_insufficiency_probability_auroc`
- `catalog_insufficiency_probability_auprc`

Known benchmark may additionally include for `rule_fusion`:

- `assignment_confidence_brier`
- `assignment_confidence_ece`
- `assignment_confidence_auroc`
- `assignment_confidence_auprc`

### `per_sample_metrics.tsv`

Typical columns:

- `expert_name`
- `sample_id`
- `active_set_precision`
- `active_set_recall`
- `active_set_f1`
- `exposure_tvd`
- `exposure_mae`
- `exposure_rmse`
- `exposure_cosine`
- `reconstruction_cosine`
- `reconstruction_tvd`

Catalog-insufficiency benchmark may additionally include:

- `catalog_insufficient_label`
- `catalog_insufficiency_score`
- `catalog_insufficiency_probability`
- `catalog_insufficiency_level`

## Discovery mode outputs

### `summary.json`

High-level trigger summary:

- `n_records`
- `n_candidate_pool`
- `n_clusters`
- `n_ready`
- `n_blocked`
- `n_packets`
- `config`
- `load_warnings`

### `trigger_candidates.tsv`

Flat table of all trigger decisions:

- `record_id`
- `sample_id`
- `mutation_type`
- `trigger_status`
- `cluster_id`
- `recurrence_count`
- `priority_score`
- `catalog_insufficiency_probability`
- `residual_structure_score`
- `mutation_count`
- `review_gate_status`
- `passed_conditions`
- `blocked_conditions`
- `rationale`

### `recurrence_clusters.tsv`

Cluster-level recurrence summary:

- `cluster_id`
- `mutation_type`
- `record_ids`
- `sample_ids`
- `fingerprint`
- `recurrence_count`
- `mean_pairwise_similarity`

### `packets.json`

List of `DiscoveryPacket` objects. Key fields:

- `packet_id`
- `packet_status`
- `trigger_summary`
- `recurrence_summary`
- `candidate_records`
- `extracted_components`
- `catalog_match_summary`
- `fit_improvement_summary`
- `recommended_actions`
- `writeback_policy`

### `packets.tsv`

One row per packet:

- `packet_id`
- `packet_status`
- `n_candidate_records`
- `n_extracted_components`
- `recurrence_count`
- `mean_pairwise_similarity`
- `mean_delta_reconstruction_cosine_vs_current`
- `mean_delta_reconstruction_cosine_vs_known_only`
- `mean_delta_relative_l1_pct_vs_current`
- `mean_delta_relative_l1_pct_vs_known_only`

### `extracted_components.tsv`

One row per extracted residual component:

- `packet_id`
- `component_id`
- `recurrence_count`
- `stability_score`
- `catalog_match_name`
- `catalog_match_cosine`
- `mean_residual_mass`

### `fit_improvements.tsv`

Mixed aggregate + per-sample fit comparison table:

- aggregate rows:
  - `mean_delta_reconstruction_cosine_vs_current`
  - `mean_delta_reconstruction_cosine_vs_known_only`
  - `mean_delta_relative_l1_pct_vs_current`
  - `mean_delta_relative_l1_pct_vs_known_only`
- per-sample rows:
  - `sample_id`
  - `known_only_reconstruction_cosine`
  - `augmented_reconstruction_cosine`
  - `delta_reconstruction_cosine_vs_current`
  - `delta_reconstruction_cosine_vs_known_only`
