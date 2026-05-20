#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PRIMARY_TOLERANCE = 1e-12
RECONSTRUCTION_PRESERVATION_TOLERANCE = 0.005


@dataclass(frozen=True)
class MetricSpec:
    evidence_block: str
    metric: str
    column: str
    direction: str
    audit_tier: str
    tolerance: float
    source_table: str


METRIC_SPECS = [
    MetricSpec(
        evidence_block="complete_catalog",
        metric="sample_f1",
        column="sample_f1_mean",
        direction="higher_is_better",
        audit_tier="primary_non_degradation",
        tolerance=PRIMARY_TOLERANCE,
        source_table="known_catalog_overall_with_uncertainty.tsv",
    ),
    MetricSpec(
        evidence_block="complete_catalog",
        metric="exposure_tvd",
        column="exposure_tvd_mean",
        direction="lower_is_better",
        audit_tier="primary_non_degradation",
        tolerance=PRIMARY_TOLERANCE,
        source_table="known_catalog_overall_with_uncertainty.tsv",
    ),
    MetricSpec(
        evidence_block="complete_catalog",
        metric="reconstruction_cosine",
        column="reconstruction_cosine_mean",
        direction="higher_is_better",
        audit_tier="diagnostic_preservation",
        tolerance=RECONSTRUCTION_PRESERVATION_TOLERANCE,
        source_table="known_catalog_overall_with_uncertainty.tsv",
    ),
    MetricSpec(
        evidence_block="controlled_removal",
        metric="catalog_insufficiency_auroc",
        column="catalog_insufficiency_auroc_mean",
        direction="higher_is_better",
        audit_tier="primary_non_degradation",
        tolerance=PRIMARY_TOLERANCE,
        source_table="catalog_insufficiency_overall_with_uncertainty.tsv",
    ),
    MetricSpec(
        evidence_block="controlled_removal",
        metric="catalog_insufficiency_auprc",
        column="catalog_insufficiency_auprc_mean",
        direction="higher_is_better",
        audit_tier="primary_non_degradation",
        tolerance=PRIMARY_TOLERANCE,
        source_table="catalog_insufficiency_overall_with_uncertainty.tsv",
    ),
]


SUITES = [
    "paper_review_response_sbs96",
    "paper_review_response_dbs78_id83",
]


def _read_table(results_root: Path, suite: str, table_name: str) -> pd.DataFrame:
    path = results_root / suite / "tables" / table_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required audit source table: {path}")
    return pd.read_csv(path, sep="\t")


def _value(frame: pd.DataFrame, *, context: str, expert_name: str, column: str) -> float:
    if "mutation_type" not in frame.columns:
        raise ValueError("Audit source table lacks mutation_type column.")
    if "expert_name" not in frame.columns:
        raise ValueError("Audit source table lacks expert_name column.")
    if column not in frame.columns:
        raise ValueError(f"Audit source table lacks required metric column: {column}")
    mask = frame["mutation_type"].astype(str).eq(context) & frame["expert_name"].astype(str).eq(expert_name)
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {expert_name} {column} row for {context}, found {len(values)}.")
    return float(values.iloc[0])


def _passes(rule_value: float, plain_value: float, *, direction: str, tolerance: float) -> bool:
    if direction == "higher_is_better":
        return rule_value + tolerance >= plain_value
    if direction == "lower_is_better":
        return rule_value <= plain_value + tolerance
    raise ValueError(f"Unsupported direction: {direction}")


def build_non_degradation_audit(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    table_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for suite in SUITES:
        for spec in METRIC_SPECS:
            key = (suite, spec.source_table)
            if key not in table_cache:
                table_cache[key] = _read_table(results_root, suite, spec.source_table)
            frame = table_cache[key]
            for context in sorted(frame["mutation_type"].astype(str).unique()):
                plain_value = _value(frame, context=context, expert_name="plain_nnls", column=spec.column)
                rule_value = _value(frame, context=context, expert_name="rule_fusion", column=spec.column)
                delta = rule_value - plain_value
                passed = _passes(
                    rule_value,
                    plain_value,
                    direction=spec.direction,
                    tolerance=spec.tolerance,
                )
                rows.append(
                    {
                        "context": context,
                        "suite": suite,
                        "evidence_block": spec.evidence_block,
                        "metric": spec.metric,
                        "direction": spec.direction,
                        "plain_nnls_value": plain_value,
                        "rule_fusion_value": rule_value,
                        "delta_rule_minus_plain": delta,
                        "tolerance": spec.tolerance,
                        "audit_tier": spec.audit_tier,
                        "audit_status": "pass" if passed else "fail",
                        "source_table": f"{suite}/tables/{spec.source_table}",
                    }
                )
    audit = pd.DataFrame.from_records(rows)
    ordered_contexts = {"SBS96": 0, "DBS78": 1, "ID83": 2}
    ordered_metrics = {spec.metric: index for index, spec in enumerate(METRIC_SPECS)}
    audit["_context_order"] = audit["context"].map(ordered_contexts).fillna(99)
    audit["_metric_order"] = audit["metric"].map(ordered_metrics).fillna(99)
    audit = audit.sort_values(["_context_order", "_metric_order"]).drop(
        columns=["_context_order", "_metric_order"]
    )
    return audit.reset_index(drop=True)


def compact_non_degradation_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context, group in audit.groupby("context", sort=False):
        primary = group.loc[group["audit_tier"].eq("primary_non_degradation")]
        diagnostic = group.loc[group["audit_tier"].eq("diagnostic_preservation")]
        status = "pass" if group["audit_status"].eq("pass").all() else "fail"
        lookup = {
            str(row["metric"]): float(row["delta_rule_minus_plain"])
            for _, row in group.iterrows()
        }
        rows.append(
            {
                "context": context,
                "primary_checks_passed": int(primary["audit_status"].eq("pass").sum()),
                "primary_checks_total": int(len(primary)),
                "diagnostic_checks_passed": int(diagnostic["audit_status"].eq("pass").sum()),
                "diagnostic_checks_total": int(len(diagnostic)),
                "sample_f1_delta": lookup.get("sample_f1"),
                "exposure_tvd_delta": lookup.get("exposure_tvd"),
                "catalog_insufficiency_auroc_delta": lookup.get("catalog_insufficiency_auroc"),
                "catalog_insufficiency_auprc_delta": lookup.get("catalog_insufficiency_auprc"),
                "reconstruction_cosine_delta": lookup.get("reconstruction_cosine"),
                "status": status,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_headline(audit: pd.DataFrame, compact: pd.DataFrame, output_dir: Path) -> None:
    primary = audit.loc[audit["audit_tier"].eq("primary_non_degradation")]
    diagnostic = audit.loc[audit["audit_tier"].eq("diagnostic_preservation")]
    lines = [
        "# Non-Degradation Audit",
        "",
        "Rule fusion is audited against plain NNLS because SigAgent is an integration layer.",
        "Primary checks require rule fusion to match or improve sample F1, exposure TVD, catalog-insufficiency AUROC, and catalog-insufficiency AUPRC.",
        f"Reconstruction cosine is treated as a preservation diagnostic with tolerance `{RECONSTRUCTION_PRESERVATION_TOLERANCE:.3f}`, not as a dominance claim.",
        "",
        f"- Primary checks passing: `{int(primary['audit_status'].eq('pass').sum())}/{len(primary)}`.",
        f"- Diagnostic preservation checks passing: `{int(diagnostic['audit_status'].eq('pass').sum())}/{len(diagnostic)}`.",
        "",
        "## Context Summary",
        "",
    ]
    for _, row in compact.iterrows():
        lines.append(
            f"- {row['context']}: primary `{int(row['primary_checks_passed'])}/{int(row['primary_checks_total'])}`, "
            f"diagnostic `{int(row['diagnostic_checks_passed'])}/{int(row['diagnostic_checks_total'])}`, "
            f"sample F1 delta `{float(row['sample_f1_delta']):.6f}`, "
            f"exposure TVD delta `{float(row['exposure_tvd_delta']):.6f}`, "
            f"AUROC/AUPRC deltas `{float(row['catalog_insufficiency_auroc_delta']):.6f}`/"
            f"`{float(row['catalog_insufficiency_auprc_delta']):.6f}`, "
            f"reconstruction-cosine delta `{float(row['reconstruction_cosine_delta']):.6f}`."
        )
    (output_dir / "non_degradation_headline.md").write_text("\n".join(lines) + "\n")


def make_non_degradation_audit(results_root: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_non_degradation_audit(results_root)
    compact = compact_non_degradation_summary(audit)
    audit.to_csv(output_dir / "non_degradation_summary.tsv", sep="\t", index=False)
    compact.to_csv(output_dir / "non_degradation_compact.tsv", sep="\t", index=False)
    write_headline(audit, compact, output_dir)
    if not audit["audit_status"].eq("pass").all():
        failing = audit.loc[audit["audit_status"].ne("pass")]
        raise RuntimeError(f"Non-degradation audit failed for {len(failing)} comparison(s).")
    return {"audit": audit, "compact": compact}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a rule-fusion non-degradation audit against plain NNLS.")
    parser.add_argument("--root", default="results/paper", help="Paper results root.")
    parser.add_argument(
        "--output-dir",
        default="results/paper/paper_non_degradation_audit/tables",
        help="Output directory for audit tables.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_non_degradation_audit(Path(args.root), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
