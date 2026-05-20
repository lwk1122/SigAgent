#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


DEFAULT_FIGURE_DPI = 300
FIGURE_SIZE = (7.2, 5.0)
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
NATURE_COLORS = {
    "blue": "#0F4D92",
    "blue_mid": "#3775BA",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "amber": "#DDAA33",
    "amber_soft": "#FFF9EA",
    "rose": "#BB5566",
    "grey": "#767676",
    "grey_light": "#CFCECE",
}
GROUP_COLORS = [
    NATURE_COLORS["blue"],
    NATURE_COLORS["blue_mid"],
    NATURE_COLORS["teal"],
    NATURE_COLORS["amber"],
    NATURE_COLORS["rose"],
    NATURE_COLORS["grey"],
]
EXPERT_COLORS = {"rule_fusion": NATURE_COLORS["blue"], "plain_nnls": NATURE_COLORS["grey"]}
TASK_COLORS = {"assignment_confidence": NATURE_COLORS["blue"], "catalog_insufficiency": NATURE_COLORS["amber"]}
NEUTRAL = {
    "ink": "#263238",
    "dark": "#2F4858",
    "mid": "#6C7A80",
    "light": "#E8EEF1",
    "wash": "#F6F8F9",
    "grid": "#D9E1E5",
}
ACCENT = {
    "sigagent": NATURE_COLORS["blue"],
    "baseline": NATURE_COLORS["grey"],
    "support": NATURE_COLORS["teal"],
    "warning": NATURE_COLORS["amber"],
    "review": NATURE_COLORS["rose"],
    "negative": NATURE_COLORS["blue_mid"],
}


@dataclass(slots=True)
class FigureRecord:
    figure_id: str
    title: str
    path_png: Path
    path_pdf: Path
    sources: list[Path]
    status: str = "ok"
    note: str = ""


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing figure source table: {path}")
    return pd.read_csv(path, sep="\t")


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": NEUTRAL["ink"],
            "axes.labelcolor": NEUTRAL["ink"],
            "xtick.color": NEUTRAL["ink"],
            "ytick.color": NEUTRAL["ink"],
            "text.color": NEUTRAL["ink"],
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _finalize_axes(fig: plt.Figure) -> None:
    for ax in fig.axes:
        if not ax.axison:
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
        color=NEUTRAL["ink"],
    )


def _write_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    _finalize_axes(fig)
    fig.savefig(png, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "SigAgent make_paper_figures.py"})
    fig.savefig(svg, bbox_inches="tight", metadata={"Creator": "SigAgent make_paper_figures.py"})
    plt.close(fig)
    return png, pdf


def _add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    title: str,
    body: str = "",
    facecolor: str = "white",
    edgecolor: str = NEUTRAL["dark"],
    title_color: str = NEUTRAL["ink"],
    body_color: str = NEUTRAL["mid"],
    linewidth: float = 1.0,
    radius: float = 0.018,
    fontsize: float = 8.0,
) -> mpatches.FancyBboxPatch:
    box = mpatches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.012, rounding_size=radius),
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        zorder=2,
    )
    box.set_path_effects([pe.withStroke(linewidth=linewidth + 1.1, foreground="white")])
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height * (0.68 if body else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=title_color,
        transform=ax.transAxes,
        zorder=3,
    )
    if body:
        ax.text(
            x + width / 2,
            y + height * 0.26,
            body,
            ha="center",
            va="center",
            fontsize=max(fontsize - 1.1, 6.2),
            color=body_color,
            linespacing=1.08,
            transform=ax.transAxes,
            zorder=3,
        )
    return box


def _add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NEUTRAL["dark"],
    linewidth: float = 1.1,
    rad: float = 0.0,
) -> None:
    arrow = mpatches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(arrow)


def _add_strip_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    stripe_color: str,
    fontsize: float = 6.8,
) -> None:
    _add_box(
        ax,
        xy,
        width,
        height,
        title=title,
        body=body,
        facecolor="white",
        edgecolor="#BFC7CC",
        title_color=NEUTRAL["ink"],
        body_color=NEUTRAL["mid"],
        linewidth=0.9,
        fontsize=fontsize,
    )
    x, y = xy
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x + 0.006, y + 0.010),
            0.006,
            height - 0.020,
            boxstyle=mpatches.BoxStyle("Round", pad=0.0, rounding_size=0.004),
            linewidth=0,
            facecolor=stripe_color,
            transform=ax.transAxes,
            zorder=4,
        )
    )


def _format_method_label(value: object) -> str:
    labels = {"rule_fusion": "Rule fusion", "plain_nnls": "Plain NNLS"}
    return labels.get(str(value), _clean_label(value).title())


def _format_group_label(value: object) -> str:
    labels = {
        "flat_signature": "Flat",
        "high_prevalence_active": "High prevalence",
        "low_prevalence_active": "Low prevalence",
        "peaky_signature": "Peaky",
        "high_similarity": "High similarity",
    }
    return labels.get(str(value), _clean_label(value).title())


def _tight_y(ax: plt.Axes, values: pd.Series | np.ndarray | list[float], *, lower_floor: float | None = None) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    span = float(arr.max() - arr.min())
    pad = max(span * 0.25, 0.02)
    lower = float(arr.min() - pad)
    upper = float(arr.max() + pad)
    if lower_floor is not None:
        lower = max(lower_floor, lower)
    ax.set_ylim(lower, upper)


def _clean_label(value: object) -> str:
    return str(value).replace("_", " ")


def _feature_label(value: object) -> str:
    labels = {
        "mean_relative_l1_pct": "residual\nmagnitude",
        "mutation_count": "mutation\ncount",
        "best_reconstruction_cosine": "best\ncosine",
        "mean_reconstruction_cosine": "mean\ncosine",
        "max_residual_structure_score": "max residual\nstructure",
        "mean_residual_structure_score": "mean residual\nstructure",
        "classifier_entropy": "classifier\nentropy",
    }
    return labels.get(str(value), _clean_label(value))


def _bar_labels(ax: plt.Axes, values: list[float], *, fmt: str = "{:.2f}") -> None:
    for patch, value in zip(ax.patches, values):
        if np.isnan(value):
            continue
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            xytext=(0, 2),
            textcoords="offset points",
        )


def _bar_labels_above_errors(
    ax: plt.Axes,
    values: list[float],
    errors: list[float] | np.ndarray | None = None,
    *,
    fmt: str = "{:.2f}",
) -> None:
    errors_array = np.zeros(len(values)) if errors is None else np.asarray(errors, dtype=float)
    y_min, y_max = ax.get_ylim()
    pad = max((y_max - y_min) * 0.025, 0.004)
    for patch, value, error in zip(ax.patches, values, errors_array):
        if np.isnan(value):
            continue
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2.0, value + (0 if np.isnan(error) else error) + pad),
            ha="center",
            va="bottom",
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 0),
        )


def _weighted_reliability_bins(bins: pd.DataFrame) -> pd.DataFrame:
    if bins.empty:
        return bins
    bins = bins.copy()
    for column in [
        "bin_index",
        "bin_lower",
        "bin_upper",
        "n_samples",
        "mean_predicted_probability",
        "observed_positive_fraction",
        "brier",
    ]:
        if column in bins.columns:
            bins[column] = pd.to_numeric(bins[column], errors="coerce")
    rows = []
    for bin_index, group in bins.groupby("bin_index", dropna=False):
        weights = group["n_samples"].fillna(1.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": float(group["bin_lower"].dropna().iloc[0]) if "bin_lower" in group else np.nan,
                "bin_upper": float(group["bin_upper"].dropna().iloc[0]) if "bin_upper" in group else np.nan,
                "mean_predicted_probability": float((group["mean_predicted_probability"] * weights).sum() / total),
                "observed_positive_fraction": float((group["observed_positive_fraction"] * weights).sum() / total),
                "n_samples": total,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("bin_index")


def figure1_system_overview(root: Path, output_dir: Path) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.axis("off")
    hero_blue = "#0F4D92"
    hero_fill = "#EEF4FA"
    line_color = "#4D4D4D"
    arm_green = "#2E9E44"
    arm_amber = "#DDAA33"
    arm_rose = "#BB5566"
    arm_teal = "#42949E"

    column_labels = [
        (0.10, "Inputs"),
        (0.32, "Assignment evidence"),
        (0.57, "Decision record"),
        (0.84, "Action queue"),
    ]
    for x_pos, label in column_labels:
        ax.text(
            x_pos,
            0.88,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=NEUTRAL["mid"],
            transform=ax.transAxes,
        )

    _add_box(
        ax,
        (0.03, 0.58),
        0.15,
        0.16,
        title="Mutation\nprofile",
        body="counts",
        facecolor="white",
        edgecolor="#AEB7BC",
        fontsize=7.5,
    )
    _add_box(
        ax,
        (0.03, 0.32),
        0.15,
        0.16,
        title="Reference\ncatalog",
        body="signatures",
        facecolor="white",
        edgecolor="#AEB7BC",
        fontsize=7.5,
    )

    _add_box(
        ax,
        (0.24, 0.63),
        0.17,
        0.105,
        title="Exposure",
        body="active set",
        facecolor="#F8F9FA",
        edgecolor="#9AA3A8",
        fontsize=7.4,
    )
    _add_box(
        ax,
        (0.24, 0.48),
        0.17,
        0.105,
        title="Fit quality",
        body="reconstruction",
        facecolor="#F8F9FA",
        edgecolor="#9AA3A8",
        fontsize=7.4,
    )
    _add_box(
        ax,
        (0.24, 0.33),
        0.17,
        0.105,
        title="Residual",
        body="unexplained signal",
        facecolor="#F8F9FA",
        edgecolor="#9AA3A8",
        fontsize=7.4,
    )

    _add_box(
        ax,
        (0.48, 0.55),
        0.22,
        0.20,
        title="SigAgent",
        body="support score\nsufficiency signal\ncalibration artifact",
        facecolor=hero_fill,
        edgecolor=hero_blue,
        title_color=hero_blue,
        linewidth=1.2,
        fontsize=7.9,
    )
    _add_box(
        ax,
        (0.48, 0.32),
        0.22,
        0.12,
        title="Provenance",
        body="traceable evidence",
        facecolor="white",
        edgecolor="#AEB7BC",
        title_color=NEUTRAL["dark"],
        fontsize=7.3,
    )

    action_boxes = [
        ("Use", "assignment", arm_green),
        ("Review", "sample", arm_amber),
        ("Reassess", "catalog", arm_rose),
        ("Discover", "cohort signal", arm_teal),
    ]
    y0 = 0.66
    for idx, (title, body, color) in enumerate(action_boxes):
        _add_strip_box(
            ax,
            (0.78, y0 - idx * 0.125),
            0.18,
            0.085,
            title=title,
            body=body,
            stripe_color=color,
            fontsize=7.2,
        )

    ax.plot([0.18, 0.205], [0.66, 0.66], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    ax.plot([0.18, 0.205], [0.40, 0.40], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    ax.plot([0.205, 0.205], [0.40, 0.66], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    _add_arrow(ax, (0.205, 0.53), (0.24, 0.53), color=line_color, linewidth=1.0)

    ax.plot([0.41, 0.435], [0.68, 0.68], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    ax.plot([0.41, 0.435], [0.53, 0.53], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    ax.plot([0.41, 0.435], [0.38, 0.38], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    ax.plot([0.435, 0.435], [0.38, 0.68], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    _add_arrow(ax, (0.435, 0.53), (0.48, 0.64), color=line_color, linewidth=1.0)

    ax.plot([0.59, 0.59], [0.55, 0.44], color="#7A858A", lw=0.9, transform=ax.transAxes, zorder=1)
    _add_arrow(ax, (0.70, 0.64), (0.74, 0.64), color=line_color, linewidth=1.0)
    ax.plot([0.74, 0.74], [0.33, 0.64], color=line_color, lw=1.0, transform=ax.transAxes, zorder=1)
    for y_center in [0.702, 0.577, 0.452, 0.327]:
        _add_arrow(ax, (0.74, y_center), (0.78, y_center), color=line_color, linewidth=0.9)

    png, pdf = _write_figure(fig, output_dir, "figure1_system_overview")
    return FigureRecord(
        figure_id="figure_1_system_overview",
        title="SigAgent routes signature assignments into downstream decisions",
        path_png=png,
        path_pdf=pdf,
        sources=[],
        note="Schematic generated from manuscript figure plan.",
    )


def figure3_complete_catalog_support(root: Path, output_dir: Path) -> FigureRecord:
    source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "known_catalog_by_burden_with_uncertainty.tsv",
        root / "paper_review_response_sbs96" / "tables" / "known_catalog_summary.tsv",
        root / "paper_known_catalog_smoke" / "metrics" / "aggregate_metrics.tsv",
    )
    overall_source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "known_catalog_overall_with_uncertainty.tsv",
        root / "paper_known_catalog_smoke" / "metrics" / "aggregate_metrics.tsv",
    )
    df = _read_tsv(source).copy()
    for column in df.columns:
        if column not in {"mutation_type", "expert_name"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    metric_map = {
        "sample_f1": ("sample_f1_mean" if "sample_f1_mean" in df.columns else "sample_f1", "sample_f1_sem"),
        "exposure_tvd": ("exposure_tvd_mean" if "exposure_tvd_mean" in df.columns else "exposure_tvd", "exposure_tvd_sem"),
        "reconstruction_cosine": (
            "reconstruction_cosine_mean" if "reconstruction_cosine_mean" in df.columns else "reconstruction_cosine",
            "reconstruction_cosine_sem",
        ),
    }
    experts = sorted(df["expert_name"].dropna().unique())
    expert_order = [expert for expert in ["plain_nnls", "rule_fusion"] if expert in set(experts)] or experts
    burdens = sorted(pd.to_numeric(df["burden"], errors="coerce").dropna().unique().tolist())
    burden_positions = {burden: index for index, burden in enumerate(burdens)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    metric_specs = [
        ("sample_f1", "Sample F1", axes[0, 0], "Higher is better"),
        ("exposure_tvd", "Exposure TVD", axes[0, 1], "Lower is better"),
        ("reconstruction_cosine", "Reconstruction cosine", axes[1, 0], "Preservation diagnostic"),
    ]
    for panel_idx, (metric, label, ax, subtitle) in enumerate(metric_specs):
        value_col, sem_col = metric_map[metric]
        all_values: list[float] = []
        for expert_index, expert in enumerate(expert_order):
            sub = df.loc[df["expert_name"].eq(expert)].sort_values("burden")
            color = EXPERT_COLORS.get(expert, OKABE_ITO[expert_index % len(OKABE_ITO)])
            marker = "o" if str(expert) == "rule_fusion" else "s"
            linestyle = "-" if str(expert) == "rule_fusion" else "--"
            linewidth = 1.45 if str(expert) == "rule_fusion" else 1.15
            if sem_col in sub.columns:
                ax.errorbar(
                    [burden_positions[burden] for burden in sub["burden"]],
                    sub[value_col],
                    yerr=sub[sem_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    markersize=4,
                    capsize=2,
                    color=color,
                    label=_format_method_label(expert),
                )
            else:
                ax.plot(
                    [burden_positions[burden] for burden in sub["burden"]],
                    sub[value_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    label=_format_method_label(expert),
                )
            all_values.extend(sub[value_col].dropna().tolist())
        ax.set_xticks(range(len(burdens)))
        ax.set_xticklabels([str(int(burden)) for burden in burdens])
        ax.set_xlim(-0.18, len(burdens) - 0.82)
        ax.set_xlabel("Mutation burden")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: {subtitle}")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        _tight_y(ax, all_values, lower_floor=0 if metric != "reconstruction_cosine" else None)
        _panel_label(ax, chr(ord("a") + panel_idx))
    axes[0, 0].legend(frameon=False, loc="upper left")

    ax_summary = axes[1, 1]
    overall = _read_tsv(overall_source).copy() if overall_source.exists() else pd.DataFrame()
    if not overall.empty and {"expert_name", "sample_f1_mean", "exposure_tvd_mean", "reconstruction_cosine_mean"}.issubset(overall.columns):
        for column in overall.columns:
            if column not in {"mutation_type", "expert_name"}:
                overall[column] = pd.to_numeric(overall[column], errors="coerce")
        by_expert = overall.set_index("expert_name")
        deltas = pd.Series(
            {
                "Sample F1": by_expert.loc["rule_fusion", "sample_f1_mean"] - by_expert.loc["plain_nnls", "sample_f1_mean"],
                "Exposure TVD": by_expert.loc["rule_fusion", "exposure_tvd_mean"] - by_expert.loc["plain_nnls", "exposure_tvd_mean"],
                "Cosine": by_expert.loc["rule_fusion", "reconstruction_cosine_mean"]
                - by_expert.loc["plain_nnls", "reconstruction_cosine_mean"],
            }
        )
        colors = [
            NATURE_COLORS["blue"]
            if (label != "Exposure TVD" and value >= 0) or (label == "Exposure TVD" and value <= 0)
            else NATURE_COLORS["amber"]
            for label, value in deltas.items()
        ]
        ax_summary.barh(range(len(deltas)), deltas.values, color=colors)
        ax_summary.axvline(0, color=NEUTRAL["mid"], linewidth=0.8)
        ax_summary.set_yticks(range(len(deltas)))
        ax_summary.set_yticklabels(deltas.index)
        ax_summary.set_xlabel("Rule fusion minus plain NNLS")
        ax_summary.set_title("Primary metric deltas")
        ax_summary.set_xlim(-0.022, 0.09)
        for y_pos, (value, color) in enumerate(zip(deltas.values, colors)):
            if abs(value) >= 0.012:
                ax_summary.text(
                    value / 2,
                    y_pos,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if color == NATURE_COLORS["blue"] else NEUTRAL["ink"],
                )
            else:
                label_x = value + 0.003 if value >= 0 else -0.002
                ax_summary.text(
                    label_x,
                    y_pos,
                    f"{value:+.3f}",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=6.7,
                    color=NEUTRAL["ink"],
                )
    else:
        value_col, _ = metric_map["sample_f1"]
        mean_values = df.groupby("expert_name", dropna=False)[value_col].mean().loc[expert_order]
        ax_summary.bar([_format_method_label(idx) for idx in mean_values.index], mean_values.values, color=[EXPERT_COLORS.get(idx, OKABE_ITO[0]) for idx in mean_values.index])
        ax_summary.set_ylabel("Mean sample F1")
        _bar_labels(ax_summary, mean_values.tolist())
    _panel_label(ax_summary, "d")
    png, pdf = _write_figure(fig, output_dir, "figure3_complete_catalog_support")
    return FigureRecord(
        "figure_3_complete_catalog_support",
        "Complete-catalog assignment remains a support check for the decision layer",
        png,
        pdf,
        [source] + ([overall_source] if overall_source.exists() else []),
    )


def figure2_catalog_insufficiency(root: Path, output_dir: Path) -> FigureRecord:
    overall_source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "catalog_insufficiency_overall_with_uncertainty.tsv",
        root / "paper_catalog_insufficiency_manifest_smoke" / "tables" / "catalog_insufficiency_overall_with_uncertainty.tsv",
    )
    source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "catalog_insufficiency_by_group.tsv",
        root / "paper_catalog_insufficiency_manifest_smoke" / "tables" / "catalog_insufficiency_by_group.tsv",
    )
    coef_source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "catalog_assessor_coefficients.tsv",
        root / "paper_catalog_insufficiency_manifest_smoke" / "tables" / "catalog_assessor_coefficients.tsv",
    )
    overall = _read_tsv(overall_source).copy() if overall_source.exists() else pd.DataFrame()
    df = _read_tsv(source).copy()
    for column in [
        "burden",
        "catalog_insufficiency_auroc",
        "catalog_insufficiency_auprc",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not overall.empty:
        for column in [
            "catalog_insufficiency_auroc_mean",
            "catalog_insufficiency_auroc_sem",
            "catalog_insufficiency_auprc_mean",
            "catalog_insufficiency_auprc_sem",
        ]:
            overall[column] = pd.to_numeric(overall[column], errors="coerce")

    fig = plt.figure(figsize=(7.2, 4.35))
    ax_design = fig.add_axes([0.075, 0.705, 0.91, 0.24])
    ax_overall = fig.add_axes([0.075, 0.125, 0.22, 0.49])
    ax_trend = fig.add_axes([0.385, 0.125, 0.28, 0.49])
    ax_coef = fig.add_axes([0.785, 0.125, 0.20, 0.49])

    hero_blue = "#0F4D92"
    hero_fill = "#EEF4FA"
    perturb_amber = "#DDAA33"
    control_grey = "#767676"
    line_color = "#4D4D4D"

    ax_design.axis("off")
    _panel_label(ax_design, "a", x=-0.024)
    ax_design.set_title("Controlled-removal benchmark design", loc="left", pad=1)
    _add_box(
        ax_design,
        (0.02, 0.34),
        0.15,
        0.36,
        title="Sample",
        body="known active set",
        facecolor="white",
        edgecolor="#AEB7BC",
        fontsize=7.2,
    )
    _add_box(
        ax_design,
        (0.285, 0.53),
        0.205,
        0.27,
        title="Active removed",
        body="positive",
        facecolor="#FFF9EA",
        edgecolor=perturb_amber,
        title_color=NEUTRAL["ink"],
        fontsize=7.0,
    )
    _add_box(
        ax_design,
        (0.285, 0.19),
        0.205,
        0.27,
        title="Inactive removed",
        body="control",
        facecolor="white",
        edgecolor=control_grey,
        title_color=NEUTRAL["ink"],
        fontsize=7.0,
    )
    _add_box(
        ax_design,
        (0.61, 0.34),
        0.205,
        0.36,
        title="Decision score",
        body="rank positives above controls",
        facecolor=hero_fill,
        edgecolor=hero_blue,
        title_color=hero_blue,
        fontsize=7.0,
    )
    _add_box(
        ax_design,
        (0.865, 0.34),
        0.12,
        0.36,
        title="AUROC\nAUPRC",
        body="five seeds",
        facecolor="white",
        edgecolor=hero_blue,
        title_color=hero_blue,
        fontsize=6.8,
    )
    _add_arrow(ax_design, (0.17, 0.54), (0.285, 0.67), color=perturb_amber, linewidth=1.0)
    _add_arrow(ax_design, (0.17, 0.50), (0.285, 0.32), color=control_grey, linewidth=1.0)
    _add_arrow(ax_design, (0.49, 0.67), (0.61, 0.56), color=line_color, linewidth=1.0)
    _add_arrow(ax_design, (0.49, 0.32), (0.61, 0.50), color=line_color, linewidth=1.0)
    _add_arrow(ax_design, (0.815, 0.52), (0.865, 0.52), color=line_color, linewidth=1.0)

    if overall.empty:
        overall = (
            df.groupby(["mutation_type", "expert_name"], dropna=False)
            .agg(
                catalog_insufficiency_auroc_mean=("catalog_insufficiency_auroc", "mean"),
                catalog_insufficiency_auroc_sem=("catalog_insufficiency_auroc", "sem"),
                catalog_insufficiency_auprc_mean=("catalog_insufficiency_auprc", "mean"),
                catalog_insufficiency_auprc_sem=("catalog_insufficiency_auprc", "sem"),
            )
            .reset_index()
        )
    metrics = [
        ("AUROC", "catalog_insufficiency_auroc_mean", "catalog_insufficiency_auroc_sem"),
        ("AUPRC", "catalog_insufficiency_auprc_mean", "catalog_insufficiency_auprc_sem"),
    ]
    expert_order = [expert for expert in ["plain_nnls", "rule_fusion"] if expert in set(overall["expert_name"])]
    if not expert_order:
        expert_order = sorted(overall["expert_name"].dropna().astype(str).unique())
    by_expert = overall.set_index("expert_name")
    y_base = np.arange(len(metrics))[::-1]
    observed_values: list[float] = []
    offsets = {"plain_nnls": -0.10, "rule_fusion": 0.10}
    for metric_index, (metric_name, value_col, sem_col) in enumerate(metrics):
        metric_values = {}
        metric_errors = {}
        for expert in expert_order:
            metric_values[expert] = float(by_expert.loc[expert, value_col])
            metric_errors[expert] = float(by_expert.loc[expert, sem_col])
            observed_values.append(metric_values[expert])
        if {"plain_nnls", "rule_fusion"}.issubset(metric_values):
            ax_overall.plot(
                [metric_values["plain_nnls"], metric_values["rule_fusion"]],
                [y_base[metric_index], y_base[metric_index]],
                color=NEUTRAL["grid"],
                linewidth=1.0,
                zorder=1,
            )
        for expert in expert_order:
            marker = "o" if str(expert) == "rule_fusion" else "s"
            y_pos = y_base[metric_index] + offsets.get(str(expert), 0.0)
            ax_overall.errorbar(
                metric_values[expert],
                y_pos,
                xerr=metric_errors[expert],
                fmt=marker,
                markersize=4.8,
                capsize=2.2,
                linewidth=0.95,
                color=EXPERT_COLORS.get(expert, OKABE_ITO[expert_order.index(expert) % len(OKABE_ITO)]),
                label=_format_method_label(expert) if metric_index == 0 else None,
                zorder=3,
            )
            ax_overall.text(
                metric_values[expert] + 0.004,
                y_pos,
                f"{metric_values[expert]:.3f}",
                va="center",
                ha="left",
                fontsize=6.4,
                color=EXPERT_COLORS.get(expert, NEUTRAL["ink"]),
            )
        if {"plain_nnls", "rule_fusion"}.issubset(metric_values):
            delta = metric_values["rule_fusion"] - metric_values["plain_nnls"]
            ax_overall.text(
                min(0.832, max(metric_values.values()) + 0.012),
                y_base[metric_index] - 0.27,
                f"delta {delta:+.3f}",
                va="center",
                ha="right",
                fontsize=6.2,
                color=NEUTRAL["mid"],
            )
    ax_overall.set_yticks(y_base)
    ax_overall.set_yticklabels([name for name, _, _ in metrics])
    ax_overall.set_ylim(-0.52, len(metrics) - 0.48)
    ax_overall.set_xlim(max(0.70, min(observed_values) - 0.035), min(0.84, max(observed_values) + 0.044))
    ax_overall.set_xlabel("Mean discrimination")
    ax_overall.set_title("Overall separability")
    ax_overall.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax_overall.legend(frameon=False, loc="upper left", fontsize=6.3)
    _panel_label(ax_overall, "b")

    rule = df.loc[df["expert_name"].astype(str).eq("rule_fusion")].copy()
    if rule.empty:
        rule = df.copy()
    group_order = (
        rule.groupby("removal_selection_groups")["catalog_insufficiency_auroc"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    burdens = sorted(pd.to_numeric(rule["burden"], errors="coerce").dropna().unique().tolist())
    heatmap = (
        rule.pivot_table(
            index="removal_selection_groups",
            columns="burden",
            values="catalog_insufficiency_auroc",
            aggfunc="mean",
        )
        .reindex(group_order)
        .reindex(columns=burdens)
    )
    ax_trend.imshow(heatmap.values, aspect="auto", cmap="cividis", vmin=0.55, vmax=1.0)
    ax_trend.set_xticks(range(len(burdens)))
    ax_trend.set_xticklabels([str(int(burden)) for burden in burdens])
    ax_trend.set_yticks(range(len(group_order)))
    ax_trend.set_yticklabels([_format_group_label(group) for group in group_order])
    ax_trend.set_xlabel("Mutation burden")
    ax_trend.set_title("AUROC by removal class")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            value = heatmap.iloc[row, col]
            if np.isfinite(value):
                ax_trend.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="white" if value < 0.76 else NEUTRAL["ink"],
                )
    ax_trend.tick_params(axis="y", labelsize=6.6, pad=2)
    ax_trend.tick_params(axis="x", labelsize=6.6)
    _panel_label(ax_trend, "c")

    if coef_source.exists():
        coef = _read_tsv(coef_source)
        value_col = (
            "standardized_log_odds_coefficient"
            if "standardized_log_odds_coefficient" in coef.columns
            else "coefficient"
            if "coefficient" in coef.columns
            else coef.select_dtypes("number").columns[-1]
        )
        name_col = "feature_name" if "feature_name" in coef.columns else coef.columns[0]
        coef = coef.loc[:, [name_col, value_col]].copy()
        coef[value_col] = pd.to_numeric(coef[value_col], errors="coerce")
        coef = coef.dropna()
        coef = coef.loc[coef[value_col].abs().gt(0.08)]
        coef = coef.sort_values(value_col, key=lambda values: values.abs(), ascending=False).head(4)
        coef = coef.iloc[::-1]
        colors = [NATURE_COLORS["blue"] if value >= 0 else NATURE_COLORS["grey"] for value in coef[value_col]]
        ax_coef.barh([_feature_label(name) for name in coef[name_col]], coef[value_col].values, color=colors)
        ax_coef.axvline(0, color="#777777", linewidth=0.8)
        ax_coef.set_title("Feature weights")
        ax_coef.set_xlabel("Std. log-odds coefficient")
        ax_coef.grid(axis="x", alpha=0.18, linewidth=0.6)
        ax_coef.tick_params(axis="y", labelsize=6.5, pad=2)
        ax_coef.tick_params(axis="x", labelsize=6.6)
        max_abs = float(coef[value_col].abs().max()) if not coef.empty else 1.0
        ax_coef.set_xlim(-max_abs * 0.24, max_abs * 1.23)
        for y_pos, value in enumerate(coef[value_col].values):
            label_x = value + 0.018 if value >= 0 else 0.018
            ax_coef.text(
                label_x,
                y_pos,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=6.6,
                color=NEUTRAL["ink"],
            )
    else:
        ax_coef.text(0.5, 0.5, "Coefficient table unavailable", ha="center", va="center")
        ax_coef.axis("off")
    _panel_label(ax_coef, "d")
    png, pdf = _write_figure(fig, output_dir, "figure2_catalog_insufficiency")
    sources = [overall_source, source] + ([coef_source] if coef_source.exists() else [])
    return FigureRecord(
        "figure_2_catalog_insufficiency",
        "Controlled catalog perturbations make insufficiency evidence measurable",
        png,
        pdf,
        sources,
    )


def figure4_calibration(root: Path, output_dir: Path) -> FigureRecord:
    summary_source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "calibration_overall_with_uncertainty.tsv",
        root / "paper_calibration_smoke" / "tables" / "reliability_summary.tsv",
    )
    bins_source = _first_existing(
        root / "paper_review_response_sbs96" / "tables" / "reliability_bins.tsv",
        root / "paper_calibration_smoke" / "tables" / "reliability_bins.tsv",
    )
    summary = _read_tsv(summary_source).copy()
    bins = _read_tsv(bins_source).copy() if bins_source.exists() else pd.DataFrame()
    for column in [
        "ece",
        "ece_mean",
        "ece_sem",
        "brier",
        "brier_mean",
        "brier_sem",
        "n_bins_nonempty_mean",
        "n_bins_nonempty_sem",
        "n_samples_mean",
        "mean_predicted_probability",
        "observed_positive_fraction",
    ]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
        if column in bins.columns:
            bins[column] = pd.to_numeric(bins[column], errors="coerce")

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.92,
        wspace=0.32,
        hspace=0.54,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
    )
    ax_ece = fig.add_subplot(gs[0, 0])
    ax_brier = fig.add_subplot(gs[0, 1])
    ax_reliability = fig.add_subplot(gs[1, 0])
    ax_support = fig.add_subplot(gs[1, 1])

    if "ece_mean" in summary.columns:
        overall = summary.loc[
            summary.get("task_name", pd.Series(dtype=str))
            .astype(str)
            .isin(["assignment_confidence", "catalog_insufficiency"])
        ].copy()
        if overall.empty:
            overall = summary.copy()
        task_order = [task for task in ["assignment_confidence", "catalog_insufficiency"] if task in set(overall["task_name"])]
        if task_order:
            overall = overall.set_index("task_name").loc[task_order].reset_index()
        labels = [_clean_label(row.task_name).replace(" ", "\n") for row in overall.itertuples()]
        ece_values = overall["ece_mean"].fillna(0).values
        ece_errors = overall["ece_sem"].fillna(0).values if "ece_sem" in overall.columns else None
        brier_values = overall["brier_mean"].fillna(0).values
        brier_errors = overall["brier_sem"].fillna(0).values if "brier_sem" in overall.columns else None
        task_names = overall["task_name"].astype(str).tolist()
    else:
        overall = summary.loc[summary["group_dimension"].astype(str).eq("overall")].copy()
        if overall.empty:
            overall = summary.copy()
        labels = [f"{_clean_label(row.group_dimension)}\n{_clean_label(row.group_value)}" for row in overall.itertuples()]
        ece_values = overall["ece"].fillna(0).values
        ece_errors = None
        brier_values = overall["brier"].fillna(0).values if "brier" in overall.columns else np.zeros_like(ece_values)
        brier_errors = None
        task_names = [str(label) for label in labels]

    def _task_summary(task_name: str) -> pd.Series:
        if "task_name" not in overall.columns:
            return overall.iloc[0]
        task_row = overall.loc[overall["task_name"].astype(str).eq(task_name)]
        return task_row.iloc[0] if not task_row.empty else overall.iloc[0]

    def _task_reliability_bins(task_name: str) -> pd.DataFrame:
        task_bins = bins.copy()
        if "task_name" in task_bins.columns:
            task_bins = task_bins.loc[
                task_bins["task_name"].astype(str).eq(task_name)
                & task_bins["group_dimension"].astype(str).eq("overall")
                & task_bins["group_value"].astype(str).eq("all")
            ].copy()
        if "bin_index" in task_bins.columns and not task_bins.empty:
            task_bins = _weighted_reliability_bins(task_bins)
        return task_bins

    def _summary_label(task_name: str) -> str:
        row = _task_summary(task_name)
        ece = float(row.get("ece_mean", row.get("ece", np.nan)))
        brier = float(row.get("brier_mean", row.get("brier", np.nan)))
        bins_count = float(row.get("n_bins_nonempty_mean", np.nan))
        pieces = []
        if np.isfinite(ece):
            pieces.append(f"ECE {ece:.3f}")
        if np.isfinite(brier):
            pieces.append(f"Brier {brier:.3f}")
        if np.isfinite(bins_count):
            pieces.append(f"{bins_count:.1f} bins")
        return "; ".join(pieces)

    def _plot_reliability_bins(ax: plt.Axes, task_name: str, color: str, title: str, panel: str) -> None:
        task_bins = _task_reliability_bins(task_name)
        if task_bins.empty:
            ax.text(0.5, 0.5, "Reliability bins unavailable", ha="center", va="center")
            ax.axis("off")
            _panel_label(ax, panel)
            return
        point_sizes = np.clip(pd.to_numeric(task_bins["n_samples"], errors="coerce").fillna(1).values, 5, None)
        point_sizes = 14 + np.sqrt(point_sizes) * 7
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        ax.scatter(
            task_bins["mean_predicted_probability"],
            task_bins["observed_positive_fraction"],
            s=point_sizes,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.74,
            clip_on=False,
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive fraction")
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            "Point area scales with bin support",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=NEUTRAL["mid"],
        )
        ax.text(
            0.04,
            0.08,
            _summary_label(task_name),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=NEUTRAL["mid"],
        )
        _panel_label(ax, panel)

    def _plot_calibration_gap(ax: plt.Axes, task_name: str, color: str, title: str, panel: str) -> None:
        task_bins = _task_reliability_bins(task_name)
        if task_bins.empty:
            ax.text(0.5, 0.5, "Reliability bins unavailable", ha="center", va="center")
            ax.axis("off")
            _panel_label(ax, panel)
            return
        gap = task_bins["observed_positive_fraction"] - task_bins["mean_predicted_probability"]
        x = task_bins["mean_predicted_probability"]
        weights = pd.to_numeric(task_bins["n_samples"], errors="coerce").fillna(1)
        bar_colors = [color if value >= 0 else NATURE_COLORS["grey"] for value in gap]
        ax.bar(x, gap, width=0.075, color=bar_colors, alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color=NEUTRAL["mid"], linewidth=0.8)
        ax.scatter(x, gap, s=12 + np.sqrt(weights.values) * 5, color=bar_colors, edgecolor="white", linewidth=0.35, zorder=3)
        ax.set_xlim(-0.03, 1.03)
        limit = max(0.12, float(np.nanmax(np.abs(gap))) * 1.22)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed minus predicted")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.20, linewidth=0.6)
        _panel_label(ax, panel)

    if not bins.empty:
        _plot_reliability_bins(
            ax_ece,
            "assignment_confidence",
            NATURE_COLORS["blue"],
            "Assignment-confidence reliability",
            "a",
        )
        _plot_reliability_bins(
            ax_brier,
            "catalog_insufficiency",
            NATURE_COLORS["amber"],
            "Catalog-insufficiency reliability",
            "b",
        )
        _plot_calibration_gap(
            ax_reliability,
            "assignment_confidence",
            NATURE_COLORS["blue"],
            "Assignment-confidence bin gaps",
            "c",
        )
        _plot_calibration_gap(
            ax_support,
            "catalog_insufficiency",
            NATURE_COLORS["amber"],
            "Catalog-insufficiency bin gaps",
            "d",
        )
    else:
        for panel, ax in zip("abcd", [ax_ece, ax_brier, ax_reliability, ax_support]):
            ax.text(0.5, 0.5, "Reliability bins unavailable", ha="center", va="center")
            ax.axis("off")
            _panel_label(ax, panel)
    png, pdf = _write_figure(fig, output_dir, "figure4_calibration")
    sources = [summary_source] + ([bins_source] if bins_source.exists() else [])
    return FigureRecord(
        "figure_4_calibration",
        "Calibration diagnostics make decision scores auditable",
        png,
        pdf,
        sources,
    )


def figure5_discovery(root: Path, output_dir: Path) -> FigureRecord:
    packet_source = root / "paper_discovery_smoke" / "tables" / "discovery_packet_summary.tsv"
    trigger_source = root / "paper_discovery_smoke" / "tables" / "discovery_trigger_summary.tsv"
    packet = _read_tsv(packet_source).copy()
    trigger = _read_tsv(trigger_source).copy() if trigger_source.exists() else pd.DataFrame()
    fig = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0])
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_cos = fig.add_subplot(gs[0, 1])
    ax_l1 = fig.add_subplot(gs[1, 1])
    ax_flow.axis("off")
    _panel_label(ax_flow, "a")
    ax_flow.set_title("Residual evidence is packaged for review", loc="left", pad=2)

    _add_box(
        ax_flow,
        (0.05, 0.62),
        0.24,
        0.14,
        title="Review-routed\nsample",
        body="residual evidence",
        facecolor="white",
        edgecolor="#AEB7BC",
        fontsize=7.4,
    )
    _add_box(
        ax_flow,
        (0.38, 0.62),
        0.24,
        0.14,
        title="Residual\ncomponent",
        body="candidate packet",
        facecolor=NATURE_COLORS["amber_soft"],
        edgecolor=ACCENT["warning"],
        title_color=NEUTRAL["ink"],
        fontsize=7.4,
    )
    _add_box(
        ax_flow,
        (0.71, 0.62),
        0.24,
        0.14,
        title="Review\ngate",
        body="no writeback",
        facecolor="white",
        edgecolor=ACCENT["review"],
        title_color=NEUTRAL["ink"],
        fontsize=7.4,
    )
    _add_arrow(ax_flow, (0.29, 0.69), (0.38, 0.69), color=NEUTRAL["dark"])
    _add_arrow(ax_flow, (0.62, 0.69), (0.71, 0.69), color=NEUTRAL["dark"])

    status_counts = {}
    if not trigger.empty and "trigger_status" in trigger.columns:
        status_counts = trigger.groupby("trigger_status", dropna=False)["n_candidates"].sum().to_dict()

    def _numeric_packet_column(column: str) -> pd.Series:
        if column not in packet.columns:
            return pd.Series([0.0] * len(packet), dtype=float)
        return pd.to_numeric(packet[column], errors="coerce")

    packet_count = int(_numeric_packet_column("n_candidate_records").fillna(0).sum())
    component_count = int(_numeric_packet_column("n_extracted_components").fillna(0).sum())
    recurrence_count = int(_numeric_packet_column("recurrence_count").fillna(0).sum())
    cards = [
        ("Packets", packet_count, ACCENT["sigagent"]),
        ("Components", component_count, ACCENT["warning"]),
        ("Recurrence", recurrence_count, ACCENT["support"]),
        ("Ready triggers", int(status_counts.get("ready", 0)), ACCENT["review"]),
    ]
    ax_flow.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.18),
            0.90,
            0.22,
            boxstyle=mpatches.BoxStyle("Round", pad=0.012, rounding_size=0.015),
            facecolor="#FAFBFC",
            edgecolor=NEUTRAL["grid"],
            linewidth=0.9,
            transform=ax_flow.transAxes,
        )
    )
    ax_flow.text(
        0.08,
        0.34,
        "Packet audit",
        fontsize=7.1,
        fontweight="bold",
        color=NEUTRAL["dark"],
        transform=ax_flow.transAxes,
    )
    for idx, (label, value, color) in enumerate(cards):
        x = 0.21 + idx * 0.18
        ax_flow.text(x, 0.30, f"{value}", fontsize=10.5, fontweight="bold", color=color, ha="center", transform=ax_flow.transAxes)
        ax_flow.text(x, 0.23, label, fontsize=6.4, color=NEUTRAL["mid"], ha="center", transform=ax_flow.transAxes)

    metrics = [
        "mean_delta_reconstruction_cosine_vs_current",
        "mean_delta_reconstruction_cosine_vs_known_only",
        "mean_delta_relative_l1_pct_vs_current",
        "mean_delta_relative_l1_pct_vs_known_only",
    ]
    values = [float(_numeric_packet_column(metric).dropna().mean()) for metric in metrics]
    cosine_values = values[:2]
    l1_values = values[2:]
    cosine_labels = ["vs current", "vs known only"]
    l1_labels = ["vs current", "vs known only"]
    ax_cos.barh(range(2), cosine_values, color=ACCENT["sigagent"])
    ax_cos.set_yticks(range(2))
    ax_cos.set_yticklabels(cosine_labels)
    ax_cos.set_xlabel("Cosine delta")
    ax_cos.set_title("Reconstruction improves")
    ax_cos.grid(axis="x", alpha=0.20, linewidth=0.6)
    ax_cos.set_xlim(0, max(cosine_values) * 1.35)
    for y_pos, value in enumerate(cosine_values):
        ax_cos.text(value + max(cosine_values) * 0.03, y_pos, f"{value:.3f}", va="center", fontsize=6.8)
    _panel_label(ax_cos, "b")

    ax_l1.barh(range(2), l1_values, color=ACCENT["support"])
    ax_l1.set_yticks(range(2))
    ax_l1.set_yticklabels(l1_labels)
    ax_l1.set_xlabel("Relative L1 improvement (%)")
    ax_l1.set_title("Residual magnitude decreases")
    ax_l1.grid(axis="x", alpha=0.20, linewidth=0.6)
    ax_l1.set_xlim(0, max(l1_values) * 1.18)
    for y_pos, value in enumerate(l1_values):
        ax_l1.text(value + max(l1_values) * 0.02, y_pos, f"{value:.1f}", va="center", fontsize=6.8)
    _panel_label(ax_l1, "c")
    png, pdf = _write_figure(fig, output_dir, "figure5_discovery_packet")
    sources = [packet_source] + ([trigger_source] if trigger_source.exists() else [])
    return FigureRecord(
        "figure_5_discovery_packet",
        "Review-gated packets package residual evidence without catalog writeback",
        png,
        pdf,
        sources,
    )


def figure6_real_data(root: Path, output_dir: Path) -> FigureRecord:
    summary_source = root / "paper_real_data_stress_smoke" / "tables" / "real_data_stress_design_summary.tsv"
    delta_source = root / "paper_real_data_stress_smoke" / "tables" / "real_data_catalog_stress_delta.tsv"
    summary = _read_tsv(summary_source).copy()
    delta = _read_tsv(delta_source).copy()
    for frame in [summary, delta]:
        for column in frame.columns:
            if column not in {"sample_id", "stress_step_name", "stress_design", "source_tumor_type", "primary_recommendation_full", "primary_recommendation_reduced", "catalog_insufficiency_level_full", "catalog_insufficiency_level_reduced", "top_signatures_full", "top_signatures_reduced", "full_step_name", "reduced_step_name"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    design_labels = {
        "data_driven_active_signature_removal": "Active-signature\nremoval",
        "fixed_sbs1_sbs5_removal": "Fixed SBS1/SBS5\nremoval",
    }
    design_colors = {
        "data_driven_active_signature_removal": ACCENT["warning"],
        "fixed_sbs1_sbs5_removal": ACCENT["baseline"],
    }
    summary = summary.sort_values("mean_catalog_insufficiency_probability_delta", ascending=False)
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 1.05])
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_waterfall = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    x = np.arange(len(summary))
    labels = [design_labels.get(str(value), _clean_label(value)) for value in summary["stress_design"]]
    mean_delta = summary["mean_catalog_insufficiency_probability_delta"].values
    max_delta = summary["max_catalog_insufficiency_probability_delta"].values
    colors = [design_colors.get(str(value), OKABE_ITO[index % len(OKABE_ITO)]) for index, value in enumerate(summary["stress_design"])]
    width = 0.34
    ax_summary.bar(x - width / 2, mean_delta, width=width, color=colors, alpha=0.55, edgecolor="white", linewidth=0.5, label="Mean")
    ax_summary.bar(x + width / 2, max_delta, width=width, color=colors, alpha=0.90, hatch="//", edgecolor=NEUTRAL["ink"], linewidth=0.5, label="Max")
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(labels)
    ax_summary.set_ylabel("Catalog-insufficiency\nprobability delta")
    ax_summary.set_title("Active removal produces stronger stress response")
    ax_summary.legend(frameon=False, loc="upper right")
    ax_summary.grid(axis="y", alpha=0.20, linewidth=0.6)
    for xi, value in zip(x + width / 2, max_delta):
        ax_summary.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8)
    _panel_label(ax_summary, "a")

    count_metrics = [
        ("n_catalog_insufficiency_delta_ge_0_10", "Delta >= 0.10", NATURE_COLORS["blue"]),
        ("n_primary_recommendation_changed", "Recommendation\nchanged", ACCENT["review"]),
    ]
    count_width = 0.26
    max_count = 0
    for metric_index, (metric, metric_label, color) in enumerate(count_metrics):
        values = summary[metric].values
        max_count = max(max_count, int(np.nanmax(values)))
        offset = (metric_index - (len(count_metrics) - 1) / 2) * count_width
        ax_counts.bar(x + offset, values, width=count_width * 0.86, color=color, edgecolor="white", linewidth=0.5, label=metric_label)
        for xi, value, denom in zip(x + offset, values, summary["n_samples"].values):
            ax_counts.text(xi, value + 0.10, f"{int(value)}/{int(denom)}", ha="center", va="bottom", fontsize=6.8)
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(labels)
    ax_counts.set_ylabel("Samples")
    ax_counts.set_title("Non-exclusive escalation counts")
    ax_counts.legend(frameon=False, loc="upper right")
    ax_counts.set_ylim(0, max(4, max_count + 0.9))
    _panel_label(ax_counts, "b")

    delta["catalog_insufficiency_probability_delta_reduced_minus_full"] = pd.to_numeric(
        delta["catalog_insufficiency_probability_delta_reduced_minus_full"], errors="coerce"
    )
    active = delta.loc[delta["stress_design"].astype(str).eq("data_driven_active_signature_removal")].copy()
    active = active.sort_values("catalog_insufficiency_probability_delta_reduced_minus_full", ascending=False)
    top = active.head(8).iloc[::-1]
    changed = top["primary_recommendation_full"].astype(str).ne(top["primary_recommendation_reduced"].astype(str))
    bar_colors = [ACCENT["review"] if flag else ACCENT["warning"] for flag in changed]
    sample_labels = [
        f"{str(row.source_tumor_type).split('-')[0]}\n{str(row.sample_id).split('::')[-1]}"
        for row in top.itertuples()
    ]
    ax_waterfall.barh(sample_labels, top["catalog_insufficiency_probability_delta_reduced_minus_full"], color=bar_colors, edgecolor="white", linewidth=0.4)
    ax_waterfall.axvline(0.10, color=NEUTRAL["mid"], linestyle="--", linewidth=0.8)
    ax_waterfall.set_xlabel("Probability delta")
    ax_waterfall.set_title("Largest sample-level active-removal responses")
    ax_waterfall.grid(axis="x", alpha=0.20, linewidth=0.6)
    ax_waterfall.text(
        0.105,
        0.98,
        "0.10 threshold",
        fontsize=6.5,
        color=NEUTRAL["mid"],
        ha="left",
        va="top",
        transform=ax_waterfall.get_xaxis_transform(),
    )
    _panel_label(ax_waterfall, "c")

    for idx, (design, sub) in enumerate(delta.groupby("stress_design", dropna=False)):
        color = design_colors.get(str(design), OKABE_ITO[idx % len(OKABE_ITO)])
        ax_scatter.scatter(
            sub["catalog_insufficiency_probability_delta_reduced_minus_full"],
            sub["residual_structure_score_delta_reduced_minus_full"],
            s=28,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
            label=design_labels.get(str(design), _clean_label(design)),
        )
    ax_scatter.axvline(0, color=NEUTRAL["grid"], linewidth=0.8)
    ax_scatter.axhline(0, color=NEUTRAL["grid"], linewidth=0.8)
    ax_scatter.set_xlabel("Insufficiency-probability delta")
    ax_scatter.set_ylabel("Residual-structure delta")
    ax_scatter.set_title("Probability and residual signals co-move")
    ax_scatter.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax_scatter.grid(alpha=0.18, linewidth=0.6)
    _panel_label(ax_scatter, "d")
    png, pdf = _write_figure(fig, output_dir, "figure6_real_data_stress")
    return FigureRecord(
        "figure_6_real_data_stress",
        "Public real-data stress testing shows stronger escalation after active-signature removal",
        png,
        pdf,
        [summary_source, delta_source],
    )


FIGURE_BUILDERS: list[Callable[[Path, Path], FigureRecord]] = [
    figure1_system_overview,
    figure2_catalog_insufficiency,
    figure3_complete_catalog_support,
    figure4_calibration,
    figure5_discovery,
    figure6_real_data,
]


def make_paper_figures(root: Path, output_dir: Path) -> pd.DataFrame:
    _apply_nature_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FigureRecord] = []
    for builder in FIGURE_BUILDERS:
        records.append(builder(root, output_dir))
    manifest = pd.DataFrame(
        [
            {
                "figure_id": record.figure_id,
                "title": record.title,
                "path_png": str(record.path_png),
                "path_pdf": str(record.path_pdf),
                "sources": ";".join(str(source) for source in record.sources),
                "status": record.status,
                "note": record.note,
            }
            for record in records
        ]
    )
    manifest.to_csv(output_dir / "figure_manifest.tsv", sep="\t", index=False)
    (output_dir / "README.md").write_text(
        "# Paper Figures\n\n"
        "Generated by `python experiments/make_paper_figures.py --root results/paper --output-dir paper/figures`.\n\n"
        "These are BMC-targeted manuscript figures generated from traceable paper-suite result tables. "
        "The figure package includes PNG previews, PDF line-art exports, and SVG exports with editable text. "
        "Figure 1 is a schematic-led workflow figure; Figures 2-4 form the main quantitative evidence chain; "
        "Figures 5-6 are supplementary boundary analyses.\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v2 manuscript figures from paper artifact tables.")
    parser.add_argument("--root", default="results/paper", help="Paper results root.")
    parser.add_argument("--output-dir", default="paper/figures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_paper_figures(Path(args.root), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
