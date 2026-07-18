from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import dose_category, regression_metrics
from .features import NUMERIC_FEATURES, PHARMACOGENOMIC_FEATURES

RESEARCH_WARNING = (
    "Research use only; this estimate is not prescribing guidance, a medical device, "
    "or a substitute for clinician-guided INR monitoring."
)
_METRIC_COLUMNS = ["n", "mae_mg_week", "rmse_mg_week", "r2", "pw20"]


def _finite_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.loc[
        predictions["prediction_status"].eq("ok")
        & np.isfinite(pd.to_numeric(predictions["y_true"], errors="coerce"))
        & np.isfinite(pd.to_numeric(predictions["y_pred"], errors="coerce"))
    ].copy()


def _metric_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows, columns=[*group_columns, *_METRIC_COLUMNS])


def _suppress(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["suppressed_n_lt_30"] = result["n"].lt(30)
    result.loc[result["suppressed_n_lt_30"], _METRIC_COLUMNS[1:]] = np.nan
    return result


def _availability(values: pd.Series) -> pd.Series:
    text = values.fillna("Unknown").astype(str).str.strip()
    return np.where(
        text.str.lower().isin({"", "unknown", "nan", "no call"}), "Missing", "Available"
    )


def _interval_rows(
    predictions: pd.DataFrame, scope: str, group_column: str | None = None
) -> list[dict[str, object]]:
    rows = []
    valid = predictions.loc[
        predictions["interval_lower"].notna() & predictions["interval_upper"].notna()
    ].copy()
    group_columns = ["procedure"] + ([group_column] if group_column is not None else [])
    for keys, group in valid.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = int(len(group))
        row = {
            "procedure": str(keys[0]),
            "scope": scope,
            "group": "All" if group_column is None else str(keys[1]),
            "n": n,
            "interval_coverage_90": float(
                (
                    (group["interval_lower"] <= group["y_true"])
                    & (group["y_true"] <= group["interval_upper"])
                ).mean()
            ),
            "interval_mean_width_mg_week": float(
                (group["interval_upper"] - group["interval_lower"]).mean()
            ),
            "suppressed_n_lt_30": n < 30,
        }
        if row["suppressed_n_lt_30"]:
            row["interval_coverage_90"] = np.nan
            row["interval_mean_width_mg_week"] = np.nan
        rows.append(row)
    return rows


def _interval_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    finite = _finite_predictions(predictions)
    rows = _interval_rows(finite, "overall") + _interval_rows(finite, "site", "site")
    audit = finite.assign(
        cyp2c9_availability=_availability(finite["cyp2c9_group"]),
        vkorc1_availability=_availability(finite["vkorc1"]),
    )
    for column in [
        "gender",
        "age_group",
        "race_audit",
        "cyp2c9_availability",
        "cyp2c9_group",
        "vkorc1_availability",
        "vkorc1",
        "dose_category",
    ]:
        rows.extend(_interval_rows(audit, f"subgroup:{column}", column))
    return pd.DataFrame(rows)


def _paired_differences(
    predictions: pd.DataFrame, paired_bootstrap: pd.DataFrame | None
) -> pd.DataFrame:
    rows = []
    finite = _finite_predictions(predictions)
    procedures = sorted(finite["procedure"].unique())
    for index, procedure_a in enumerate(procedures):
        left = finite.loc[
            finite["procedure"].eq(procedure_a), ["row_key", "site", "y_true", "y_pred"]
        ]
        for procedure_b in procedures[index + 1 :]:
            right = finite.loc[
                finite["procedure"].eq(procedure_b), ["row_key", "site", "y_true", "y_pred"]
            ]
            paired = left.merge(
                right, on=["row_key", "site"], suffixes=("_a", "_b"), validate="one_to_one"
            )
            if paired.empty or not np.array_equal(paired["y_true_a"], paired["y_true_b"]):
                continue
            metrics_a = regression_metrics(paired["y_true_a"], paired["y_pred_a"])
            metrics_b = regression_metrics(paired["y_true_b"], paired["y_pred_b"])
            row: dict[str, object] = {
                "procedure_a": procedure_a,
                "procedure_b": procedure_b,
                "n_shared_finite": metrics_a["n"],
            }
            for metric in _METRIC_COLUMNS[1:]:
                name = f"{metric}_difference"
                row[name] = metrics_a[metric] - metrics_b[metric]
                row[f"{name}_ci95_lower"] = np.nan
                row[f"{name}_ci95_upper"] = np.nan
                if paired_bootstrap is not None and {"procedure_a", "procedure_b", name} <= set(
                    paired_bootstrap
                ):
                    values = paired_bootstrap.loc[
                        paired_bootstrap["procedure_a"].eq(procedure_a)
                        & paired_bootstrap["procedure_b"].eq(procedure_b),
                        name,
                    ].dropna()
                    if not values.empty:
                        row[f"{name}_ci95_lower"], row[f"{name}_ci95_upper"] = np.quantile(
                            values, [0.025, 0.975]
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_figure(path: Path, draw) -> None:
    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=(7, 5))
        draw(axis)
        figure.tight_layout()
        figure.savefig(path, dpi=300)
        plt.close(figure)


def _report_figures(predictions: pd.DataFrame, figures: Path, ranks: pd.DataFrame | None) -> None:
    finite = _finite_predictions(predictions)

    def observed(axis):
        procedures = ["clinical_ml", "pharmacogenomic_ml", "iwpc_pharmacogenetic"]
        paired = (
            finite.loc[finite["procedure"].isin(procedures)]
            .pivot(index="row_key", columns="procedure", values=["y_true", "y_pred"])
            .dropna()
        )
        required = {
            (value, procedure)
            for value in ["y_true", "y_pred"]
            for procedure in procedures
        }
        if paired.empty or not required.issubset(paired.columns):
            axis.text(0.5, 0.5, "No finite paired predictions", ha="center", va="center")
        else:
            values = paired["y_true"].to_numpy().ravel()
            lower, upper = float(values.min()), float(values.max())
            for procedure in procedures:
                axis.scatter(
                    paired[("y_true", procedure)],
                    paired[("y_pred", procedure)],
                    s=12,
                    label=procedure,
                )
            axis.plot([lower, upper], [lower, upper], "k--", label="identity")
            axis.legend(fontsize=8)
        axis.set(xlabel="Observed weekly dose (mg/week)", ylabel="Predicted weekly dose (mg/week)")

    def site_mae(axis):
        values = finite.loc[
            finite["procedure"].isin(["clinical_ml", "pharmacogenomic_ml", "iwpc_pharmacogenetic"])
        ]
        values = values.assign(absolute_error=(values["y_true"] - values["y_pred"]).abs())
        summary = (
            values.groupby(["procedure", "site"], sort=True)["absolute_error"]
            .agg(["mean", "size"])
            .reset_index()
        )
        summary = summary.loc[summary["size"].ge(30)]
        if summary.empty:
            axis.text(0.5, 0.5, "Site metrics suppressed for n < 30", ha="center", va="center")
        else:
            summary.pivot(index="site", columns="procedure", values="mean").plot.barh(ax=axis)
            axis.legend(fontsize=8)
        axis.set(xlabel="MAE (mg/week)", ylabel="Site")

    def stability(axis):
        if ranks is None or ranks.empty:
            axis.text(0.5, 0.5, "No saved feature ranking artifact", ha="center", va="center")
        else:
            values = ranks.sort_values(["median_rank", "feature_block"])
            axis.errorbar(
                values["median_rank"], values["feature_block"], xerr=values["rank_std"], fmt="o"
            )
        axis.set(xlabel="Median feature rank (± SD)", ylabel="Feature block")

    def coverage(axis):
        values = finite.loc[
            finite["procedure"].isin(["clinical_ml", "pharmacogenomic_ml"])
            & finite["interval_lower"].notna()
            & finite["interval_upper"].notna()
        ].copy()
        values["covered"] = (values["interval_lower"] <= values["y_true"]) & (
            values["y_true"] <= values["interval_upper"]
        )
        summary = (
            values.groupby(["procedure", "site"], sort=True)["covered"]
            .agg(["mean", "size"])
            .reset_index()
        )
        summary = summary.loc[summary["size"].ge(30)]
        if summary.empty:
            axis.text(0.5, 0.5, "Site metrics suppressed for n < 30", ha="center", va="center")
        else:
            summary.pivot(index="site", columns="procedure", values="mean").plot.bar(ax=axis)
            axis.legend(fontsize=8)
        axis.axhline(0.90, color="black", linestyle="--", label="nominal 90%")
        axis.set(xlabel="Site", ylabel="Empirical 90% interval coverage")

    _write_figure(figures / "observed_vs_predicted.png", observed)
    _write_figure(figures / "mae_by_site.png", site_mae)
    _write_figure(figures / "feature_rank_stability.png", stability)
    _write_figure(figures / "interval_coverage_by_site.png", coverage)


def _saved_ranks(run_dir: Path) -> pd.DataFrame | None:
    candidates = [
        run_dir / "feature_rankings.csv",
        run_dir.parent / "feature-selection" / "feature_rankings.csv",
        run_dir.parent / "ablation" / "feature_rankings.csv",
    ]
    for path in candidates:
        if path.exists():
            ranks = pd.read_csv(path)
            required = {"feature_block", "median_rank", "rank_std"}
            if required <= set(ranks):
                return ranks.loc[:, ["feature_block", "median_rank", "rank_std"]].copy()
    return None


def _copy_secondary_tables(run_dir: Path, tables: Path) -> None:
    ablation = run_dir.parent / "ablation" / "metrics.csv"
    if ablation.exists():
        pd.read_csv(ablation).to_csv(tables / "ablation_metrics.csv", index=False)
    rows = []
    for analysis in ["complete-case", "random-cv"]:
        path = run_dir.parent / analysis / "metrics.csv"
        if path.exists():
            rows.append(pd.read_csv(path).assign(analysis=analysis))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(tables / "sensitivity_metrics.csv", index=False)


def _manuscript(manifest: dict[str, object]) -> str:
    metric_links = (
        "[overall metrics](tables/overall_metrics.csv), [site metrics](tables/site_metrics.csv)"
    )
    return f"""# Site-Aware Warfarin Dose Prediction from Public IWPC Data

## Research question
Can pre-treatment clinical and pharmacogenomic information estimate stable weekly warfarin dose
under site-held-out evaluation? This report describes saved research artifacts only; it is not a
dose recommendation.

## Public data and cohort
The saved run records {manifest.get("cohort_rows", "unknown")} eligible rows across
{manifest.get("site_count", "unknown")} sites from the reviewed public IWPC source
(SHA-256: `{manifest.get("source_sha256", "unknown")}`).

## Pre-treatment clinical and pharmacogenomic features
Learned inputs exclude dose, post-treatment INR, identifiers, site, and race. Missing expected
inputs are handled by the fitted preprocessing pipeline.

## Leakage-safe validation and model selection
Primary results use leave-one-site-out outer validation with training-site-only model selection
and conformal calibration. Selected models are recorded in [selections.csv](../selections.csv).

## Primary site-held-out performance
Saved overall performance is available in {metric_links}. All doses and errors are mg/week.

## Comparison with fixed and published IWPC algorithms
The fixed 35 mg/week comparator is a population reference, not an individual recommendation.
Paired saved-prediction comparisons are in [paired differences](tables/paired_differences.csv).

## Prediction uncertainty
Conformal interval coverage is empirical rather than guaranteed under hospital shift. See
[interval metrics](tables/interval_metrics.csv).

## Feature stability, ablations, and sensitivity analyses
Feature ranking, ablation, and sensitivity tables are included only when corresponding saved
analysis artifacts exist. Permutation importance is associational and correlation-sensitive.

## Subgroup and site audit
Subgroup, site, and dose-category metrics are suppressed when n < 30. Race is an audit field,
not a learned input.

## Limitations
Performance under site shift may differ. Rare genotypes, high doses, missingness, and
source-specific stable-dose definitions can limit transportability. No result is a dose
recommendation.

## Reproducibility
This report was generated from saved CSV/JSON artifacts; it neither fits models nor recomputes
predictions. The run manifest is [manifest.json](../manifest.json).

## Research-use warning
{RESEARCH_WARNING}
"""


def build_report(run_dir: Path) -> Path:
    """Build deterministic research-report artifacts from a saved run only."""
    run_dir = Path(run_dir)
    if not (run_dir / "predictions.csv").exists() and (run_dir / "primary").is_dir():
        run_dir = run_dir / "primary"
    predictions = pd.read_csv(run_dir / "predictions.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report_dir, tables, figures = (
        run_dir / "report",
        run_dir / "report" / "tables",
        run_dir / "report" / "figures",
    )
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    finite = _finite_predictions(predictions).assign(
        dose_category=dose_category(
            predictions.loc[_finite_predictions(predictions).index, "y_true"]
        )
    )
    overall = _metric_table(finite, ["procedure"])
    site = _suppress(_metric_table(finite, ["procedure", "site"]))
    category = _suppress(_metric_table(finite, ["procedure", "dose_category"]))
    audit = finite.assign(
        cyp2c9_availability=_availability(finite["cyp2c9_group"]),
        vkorc1_availability=_availability(finite["vkorc1"]),
    )
    subgroup = pd.concat(
        [
            _suppress(_metric_table(audit, ["procedure", column]))
            .assign(subgroup_type=column)
            .rename(columns={column: "subgroup"})
            for column in [
                "gender",
                "age_group",
                "race_audit",
                "cyp2c9_availability",
                "cyp2c9_group",
                "vkorc1_availability",
                "vkorc1",
                "dose_category",
            ]
        ],
        ignore_index=True,
    )
    paired_bootstrap_path = run_dir / "paired_bootstrap.csv"
    paired_bootstrap = (
        pd.read_csv(paired_bootstrap_path) if paired_bootstrap_path.exists() else None
    )
    overall.to_csv(tables / "overall_metrics.csv", index=False)
    site.to_csv(tables / "site_metrics.csv", index=False)
    category.to_csv(tables / "dose_category_metrics.csv", index=False)
    subgroup.to_csv(tables / "subgroup_metrics.csv", index=False)
    _interval_metrics(audit).to_csv(tables / "interval_metrics.csv", index=False)
    _paired_differences(finite, paired_bootstrap).to_csv(
        tables / "paired_differences.csv", index=False
    )
    ranks = _saved_ranks(run_dir)
    if ranks is not None:
        ranks.to_csv(tables / "feature_stability.csv", index=False)
    _copy_secondary_tables(run_dir, tables)
    _report_figures(finite, figures, ranks)
    report = report_dir / "report.md"
    report.write_text(_manuscript(manifest), encoding="utf-8")
    return report


def predict_patient(model_path: Path, input_path: Path) -> dict[str, object]:
    artifact = joblib.load(model_path)
    patient = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(patient, dict):
        raise ValueError("inference input must be a JSON object")
    allowed = set(PHARMACOGENOMIC_FEATURES) | {"statin"}
    forbidden = {"weekly_dose_mg", "site", "row_key", "race"}
    unknown = sorted(set(patient) - allowed - forbidden)
    supplied_forbidden = sorted(set(patient) & forbidden)
    if unknown or supplied_forbidden:
        raise ValueError(
            f"incompatible inference schema; unknown={unknown}, forbidden={supplied_forbidden}"
        )
    for name in set(patient) & set(NUMERIC_FEATURES):
        if patient[name] is not None:
            try:
                value = float(patient[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric input: {name}") from error
            if not np.isfinite(value):
                raise ValueError(f"nonfinite numeric input: {name}")
            patient[name] = value
    for name in set(patient) - set(NUMERIC_FEATURES):
        if patient[name] is not None:
            patient[name] = str(patient[name])
    expected = artifact["feature_columns"]
    missing = [name for name in expected if name not in patient or patient[name] is None]
    row = pd.DataFrame([{name: patient.get(name, np.nan) for name in expected}])
    prediction = max(0.0, float(artifact["pipeline"].predict(row)[0]))
    radius = float(artifact["conformal_radius"])
    lower, upper = max(0.0, prediction - radius), prediction + radius
    numeric_flags = [
        name
        for name, limits in artifact["numeric_training_ranges"].items()
        if patient.get(name) is not None and not limits[0] <= patient[name] <= limits[1]
    ]
    target_limits = artifact["target_training_range"]
    return {
        "weekly_dose_mg": prediction,
        "average_daily_dose_mg": prediction / 7,
        "interval_90_mg_week": [lower, upper],
        "missing_inputs": missing,
        "extrapolation_flags": {
            "numeric_inputs_outside_training_range": numeric_flags,
            "prediction_outside_training_target_range": not target_limits[0]
            <= prediction
            <= target_limits[1],
        },
        "model_version": artifact["git_revision"],
        "source_sha256": artifact["source_sha256"],
        "warning": RESEARCH_WARNING,
    }
