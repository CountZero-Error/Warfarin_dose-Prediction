from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .data import prepare_cohort, read_raw, sha256_file
from .features import (
    NUMERIC_FEATURES,
    build_feature_frame,
    feature_columns,
    select_feature_matrix,
    statin_gate,
)
from .models import (
    ModelSpec,
    iwpc_clinical,
    iwpc_pharmacogenetic,
    make_model_pipeline,
    model_candidates,
)

DEFAULT_SEED = 20260717
BOOTSTRAP_ITERATIONS = 2_000


def _validate_split(
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    patient_keys: np.ndarray | None = None,
) -> None:
    if set(train) & set(test):
        raise ValueError("overlapping train/test row positions")
    if set(groups[train]) & set(groups[test]):
        raise ValueError("overlapping train/test sites")
    if patient_keys is not None and set(patient_keys[train]) & set(patient_keys[test]):
        raise ValueError("overlapping train/test patients")


def site_outer_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["site"].astype(str).to_numpy()
    patient_keys = frame["patient_key"].astype(str).to_numpy()
    splits = list(LeaveOneGroupOut().split(frame, groups=groups))
    coverage = np.zeros(len(frame), dtype=int)
    for train, test in splits:
        _validate_split(train, test, groups, patient_keys)
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ValueError("every eligible row must have exactly one outer site fold")
    return splits


def inner_site_splits(sites: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(sites, dtype=str)
    n_sites = len(np.unique(groups))
    if n_sites < 3:
        raise ValueError("inner grouped validation requires at least three training sites")
    splits = list(GroupKFold(n_splits=min(5, n_sites)).split(np.zeros(len(groups)), groups=groups))
    for train, validation in splits:
        _validate_split(train, validation, groups)
    return splits


def dose_category(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values <= 21, "low", np.where(values >= 49, "high", "intermediate"))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if (
        truth.shape != prediction.shape
        or not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("metrics require same-shaped finite truth and predictions")
    if len(truth) == 0:
        raise ValueError("metrics require at least one prediction")
    return {
        "n": int(len(truth)),
        "mae_mg_week": float(mean_absolute_error(truth, prediction)),
        "rmse_mg_week": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "pw20": float(np.mean(np.abs(prediction - truth) <= 0.20 * truth)),
    }


def conformal_quantile(residuals: Sequence[float], coverage: float = 0.90) -> float:
    values = np.asarray(residuals, dtype=float)
    if (
        not 0 < coverage < 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("conformal residuals and coverage must be finite and valid")
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_interval(prediction: Sequence[float], radius: float) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(radius) or radius < 0:
        raise ValueError("nonfinite prediction or conformal radius")
    return np.clip(prediction - radius, 0.0, None), prediction + radius


def _validate_iterations(iterations: int) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("bootstrap iterations must be a positive integer")


def _validate_bootstrap_predictions(predictions: pd.DataFrame, *, paired: bool = False) -> None:
    required = {"site", "y_true", "y_pred"}
    if paired:
        required.add("row_key")
    if not isinstance(predictions, pd.DataFrame) or required - set(predictions):
        missing = (
            sorted(required - set(predictions))
            if isinstance(predictions, pd.DataFrame)
            else sorted(required)
        )
        raise ValueError(f"bootstrap missing columns: {missing}")
    if predictions.empty:
        raise ValueError("bootstrap requires at least one prediction")
    if predictions["site"].isna().any() or (paired and predictions["row_key"].isna().any()):
        raise ValueError("bootstrap pairing keys must be present")
    regression_metrics(predictions["y_true"], predictions["y_pred"])


def _bootstrap_sites(predictions: pd.DataFrame) -> np.ndarray:
    sites = predictions["site"].drop_duplicates().to_numpy()
    if len(sites) < 2:
        raise ValueError("site-cluster bootstrap requires at least two sites")
    return sites


def cluster_bootstrap(
    predictions: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    _validate_iterations(iterations)
    _validate_bootstrap_predictions(predictions)
    sites = _bootstrap_sites(predictions)
    site_rows = {site: group for site, group in predictions.groupby("site", sort=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        chosen = rng.choice(sites, size=len(sites), replace=True)
        sampled = pd.concat([site_rows[site] for site in chosen], ignore_index=True)
        metrics = regression_metrics(sampled["y_true"], sampled["y_pred"])
        rows.append({"iteration": iteration, **metrics})
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    _validate_iterations(iterations)
    _validate_bootstrap_predictions(predictions_a, paired=True)
    _validate_bootstrap_predictions(predictions_b, paired=True)
    pairing_keys = ["row_key", "site"]
    if predictions_a.duplicated(pairing_keys).any() or predictions_b.duplicated(pairing_keys).any():
        raise ValueError("paired bootstrap contains duplicate row_key/site pairs")
    paired = predictions_a.merge(
        predictions_b,
        on=pairing_keys,
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("paired bootstrap has no matched row_key/site pairs")
    if not np.array_equal(paired["y_true_a"].to_numpy(), paired["y_true_b"].to_numpy()):
        raise ValueError("paired bootstrap requires matching outcomes")
    sites = _bootstrap_sites(paired)
    site_rows = {site: group for site, group in paired.groupby("site", sort=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        chosen = rng.choice(sites, size=len(sites), replace=True)
        sampled = pd.concat([site_rows[site] for site in chosen], ignore_index=True)
        metrics_a = regression_metrics(sampled["y_true_a"], sampled["y_pred_a"])
        metrics_b = regression_metrics(sampled["y_true_b"], sampled["y_pred_b"])
        differences = {
            f"{name}_difference": value - metrics_b[name]
            for name, value in metrics_a.items()
            if name != "n"
        }
        rows.append({"iteration": iteration, "n": metrics_a["n"], **differences})
    return pd.DataFrame(rows)


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    if len(finite) == 0 or not np.isfinite(finite).all():
        raise ValueError("bootstrap interval contains nonfinite values")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


_FAILURE_COLUMNS = [
    "stage",
    "candidate_key",
    "fold",
    "error_type",
    "message",
    "procedure",
    "outer_fold",
    "outer_site",
]
_AUDIT_COLUMNS = ["gender", "age_group", "race_audit", "cyp2c9_group", "vkorc1", "dose_category"]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def score_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    candidates: Sequence[ModelSpec],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    splits = inner_site_splits(sites)
    for spec in candidates:
        for fold, (train, validation) in enumerate(splits):
            try:
                pipeline = make_model_pipeline(columns, spec, seed + fold)
                pipeline.fit(X.iloc[train], y[train])
                prediction = pipeline.predict(X.iloc[validation])
                score_rows.append(
                    {
                        "candidate_key": spec.key,
                        "fold": fold,
                        "mae_mg_week": float(mean_absolute_error(y[validation], prediction)),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "stage": "inner_fit",
                        "candidate_key": spec.key,
                        "fold": fold,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
    scores = pd.DataFrame(score_rows, columns=["candidate_key", "fold", "mae_mg_week"])
    complete = (
        scores.groupby("candidate_key")["fold"].nunique().eq(len(splits))
        if not scores.empty
        else pd.Series()
    )
    scores = scores[scores["candidate_key"].isin(complete[complete].index)].reset_index(drop=True)
    if scores.empty:
        raise RuntimeError(f"no successful candidate model; failures={failures}")
    return scores, pd.DataFrame(failures, columns=_FAILURE_COLUMNS[:5])


def select_one_se(scores: pd.DataFrame, candidates: Sequence[ModelSpec]) -> ModelSpec:
    required = {"candidate_key", "fold", "mae_mg_week"}
    if required - set(scores) or scores.empty:
        raise RuntimeError("one-standard-error selection requires successful inner scores")
    expected_fold_count = int(scores["fold"].nunique())
    complete = scores.groupby("candidate_key")["fold"].nunique().eq(expected_fold_count)
    summary = scores[scores["candidate_key"].isin(complete[complete].index)].groupby(
        "candidate_key"
    )["mae_mg_week"].agg(["mean", "std", "count"])
    if summary.empty:
        raise RuntimeError("one-standard-error selection found no successful candidate")
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    best_key = summary["mean"].idxmin()
    threshold = float(summary.loc[best_key, "mean"] + summary.loc[best_key, "se"])
    eligible = set(summary.index[summary["mean"] <= threshold])
    successful = [spec for spec in candidates if spec.key in eligible]
    if not successful:
        raise RuntimeError("one-standard-error selection found no successful candidate")
    return min(
        successful,
        key=lambda spec: (
            spec.family_order,
            0 if spec.target_mode == "direct" else 1,
            spec.complexity_order,
        ),
    )


def calibration_residuals(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    spec: ModelSpec,
    seed: int,
) -> np.ndarray:
    residuals = np.full(len(X), np.nan)
    for fold, (train, validation) in enumerate(inner_site_splits(sites)):
        pipeline = make_model_pipeline(columns, spec, seed + fold)
        pipeline.fit(X.iloc[train], y[train])
        residuals[validation] = np.abs(y[validation] - pipeline.predict(X.iloc[validation]))
    if not np.isfinite(residuals).all():
        raise ValueError("inner grouped conformal calibration did not cover every training row")
    return residuals


def _age_group(value: object) -> str:
    if not np.isfinite(value):
        return "Unknown"
    return "<50" if value < 5 else "50-69" if value < 7 else "70+"


def _prediction_row(
    frame: pd.DataFrame,
    position: int,
    procedure: str,
    outer_fold: int,
    prediction: float,
    *,
    lower: float = np.nan,
    upper: float = np.nan,
    status: str = "ok",
) -> dict[str, object]:
    row = frame.iloc[position]
    return {
        "row_key": row["row_key"],
        "site": row["site"],
        "outer_site": row["site"],
        "outer_fold": outer_fold,
        "procedure": procedure,
        "y_true": float(row["weekly_dose_mg"]),
        "y_pred": prediction,
        "interval_lower": lower,
        "interval_upper": upper,
        "prediction_status": status,
        "gender": row["gender"],
        "age_group": _age_group(row["age_decade"]),
        "race_audit": row["race"],
        "cyp2c9_group": row["cyp2c9_group"],
        "vkorc1": row["vkorc1"],
        "dose_category": dose_category([row["weekly_dose_mg"]])[0],
    }


def _run_learned_procedure(
    frame: pd.DataFrame,
    procedure: str,
    columns: Sequence[str],
    candidates: Sequence[ModelSpec],
    seed: int,
    statin_included_by_fold: Sequence[bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame["weekly_dose_mg"].to_numpy(float)
    predictions: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    outer_splits = site_outer_splits(frame)
    if statin_included_by_fold is not None and len(statin_included_by_fold) != len(outer_splits):
        raise ValueError("statin decisions must match outer folds")
    for outer_fold, (train, test) in enumerate(outer_splits):
        fold_columns = list(columns)
        if statin_included_by_fold is not None:
            fold_columns = [name for name in fold_columns if name != "statin"]
            if statin_included_by_fold[outer_fold]:
                fold_columns.append("statin")
        X = select_feature_matrix(frame, fold_columns)
        training_sites = frame.iloc[train]["site"].astype(str).to_numpy()
        scores, fold_failures = score_candidates(
            X.iloc[train].reset_index(drop=True),
            y[train],
            training_sites,
            fold_columns,
            candidates,
            seed + outer_fold * 100,
        )
        selected = select_one_se(scores, candidates)
        residuals = calibration_residuals(
            X.iloc[train].reset_index(drop=True),
            y[train],
            training_sites,
            fold_columns,
            selected,
            seed + outer_fold * 100,
        )
        radius = conformal_quantile(residuals, coverage=0.90)
        pipeline = make_model_pipeline(fold_columns, selected, seed + outer_fold)
        pipeline.fit(X.iloc[train], y[train])
        predicted = pipeline.predict(X.iloc[test])
        lower, upper = conformal_interval(predicted, radius)
        target_min, target_max = float(y[train].min()), float(y[train].max())
        for position, prediction, low, high in zip(test, predicted, lower, upper, strict=True):
            item = _prediction_row(
                frame,
                position,
                procedure,
                outer_fold,
                float(prediction),
                lower=float(low),
                upper=float(high),
            )
            item.update(
                {
                    "extrapolated_target": bool(prediction < target_min or prediction > target_max),
                    "model_family": selected.family,
                    "target_mode": selected.target_mode,
                    "candidate_key": selected.key,
                }
            )
            predictions.append(item)
        outer_site = str(frame.iloc[test[0]]["site"])
        selections.append(
            {
                "procedure": procedure,
                "outer_fold": outer_fold,
                "outer_site": outer_site,
                "candidate_key": selected.key,
                "conformal_radius": radius,
                "statin_included": "statin" in fold_columns,
            }
        )
        if not fold_failures.empty:
            failures.extend(
                fold_failures.assign(
                    procedure=procedure, outer_fold=outer_fold, outer_site=outer_site
                ).to_dict("records")
            )
    result = pd.DataFrame(predictions)
    counts = result.groupby("row_key").size()
    if len(result) != len(frame) or not counts.eq(1).all():
        raise ValueError(f"{procedure} did not produce exactly one outer prediction per patient")
    return result, pd.DataFrame(selections), pd.DataFrame(failures, columns=_FAILURE_COLUMNS)


def _run_comparators(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["weekly_dose_mg"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for outer_fold, (train, test) in enumerate(site_outer_splits(frame)):
        predictions = {
            "fixed_35_mg_week": np.full(len(test), 35.0),
            "training_mean": np.full(len(test), float(y[train].mean())),
            "training_median": np.full(len(test), float(np.median(y[train]))),
            "iwpc_clinical": iwpc_clinical(frame.iloc[test]),
            "iwpc_pharmacogenetic": iwpc_pharmacogenetic(frame.iloc[test]),
        }
        for procedure, values in predictions.items():
            for position, prediction in zip(test, values, strict=True):
                finite = bool(np.isfinite(prediction))
                item = _prediction_row(
                    frame,
                    position,
                    procedure,
                    outer_fold,
                    float(prediction) if finite else np.nan,
                    status="ok" if finite else "missing_required_comparator_input",
                )
                item.update(
                    {
                        "extrapolated_target": np.nan,
                        "model_family": np.nan,
                        "target_mode": np.nan,
                        "candidate_key": np.nan,
                    }
                )
                rows.append(item)
    return pd.DataFrame(rows)


def _finite_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.loc[
        np.isfinite(predictions["y_true"].to_numpy(float))
        & np.isfinite(predictions["y_pred"].to_numpy(float))
    ].copy()


def _metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for procedure, group in _finite_predictions(predictions).groupby("procedure", sort=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        interval = group[["interval_lower", "interval_upper"]].dropna()
        rows.append(
            {
                "procedure": procedure,
                **metrics,
                "interval_coverage": (
                    float(
                        ((interval["interval_lower"] <= group.loc[interval.index, "y_true"])
                        & (group.loc[interval.index, "y_true"] <= interval["interval_upper"])
                    ).mean()
                    )
                    if not interval.empty
                    else np.nan
                ),
                "interval_mean_width": (
                    float((interval["interval_upper"] - interval["interval_lower"]).mean())
                    if not interval.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _suppressed_metrics(predictions: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for (procedure, group), values in _finite_predictions(predictions).groupby(
        ["procedure", column]
    ):
        metrics = regression_metrics(values["y_true"], values["y_pred"])
        suppressed = metrics["n"] < 30
        if suppressed:
            metrics.update({key: np.nan for key in metrics if key != "n"})
        rows.append(
            {"procedure": procedure, label: group, "suppressed_n_lt_30": suppressed, **metrics}
        )
    return pd.DataFrame(rows)


def _paired_differences(predictions: pd.DataFrame) -> pd.DataFrame:
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
                right,
                on=["row_key", "site"],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            if paired.empty or not np.array_equal(paired["y_true_a"], paired["y_true_b"]):
                continue
            metrics_a = regression_metrics(paired["y_true_a"], paired["y_pred_a"])
            metrics_b = regression_metrics(paired["y_true_b"], paired["y_pred_b"])
            rows.append(
                {
                    "procedure_a": procedure_a,
                    "procedure_b": procedure_b,
                    "n_shared_finite": metrics_a["n"],
                    **{
                        f"{name}_difference": value - metrics_b[name]
                        for name, value in metrics_a.items()
                        if name != "n"
                    },
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_table(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for offset, (procedure, group) in enumerate(
        _finite_predictions(predictions).groupby("procedure", sort=True)
    ):
        if group["site"].nunique() >= 2:
            rows.append(cluster_bootstrap(group, seed=seed + offset).assign(procedure=procedure))
    return pd.concat(rows, ignore_index=True) if rows else _empty(["procedure", "iteration"])


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "featranker", "xlrd", "joblib", "matplotlib"]
    return {name: version(name) if _installed(name) else "not_installed" for name in names}


def _installed(name: str) -> bool:
    try:
        version(name)
    except PackageNotFoundError:
        return False
    return True


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def fit_final_model(
    frame: pd.DataFrame,
    feature_set: str,
    candidates: Sequence[ModelSpec],
    output_path: Path,
    seed: int,
) -> dict[str, object]:
    columns = feature_columns(feature_set, bool(frame.attrs["include_statin"]))
    X = select_feature_matrix(frame, columns)
    y = frame["weekly_dose_mg"].to_numpy(float)
    sites = frame["site"].astype(str).to_numpy()
    scores, _ = score_candidates(X, y, sites, columns, candidates, seed)
    selected = select_one_se(scores, candidates)
    residuals = calibration_residuals(X, y, sites, columns, selected, seed)
    pipeline = make_model_pipeline(columns, selected, seed)
    pipeline.fit(X, y)
    numeric_ranges = {
        column: [float(X[column].min()), float(X[column].max())]
        for column in columns
        if column in NUMERIC_FEATURES
    }
    payload = {
        "pipeline": pipeline,
        "feature_columns": columns,
        "feature_set": feature_set,
        "model_spec": selected,
        "conformal_radius": conformal_quantile(residuals, coverage=0.90),
        "numeric_training_ranges": numeric_ranges,
        "target_training_range": [float(y.min()), float(y.max())],
        "source_sha256": frame.attrs.get("source_sha256", _frame_sha256(frame)),
        "git_revision": _git_revision(),
        "research_warning": "Research use only; not prescribing guidance or a medical device.",
    }
    joblib.dump(payload, output_path)
    return payload


def _best_feature_set(predictions: pd.DataFrame) -> str:
    site_maes = (
        _finite_predictions(predictions)
        .query("procedure in ['clinical_ml', 'pharmacogenomic_ml']")
        .assign(absolute_error=lambda values: (values["y_true"] - values["y_pred"]).abs())
        .groupby(["procedure", "site"], as_index=False)["absolute_error"]
        .mean()
    )
    summary = site_maes.groupby("procedure")["absolute_error"].agg(["mean", "std", "count"])
    best = summary["mean"].idxmin()
    best_se = (0.0 if pd.isna(summary.loc[best, "std"]) else summary.loc[best, "std"]) / np.sqrt(
        summary.loc[best, "count"]
    )
    threshold = float(summary.loc[best, "mean"] + best_se)
    return (
        "clinical"
        if summary.loc["clinical_ml", "mean"] <= threshold
        else best.removesuffix("_ml")
    )


def _outer_statin_decisions(raw: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, object]]:
    decisions = []
    for outer_fold, (train, test) in enumerate(site_outer_splits(frame)):
        _, included, reason = statin_gate(raw.iloc[train])
        decisions.append(
            {
                "outer_fold": outer_fold,
                "outer_site": str(frame.iloc[test[0]]["site"]),
                "included": included,
                "reason": reason,
            }
        )
    return decisions


def run_primary_frame(
    raw: pd.DataFrame,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"refusing to overwrite existing run manifest: {output_dir / 'manifest.json'}"
        )
    cohort = prepare_cohort(raw)
    frame, metadata = build_feature_frame(cohort.data)
    frame.attrs["include_statin"] = metadata["include_statin"]
    frame.attrs["source_sha256"] = raw.attrs.get("source_sha256", _frame_sha256(raw))
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    if not candidates:
        raise ValueError("primary experiment requires at least one candidate")
    outer_statin = _outer_statin_decisions(cohort.data, frame)
    metadata["outer_fold_statin"] = outer_statin
    statin_included_by_fold = [bool(item["included"]) for item in outer_statin]
    clinical_columns = feature_columns("clinical", include_statin=False)
    pharmacogenomic_columns = feature_columns("pharmacogenomic", include_statin=False)
    clinical, clinical_selections, clinical_failures = _run_learned_procedure(
        frame,
        "clinical_ml",
        clinical_columns,
        candidates,
        seed,
        statin_included_by_fold,
    )
    pharmacogenomic, pharmacogenomic_selections, pharmacogenomic_failures = _run_learned_procedure(
        frame,
        "pharmacogenomic_ml",
        pharmacogenomic_columns,
        candidates,
        seed,
        statin_included_by_fold,
    )
    predictions = pd.concat([clinical, pharmacogenomic, _run_comparators(frame)], ignore_index=True)
    selections = pd.concat([clinical_selections, pharmacogenomic_selections], ignore_index=True)
    failures = pd.concat([clinical_failures, pharmacogenomic_failures], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "predictions.csv": predictions,
        "selections.csv": selections,
        "failures.csv": failures.reindex(columns=_FAILURE_COLUMNS),
        "cohort_flow.csv": cohort.flow,
        "issues.csv": cohort.issues,
        "metrics.csv": _metrics_table(predictions),
        "site_metrics.csv": _suppressed_metrics(predictions, "site", "site"),
        "dose_category_metrics.csv": _suppressed_metrics(
            predictions, "dose_category", "dose_category"
        ),
        "subgroup_metrics.csv": pd.concat(
            [
                _suppressed_metrics(predictions, column, "subgroup")
                for column in _AUDIT_COLUMNS[:-1]
            ],
            keys=_AUDIT_COLUMNS[:-1],
            names=["subgroup_type"],
        ).reset_index(level=0),
        "bootstrap.csv": _bootstrap_table(predictions, seed),
        "paired_differences.csv": _paired_differences(predictions),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_feature_set = _best_feature_set(predictions)
    fit_final_model(
        frame, selected_feature_set, candidates, output_dir / "final_model.joblib", seed
    )
    output_files = [*tables, "feature_metadata.json", "final_model.joblib", "manifest.json"]
    manifest = {
        "analysis": "primary",
        "seed": seed,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "source_sha256": frame.attrs["source_sha256"],
        "cohort_rows": len(frame),
        "site_count": int(frame["site"].nunique()),
        "model_grid": [
            {
                "candidate_key": spec.key,
                "family": spec.family,
                "target_mode": spec.target_mode,
                "params": spec.params,
            }
            for spec in candidates
        ],
        "output_files": output_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def run_primary_experiment(
    raw_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    candidates: Sequence[ModelSpec] | None = None,
) -> Path:
    raw = read_raw(raw_path)
    raw.attrs["source_sha256"] = sha256_file(raw_path)
    return run_primary_frame(raw, output_dir, candidates=candidates, seed=seed)
