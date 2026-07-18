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
from featranker import FeatureRanker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from .data import prepare_cohort, read_raw, sha256_file
from .features import (
    NUMERIC_FEATURES,
    build_feature_frame,
    feature_columns,
    make_preprocessor,
    select_feature_matrix,
    semantic_feature_groups,
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


def random_outer_splits(n_rows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(n_splits=10, shuffle=True, random_state=seed).split(np.arange(n_rows)))


def random_inner_splits(n_rows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_rows)))


def resolve_inner_splits(
    sites: np.ndarray, seed: int, mode: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if mode == "site":
        return inner_site_splits(sites)
    if mode == "random":
        return random_inner_splits(len(sites), seed)
    raise ValueError(f"unknown inner split mode: {mode}")


def rank_feature_blocks(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rank_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for fold, (train, validation) in enumerate(inner_site_splits(sites)):
        try:
            preprocessor = make_preprocessor(columns, scale_numeric=False)
            X_train = preprocessor.fit_transform(X.iloc[train])
            X_validation = preprocessor.transform(X.iloc[validation])
            names = preprocessor.get_feature_names_out().tolist()
            groups = semantic_feature_groups(names)
            ranker = FeatureRanker(task="reg", group="all")
            ranker.fit(X_train, y[train], feature_names=names)
            report = ranker.rank_features(
                X_validation,
                y[validation],
                scoring="neg_mean_absolute_error",
                feature_groups=groups,
                n_repeats=20,
                random_state=seed + fold,
            )
            if report.get("evaluation_mode") != "held_out":
                raise ValueError("FeatRanker must report held_out evaluation mode")
            reports.append({"fold": fold, "report": report})
        except Exception as error:
            reports.append(
                {
                    "fold": fold,
                    "failure": {"error_type": type(error).__name__, "message": str(error)},
                }
            )
            continue
        for model, model_report in report["models"].items():
            for block, importance in model_report["importance"].items():
                rank_rows.append(
                    {
                        "fold": fold,
                        "model": model,
                        "feature_block": block,
                        "rank": importance["rank"],
                        "importance_mean": importance["mean"],
                        "importance_std": importance["std"],
                    }
                )
    ranks = pd.DataFrame(rank_rows)
    if ranks.empty:
        raise RuntimeError("FeatRanker produced no successful model rankings")
    aggregate = (
        ranks.groupby("feature_block")["rank"]
        .agg(median_rank="median", mean_rank="mean", rank_std="std", observations="size")
        .reset_index()
    )
    top5 = ranks.groupby("feature_block")["rank"].apply(lambda values: float((values <= 5).mean()))
    aggregate["top5_frequency"] = aggregate["feature_block"].map(top5)
    aggregate = aggregate.sort_values(
        ["median_rank", "mean_rank", "feature_block"]
    ).reset_index(drop=True)
    return aggregate, reports


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
    inner_mode: str = "site",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    splits = resolve_inner_splits(sites, seed, inner_mode)
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
    inner_mode: str = "site",
) -> np.ndarray:
    residuals = np.full(len(X), np.nan)
    for fold, (train, validation) in enumerate(resolve_inner_splits(sites, seed, inner_mode)):
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
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    inner_mode: str = "site",
    analysis_label: str = "site_held_out",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame["weekly_dose_mg"].to_numpy(float)
    predictions: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    outer_splits = site_outer_splits(frame) if outer_splits is None else list(outer_splits)
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
            inner_mode,
        )
        selected = select_one_se(scores, candidates)
        residuals = calibration_residuals(
            X.iloc[train].reset_index(drop=True),
            y[train],
            training_sites,
            fold_columns,
            selected,
            seed + outer_fold * 100,
            inner_mode,
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
                    "analysis_label": analysis_label,
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
                "analysis_label": analysis_label,
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


def _paired_bootstrap_table(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    finite = _finite_predictions(predictions)
    procedures = sorted(finite["procedure"].unique())
    pair_index = 0
    for index, procedure_a in enumerate(procedures):
        left = finite.loc[finite["procedure"].eq(procedure_a)]
        for procedure_b in procedures[index + 1 :]:
            right = finite.loc[finite["procedure"].eq(procedure_b)]
            shared_sites = left[["row_key", "site"]].merge(
                right[["row_key", "site"]], on=["row_key", "site"]
            )["site"].nunique()
            if shared_sites >= 2:
                rows.append(
                    paired_cluster_bootstrap(left, right, seed=seed + pair_index).assign(
                        procedure_a=procedure_a, procedure_b=procedure_b
                    )
                )
            pair_index += 1
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else _empty(["procedure_a", "procedure_b", "iteration"])
    )


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


def _atomic_joblib_dump(payload: object, output_path: Path) -> None:
    output_path = Path(output_path)
    partial = output_path.with_suffix(output_path.suffix + ".tmp")
    partial.unlink(missing_ok=True)
    try:
        joblib.dump(payload, partial)
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def fit_final_model(
    frame: pd.DataFrame,
    feature_set: str,
    candidates: Sequence[ModelSpec],
    output_path: Path,
    seed: int,
    columns: Sequence[str] | None = None,
) -> dict[str, object]:
    columns = list(
        feature_columns(feature_set, bool(frame.attrs["include_statin"]))
        if columns is None
        else columns
    )
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
    categorical_training_values = {
        column: sorted(X[column].dropna().astype(str).unique().tolist())
        for column in columns
        if column not in NUMERIC_FEATURES
    }
    payload = {
        "pipeline": pipeline,
        "feature_columns": columns,
        "feature_set": feature_set,
        "model_spec": selected,
        "conformal_radius": conformal_quantile(residuals, coverage=0.90),
        "numeric_training_ranges": numeric_ranges,
        "categorical_training_values": categorical_training_values,
        "target_training_range": [float(y.min()), float(y.max())],
        "source_sha256": frame.attrs.get("source_sha256", _frame_sha256(frame)),
        "git_revision": _git_revision(),
        "research_warning": "Research use only; not prescribing guidance or a medical device.",
    }
    _atomic_joblib_dump(payload, output_path)
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


def _statin_included_for_splits(
    raw: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[bool]:
    return [bool(statin_gate(raw.iloc[train])[1]) for train, _ in splits]


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )


def _new_analysis_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_secondary_analysis(
    output_dir: Path,
    analysis: str,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    failures: pd.DataFrame,
    seed: int,
    *,
    analysis_label: str = "site_held_out",
    extra: dict[str, object] | None = None,
) -> Path:
    output_dir = _new_analysis_dir(output_dir)
    tables = {
        "predictions.csv": predictions,
        "selections.csv": selections,
        "failures.csv": failures.reindex(columns=_FAILURE_COLUMNS),
        "metrics.csv": _metrics_table(predictions),
        "site_metrics.csv": _suppressed_metrics(predictions, "site", "site"),
    }
    if analysis_label == "optimism_comparator":
        tables = {
            name: table.assign(analysis_label=analysis_label)
            for name, table in tables.items()
        }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    manifest = {
        "analysis": analysis,
        "analysis_label": analysis_label,
        "seed": seed,
        "git_revision": _git_revision(),
        "output_files": [*tables, "manifest.json"],
    }
    if extra:
        manifest.update(extra)
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def _subset_summary(
    scores: pd.DataFrame, selected: ModelSpec, name: str, columns: Sequence[str]
) -> dict[str, object]:
    mae = scores.loc[scores["candidate_key"].eq(selected.key), "mae_mg_week"]
    return {
        "subset": name,
        "feature_blocks": list(columns),
        "candidate_key": selected.key,
        "mean_mae_mg_week": float(mae.mean()),
        "se_mae_mg_week": float(mae.std(ddof=1) / np.sqrt(len(mae))) if len(mae) > 1 else 0.0,
    }


def _run_ranked_procedure(
    frame: pd.DataFrame,
    columns: Sequence[str],
    candidates: Sequence[ModelSpec],
    seed: int,
    statin_included_by_fold: Sequence[bool],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    y = frame["weekly_dose_mg"].to_numpy(float)
    predictions: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    ranking_rows: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    outer_splits = site_outer_splits(frame)
    if len(statin_included_by_fold) != len(outer_splits):
        raise ValueError("statin decisions must match outer folds")
    for outer_fold, (train, test) in enumerate(outer_splits):
        fold_columns = list(columns)
        if statin_included_by_fold[outer_fold]:
            fold_columns.append("statin")
        X = select_feature_matrix(frame, fold_columns)
        training = X.iloc[train].reset_index(drop=True)
        training_sites = frame.iloc[train]["site"].astype(str).to_numpy()
        outer_site = str(frame.iloc[test[0]]["site"])
        ranks, fold_reports = rank_feature_blocks(
            training, y[train], training_sites, fold_columns, seed + outer_fold * 100
        )
        ranking_rows.append(ranks.assign(outer_fold=outer_fold, outer_site=outer_site))
        reports.extend(
            [
                {"outer_fold": outer_fold, "outer_site": outer_site, **report}
                for report in fold_reports
            ]
        )
        ordered = ranks["feature_block"].tolist()
        subsets = [("top5", ordered[:5]), ("top10", ordered[:10]), ("all", ordered)]
        summaries: list[dict[str, object]] = []
        selected_by_subset: dict[str, ModelSpec] = {}
        for name, subset_columns in subsets:
            scores, fold_failures = score_candidates(
                training,
                y[train],
                training_sites,
                subset_columns,
                candidates,
                seed + outer_fold * 100,
            )
            selected = select_one_se(scores, candidates)
            selected_by_subset[name] = selected
            summaries.append(_subset_summary(scores, selected, name, subset_columns))
            failures.extend(
                fold_failures.assign(
                    procedure="pharmacogenomic_ranked", outer_fold=outer_fold, outer_site=outer_site
                ).to_dict("records")
            )
        best = min(summaries, key=lambda item: float(item["mean_mae_mg_week"]))
        threshold = float(best["mean_mae_mg_week"]) + float(best["se_mae_mg_week"])
        chosen = next(item for item in summaries if float(item["mean_mae_mg_week"]) <= threshold)
        chosen_columns = list(chosen["feature_blocks"])
        selected = selected_by_subset[str(chosen["subset"])]
        residuals = calibration_residuals(
            training,
            y[train],
            training_sites,
            chosen_columns,
            selected,
            seed + outer_fold * 100,
        )
        pipeline = make_model_pipeline(chosen_columns, selected, seed + outer_fold)
        pipeline.fit(X.iloc[train], y[train])
        predicted = pipeline.predict(X.iloc[test])
        lower, upper = conformal_interval(predicted, conformal_quantile(residuals, coverage=0.90))
        target_min, target_max = float(y[train].min()), float(y[train].max())
        for position, prediction, low, high in zip(test, predicted, lower, upper, strict=True):
            item = _prediction_row(
                frame,
                position,
                "pharmacogenomic_ranked",
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
                    "analysis_label": "site_held_out",
                }
            )
            predictions.append(item)
        selections.append(
            {
                "procedure": "pharmacogenomic_ranked",
                "outer_fold": outer_fold,
                "outer_site": outer_site,
                "subset": chosen["subset"],
                "feature_blocks": json.dumps(chosen_columns),
                "candidate_key": selected.key,
                "subset_summaries": json.dumps(summaries),
                "statin_included": "statin" in fold_columns,
            }
        )
    result = pd.DataFrame(predictions)
    if len(result) != len(frame) or not result.groupby("row_key").size().eq(1).all():
        raise ValueError(
            "pharmacogenomic_ranked did not produce exactly one outer prediction per patient"
        )
    return (
        result,
        pd.DataFrame(selections),
        pd.DataFrame(failures, columns=_FAILURE_COLUMNS),
        pd.concat(ranking_rows, ignore_index=True),
        reports,
    )


def _site_mae_summary(predictions: pd.DataFrame, procedure: str) -> dict[str, float]:
    values = _finite_predictions(predictions).loc[lambda rows: rows["procedure"].eq(procedure)]
    per_site = (
        (values["y_true"] - values["y_pred"]).abs().groupby(values["site"]).mean().to_numpy(float)
    )
    return {
        "mean_site_mae_mg_week": float(per_site.mean()),
        "site_mae_se_mg_week": float(per_site.std(ddof=1) / np.sqrt(len(per_site)))
        if len(per_site) > 1
        else 0.0,
        "site_count": int(len(per_site)),
    }


def _aggregate_outer_rankings(ranks: pd.DataFrame) -> pd.DataFrame:
    aggregate = ranks.groupby("feature_block", as_index=False).agg(
        median_rank=("median_rank", "median"),
        mean_rank=("mean_rank", "mean"),
        rank_std=("median_rank", "std"),
        top5_frequency=("top5_frequency", "mean"),
        outer_folds=("outer_fold", "nunique"),
    )
    aggregate["rank_std"] = aggregate["rank_std"].fillna(0.0)
    return aggregate.sort_values(
        ["median_rank", "mean_rank", "feature_block"]
    ).reset_index(drop=True)


def run_feature_selection_frame(
    raw: pd.DataFrame,
    primary_dir: Path,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    cohort = prepare_cohort(raw)
    frame, metadata = build_feature_frame(cohort.data)
    frame.attrs["include_statin"] = metadata["include_statin"]
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    statin = _outer_statin_decisions(cohort.data, frame)
    base_columns = feature_columns("pharmacogenomic", include_statin=False)
    predictions, selections, failures, ranks, reports = _run_ranked_procedure(
        frame, base_columns, candidates, seed, [bool(item["included"]) for item in statin]
    )
    primary_predictions = pd.read_csv(Path(primary_dir) / "predictions.csv")
    all_feature = _site_mae_summary(primary_predictions, "pharmacogenomic_ml")
    ranked = _site_mae_summary(predictions, "pharmacogenomic_ranked")
    aggregate_ranks = _aggregate_outer_rankings(ranks)
    available_columns = base_columns + (["statin"] if metadata["include_statin"] else [])
    aggregate_ranks = aggregate_ranks.loc[
        aggregate_ranks["feature_block"].isin(available_columns)
    ].reset_index(drop=True)
    selected_counts = selections["feature_blocks"].map(lambda value: len(json.loads(value)))
    ranked_blocks = min(int(selected_counts.median()), len(aggregate_ranks))
    selected_feature_blocks = aggregate_ranks["feature_block"].head(ranked_blocks).tolist()
    all_blocks = len(available_columns)
    adopt = ranked["mean_site_mae_mg_week"] < all_feature["mean_site_mae_mg_week"] or (
        ranked["mean_site_mae_mg_week"]
        <= all_feature["mean_site_mae_mg_week"] + all_feature["site_mae_se_mg_week"]
        and ranked_blocks <= 0.70 * all_blocks
    )
    decision = {
        "decision": "adopt_ranked_subset" if adopt else "retain_all_features",
        "rule": (
            "lower mean site MAE, or within one all-feature site-MAE SE with at least "
            "30% fewer semantic blocks"
        ),
        "all_feature": all_feature,
        "ranked": ranked,
        "ranked_block_count": ranked_blocks,
        "all_feature_block_count": all_blocks,
        "selected_feature_blocks": selected_feature_blocks,
        "outer_predictions_frozen": True,
    }
    output_dir = _write_secondary_analysis(
        output_dir, "feature-selection", predictions, selections, failures, seed
    )
    ranks.to_csv(output_dir / "feature_rankings_by_outer_fold.csv", index=False)
    aggregate_ranks.to_csv(output_dir / "feature_rankings.csv", index=False)
    _write_json(output_dir / "featranker_reports.json", reports)
    _write_json(output_dir / "feature_selection_decision.json", decision)
    return {
        "decision": decision,
        "frame": frame,
        "base_columns": base_columns,
        "ranks": aggregate_ranks,
    }


def _complete_case_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    numeric = [column for column in columns if column in NUMERIC_FEATURES]
    categorical = [column for column in columns if column not in NUMERIC_FEATURES]
    return pd.Series(
        np.isfinite(frame[numeric].to_numpy(float)).all(axis=1)
        & frame[categorical].ne("Unknown").all(axis=1).to_numpy(),
        index=frame.index,
    )


def run_complete_case_frame(
    raw: pd.DataFrame,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    cohort = prepare_cohort(raw)
    frame, metadata = build_feature_frame(cohort.data)
    frame.attrs["include_statin"] = metadata["include_statin"]
    base_columns = feature_columns("pharmacogenomic", include_statin=False)
    mask = _complete_case_mask(frame, base_columns)
    complete_frame = frame.loc[mask].reset_index(drop=True)
    retained = complete_frame.groupby("site").size().rename("complete_case_n").to_dict()
    totals = frame.groupby("site").size().rename("eligible_n").to_dict()
    counts = {
        "eligible_rows": len(frame),
        "complete_case_rows": len(complete_frame),
        "site_retention": [
            {
                "site": site,
                "eligible_n": int(totals[site]),
                "complete_case_n": int(retained.get(site, 0)),
            }
            for site in sorted(totals)
        ],
    }
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    outer_splits = site_outer_splits(complete_frame)
    predictions, selections, failures = _run_learned_procedure(
        complete_frame,
        "pharmacogenomic_complete_case",
        base_columns,
        candidates,
        seed,
        [False] * len(outer_splits),
        outer_splits=outer_splits,
    )
    output_dir = _write_secondary_analysis(
        output_dir, "complete-case", predictions, selections, failures, seed
    )
    _write_json(output_dir / "cohort_counts.json", counts)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    manifest["selects_primary_artifact"] = False
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def run_random_cv_frame(
    raw: pd.DataFrame,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    cohort = prepare_cohort(raw)
    frame, metadata = build_feature_frame(cohort.data)
    frame.attrs["include_statin"] = metadata["include_statin"]
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    outer_splits = random_outer_splits(len(frame), seed)
    statin_included_by_fold = _statin_included_for_splits(cohort.data, outer_splits)
    predictions, selections, failures = _run_learned_procedure(
        frame,
        "pharmacogenomic_random_cv",
        feature_columns("pharmacogenomic", include_statin=False),
        candidates,
        seed,
        statin_included_by_fold,
        outer_splits=outer_splits,
        inner_mode="random",
        analysis_label="optimism_comparator",
    )
    return _write_secondary_analysis(
        output_dir,
        "random-cv",
        predictions,
        selections,
        failures,
        seed,
        analysis_label="optimism_comparator",
    )


def run_ablation_frame(
    raw: pd.DataFrame,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    cohort = prepare_cohort(raw)
    frame, metadata = build_feature_frame(cohort.data)
    frame.attrs["include_statin"] = metadata["include_statin"]
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    base_columns = feature_columns("pharmacogenomic", include_statin=False)
    statin = _outer_statin_decisions(cohort.data, frame)
    blocks = {
        "demographics": ["age_decade", "gender"],
        "anthropometrics": ["height_cm", "weight_kg"],
        "clinical_conditions": [
            "indication",
            "target_inr",
            "diabetes",
            "chf_cardiomyopathy",
            "valve_replacement",
        ],
        "medications": ["amiodarone", "enzyme_inducer", "smoker", "statin"],
        "pharmacogenomics": ["cyp2c9_group", "vkorc1"],
    }
    all_predictions: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    all_failures: list[pd.DataFrame] = []
    for name, removed in blocks.items():
        columns = [column for column in base_columns if column not in removed]
        fold_statin = [False] * len(statin) if "statin" in removed else [
            bool(item["included"]) for item in statin
        ]
        predictions, selections, failures = _run_learned_procedure(
            frame,
            f"pharmacogenomic_ablation_{name}",
            columns,
            candidates,
            seed,
            fold_statin,
        )
        all_predictions.append(predictions)
        all_selections.append(selections.assign(ablation=name, removed_blocks=json.dumps(removed)))
        all_failures.append(failures)
    output_dir = _write_secondary_analysis(
        output_dir,
        "ablation",
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_selections, ignore_index=True),
        pd.concat(all_failures, ignore_index=True),
        seed,
    )
    _write_json(output_dir / "ablation_plan.json", {"blocks": blocks})
    return output_dir


def run_all_analyses_frame(
    raw: pd.DataFrame,
    output_dir: Path,
    candidates: Sequence[ModelSpec] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    output_dir = Path(output_dir)
    candidates = list(model_candidates(seed) if candidates is None else candidates)
    primary_dir = run_primary_frame(raw, output_dir / "primary", candidates=candidates, seed=seed)
    result = run_feature_selection_frame(
        raw, primary_dir, output_dir / "feature-selection", candidates=candidates, seed=seed
    )
    run_complete_case_frame(raw, output_dir / "complete-case", candidates=candidates, seed=seed)
    run_random_cv_frame(raw, output_dir / "random-cv", candidates=candidates, seed=seed)
    run_ablation_frame(raw, output_dir / "ablation", candidates=candidates, seed=seed)
    if result["decision"]["decision"] == "adopt_ranked_subset":
        frame = result["frame"]
        columns = list(result["decision"]["selected_feature_blocks"])
        frame.attrs["source_sha256"] = raw.attrs.get("source_sha256", _frame_sha256(raw))
        payload = fit_final_model(
            frame,
            "pharmacogenomic_ranked",
            candidates,
            primary_dir / "final_model.joblib",
            seed,
            columns,
        )
        manifest_path = primary_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["final_feature_set"] = payload["feature_set"]
        manifest["final_feature_columns"] = payload["feature_columns"]
        manifest["final_model_git_revision"] = payload["git_revision"]
        manifest["final_model_source_sha256"] = payload["source_sha256"]
        _write_json(manifest_path, manifest)
    return output_dir


def run_all_analyses(raw_path: Path, output_dir: Path, seed: int = DEFAULT_SEED) -> Path:
    raw = read_raw(raw_path)
    raw.attrs["source_sha256"] = sha256_file(raw_path)
    return run_all_analyses_frame(raw, output_dir, seed=seed)


def validate_primary_run(raw: pd.DataFrame, primary_dir: Path, seed: int) -> None:
    primary_dir = Path(primary_dir)
    manifest = json.loads((primary_dir / "manifest.json").read_text(encoding="utf-8"))
    cohort = prepare_cohort(raw)
    expected_source = raw.attrs.get("source_sha256", _frame_sha256(raw))
    problems = []
    if manifest.get("source_sha256") != expected_source:
        problems.append("source checksum")
    if manifest.get("seed") != seed:
        problems.append("seed")
    if manifest.get("git_revision") != _git_revision():
        problems.append("git revision")
    if manifest.get("cohort_rows") != len(cohort.data):
        problems.append("cohort size")
    predictions = pd.read_csv(
        primary_dir / "predictions.csv", usecols=["row_key", "procedure"], low_memory=False
    )
    learned = predictions.loc[predictions["procedure"].eq("clinical_ml"), "row_key"]
    expected_keys = set(cohort.data["row_key"].astype(str))
    if learned.duplicated().any() or set(learned.astype(str)) != expected_keys:
        problems.append("patient-key coverage")
    if problems:
        raise ValueError(f"incompatible primary run: {', '.join(problems)}")


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
        "paired_bootstrap.csv": _paired_bootstrap_table(predictions, seed),
        "paired_differences.csv": _paired_differences(predictions),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_feature_set = _best_feature_set(predictions)
    final_payload = fit_final_model(
        frame, final_feature_set, candidates, output_dir / "final_model.joblib", seed
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
        "final_feature_set": final_feature_set,
        "final_feature_columns": final_payload["feature_columns"],
        "final_model_git_revision": final_payload["git_revision"],
        "final_model_source_sha256": final_payload["source_sha256"],
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
