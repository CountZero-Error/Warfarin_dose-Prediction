from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

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
