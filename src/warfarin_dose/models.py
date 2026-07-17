from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def _race_terms(race: pd.Series, clinical: bool) -> np.ndarray:
    text = race.fillna("Unknown").astype(str).str.lower()
    asian = text.str.contains("asian").astype(float)
    black = text.str.contains("black|african").astype(float)
    missing_mixed = text.str.contains("unknown|missing|mixed").astype(float)
    if clinical:
        return -0.6752 * asian + 0.4060 * black + 0.0443 * missing_mixed
    return -0.1092 * asian - 0.2760 * black - 0.1032 * missing_mixed


def _required_iwpc(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, np.ndarray]:
    age = pd.to_numeric(frame["age_decade"], errors="coerce")
    height = pd.to_numeric(frame["height_cm"], errors="coerce")
    weight = pd.to_numeric(frame["weight_kg"], errors="coerce")
    valid = np.isfinite(age) & np.isfinite(height) & np.isfinite(weight)
    return age, height, weight, valid


def iwpc_clinical(frame: pd.DataFrame) -> np.ndarray:
    age, height, weight, valid = _required_iwpc(frame)
    linear = (
        4.0376
        - 0.2546 * age
        + 0.0118 * height
        + 0.0134 * weight
        + _race_terms(frame["race"], clinical=True)
        + 1.2799 * frame["enzyme_inducer"].eq("Yes").astype(float)
        - 0.5695 * frame["amiodarone"].eq("Yes").astype(float)
    )
    return np.where(valid, np.square(linear), np.nan)


def iwpc_pharmacogenetic(frame: pd.DataFrame) -> np.ndarray:
    age, height, weight, valid = _required_iwpc(frame)
    vkor = frame["vkorc1"].fillna("Unknown")
    cyp = frame["cyp2c9_diplotype"].fillna("Unknown")
    linear = (
        5.6044
        - 0.2614 * age
        + 0.0087 * height
        + 0.0128 * weight
        - 0.8677 * vkor.eq("A/G")
        - 1.6974 * vkor.eq("A/A")
        - 0.4854 * vkor.eq("Unknown")
        - 0.5211 * cyp.eq("*1/*2")
        - 0.9357 * cyp.eq("*1/*3")
        - 1.0616 * cyp.eq("*2/*2")
        - 1.9206 * cyp.eq("*2/*3")
        - 2.3312 * cyp.eq("*3/*3")
        - 0.2188 * cyp.eq("Unknown")
        + _race_terms(frame["race"], clinical=False)
        + 1.1816 * frame["enzyme_inducer"].eq("Yes").astype(float)
        - 0.5503 * frame["amiodarone"].eq("Yes").astype(float)
    )
    return np.where(valid, np.square(linear.astype(float)), np.nan)


class DoseRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator: BaseEstimator, target_mode: str = "direct"):
        self.estimator = estimator
        self.target_mode = target_mode

    def fit(self, X, y):
        if self.target_mode not in {"direct", "sqrt"}:
            raise ValueError(f"unsupported target mode: {self.target_mode}")
        target = np.asarray(y, dtype=float)
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("training targets must be finite positive mg/week")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.sqrt(target) if self.target_mode == "sqrt" else target)
        return self

    def predict(self, X):
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("model produced nonfinite weekly-dose predictions")
        prediction = np.clip(prediction, 0.0, None)
        prediction = np.square(prediction) if self.target_mode == "sqrt" else prediction
        if not np.isfinite(prediction).all():
            raise ValueError("model produced nonfinite weekly-dose predictions")
        return prediction


@dataclass(frozen=True)
class ModelSpec:
    family: str
    params: dict[str, object]
    target_mode: str
    family_order: int
    complexity_order: int

    @property
    def key(self) -> str:
        values = ",".join(f"{key}={self.params[key]}" for key in sorted(self.params))
        return f"{self.family}|{self.target_mode}|{values}"


GRIDS = [
    ("ridge", 0, {"alpha": [0.1, 1.0, 10.0]}),
    ("elasticnet", 1, {"alpha": [0.001, 0.01], "l1_ratio": [0.25, 0.75]}),
    ("hist_gb", 2, {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    ("random_forest", 3, {"max_depth": [None, 10], "min_samples_leaf": [2, 8]}),
    ("mlp", 4, {"alpha": [0.001, 0.01], "hidden_layer_sizes": [(32,), (32, 16)]}),
]


def model_candidates(seed: int) -> list[ModelSpec]:
    del seed
    result = []
    for family, family_order, grid in GRIDS:
        for complexity_order, params in enumerate(ParameterGrid(grid)):
            for target_mode in ("direct", "sqrt"):
                result.append(
                    ModelSpec(family, params, target_mode, family_order, complexity_order)
                )
    return result


def _estimator(spec: ModelSpec, seed: int = 20260717) -> BaseEstimator:
    if spec.family == "ridge":
        return Ridge(**spec.params)
    if spec.family == "elasticnet":
        return ElasticNet(max_iter=20_000, random_state=seed, **spec.params)
    if spec.family == "hist_gb":
        return HistGradientBoostingRegressor(max_iter=300, random_state=seed, **spec.params)
    if spec.family == "random_forest":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed, **spec.params)
    if spec.family == "mlp":
        return MLPRegressor(
            early_stopping=True,
            max_iter=500,
            random_state=seed,
            learning_rate_init=0.001,
            **spec.params,
        )
    raise ValueError(f"unknown model family: {spec.family}")


def make_model_pipeline(
    columns: Sequence[str], spec: ModelSpec, seed: int = 20260717
) -> Pipeline:
    scale = spec.family in {"ridge", "elasticnet", "mlp"}
    return Pipeline(
        [
            ("preprocess", make_preprocessor(columns, scale_numeric=scale)),
            ("regressor", DoseRegressor(_estimator(spec, seed), target_mode=spec.target_mode)),
        ]
    )
