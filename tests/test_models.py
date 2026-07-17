import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from warfarin_dose.models import (
    DoseRegressor,
    iwpc_clinical,
    iwpc_pharmacogenetic,
    model_candidates,
)


def test_weekly_and_daily_units_are_seven_days():
    weekly = 35.0
    assert weekly / 7 == 5.0


def test_published_iwpc_worked_examples():
    frame = pd.DataFrame(
        {
            "age_decade": [5.0],
            "height_cm": [175.0],
            "weight_kg": [80.0],
            "race": ["White"],
            "enzyme_inducer": ["No"],
            "amiodarone": ["No"],
            "vkorc1": ["G/G"],
            "cyp2c9_diplotype": ["*1/*1"],
        }
    )

    np.testing.assert_allclose(iwpc_clinical(frame), [34.82888256], rtol=0, atol=1e-8)
    np.testing.assert_allclose(iwpc_pharmacogenetic(frame), [46.83896721], rtol=0, atol=1e-8)
    frame.loc[0, ["vkorc1", "cyp2c9_diplotype"]] = ["A/G", "*1/*2"]
    np.testing.assert_allclose(iwpc_pharmacogenetic(frame), [29.75811601], rtol=0, atol=1e-8)


def test_iwpc_requires_age_height_weight_but_supports_unknown_genotype():
    frame = pd.DataFrame(
        {
            "age_decade": [np.nan, 5.0],
            "height_cm": [175.0, 175.0],
            "weight_kg": [80.0, 80.0],
            "race": ["Unknown", "Unknown"],
            "enzyme_inducer": ["Unknown", "Unknown"],
            "amiodarone": ["Unknown", "Unknown"],
            "vkorc1": ["Unknown", "Unknown"],
            "cyp2c9_diplotype": ["Unknown", "Unknown"],
        }
    )
    assert np.isnan(iwpc_pharmacogenetic(frame)[0])
    assert np.isfinite(iwpc_pharmacogenetic(frame)[1])


def test_dose_regressor_clips_before_inverse_square_root():
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, 4.0])
    model = DoseRegressor(DummyRegressor(strategy="constant", constant=-2), target_mode="sqrt").fit(
        X, y
    )
    assert model.predict(X).tolist() == [0.0, 0.0]


def test_dose_regressor_rejects_nonfinite_raw_prediction():
    X = np.array([[0.0]])
    model = DoseRegressor(DummyRegressor()).fit(X, [1.0])
    model.estimator_.constant_[:] = -np.inf

    with pytest.raises(ValueError, match="nonfinite"):
        model.predict(X)


def test_candidate_grid_is_small_and_deterministic():
    candidates = model_candidates(seed=42)
    assert {item.family for item in candidates} == {
        "ridge",
        "elasticnet",
        "hist_gb",
        "random_forest",
        "mlp",
    }
    assert {item.target_mode for item in candidates} == {"direct", "sqrt"}
    assert len(candidates) == 38
