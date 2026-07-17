import numpy as np
import pandas as pd
import pytest

from warfarin_dose.evaluation import (
    cluster_bootstrap,
    conformal_interval,
    conformal_quantile,
    inner_site_splits,
    paired_cluster_bootstrap,
    regression_metrics,
    site_outer_splits,
)


def test_site_splits_are_disjoint_and_cover_each_row_once():
    frame = pd.DataFrame(
        {
            "site": np.repeat(["a", "b", "c", "d"], 3),
            "patient_key": [f"patient-{index}" for index in range(12)],
        }
    )
    seen = np.zeros(len(frame), dtype=int)
    for train, test in site_outer_splits(frame):
        assert set(frame.iloc[train]["site"]).isdisjoint(frame.iloc[test]["site"])
        seen[test] += 1
    assert seen.tolist() == [1] * len(frame)


def test_inner_site_splits_require_three_sites():
    with pytest.raises(ValueError, match="at least three"):
        inner_site_splits(["a", "a", "b", "b"])


def test_metrics_and_dose_categories():
    metrics = regression_metrics(
        np.array([10.0, 35.0, 60.0]), np.array([12.0, 28.0, 66.0])
    )
    assert metrics["mae_mg_week"] == 5.0
    assert metrics["rmse_mg_week"] == pytest.approx(np.sqrt(89 / 3))
    assert metrics["pw20"] == 1.0


def test_finite_sample_conformal_quantile_and_nonnegative_lower_bound():
    assert conformal_quantile([1, 2, 3, 4], coverage=0.80) == 4
    lower, upper = conformal_interval(np.array([2.0]), radius=4.0)
    assert lower.tolist() == [0.0]
    assert upper.tolist() == [6.0]


def test_site_cluster_bootstrap_is_seeded():
    predictions = pd.DataFrame(
        {
            "site": ["a", "a", "b", "b"],
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [11.0, 18.0, 33.0, 36.0],
        }
    )
    first = cluster_bootstrap(predictions, iterations=20, seed=7)
    second = cluster_bootstrap(predictions, iterations=20, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_paired_cluster_bootstrap_uses_only_matched_rows_and_preserves_alignment():
    predictions_a = pd.DataFrame(
        {
            "row_key": ["row-a", "row-b", "only-a"],
            "site": ["a", "b", "b"],
            "y_true": [10.0, 20.0, 1_000.0],
            "y_pred": [12.0, 22.0, 0.0],
        }
    )
    predictions_b = pd.DataFrame(
        {
            "row_key": ["row-b", "only-b", "row-a"],
            "site": ["b", "a", "a"],
            "y_true": [20.0, 1_000.0, 10.0],
            "y_pred": [21.0, 0.0, 11.0],
        }
    )

    differences = paired_cluster_bootstrap(predictions_a, predictions_b, iterations=10, seed=7)

    assert differences["mae_mg_week_difference"].tolist() == [1.0] * 10


def test_paired_cluster_bootstrap_rejects_duplicate_pairing_keys():
    predictions = pd.DataFrame(
        {
            "row_key": ["row-a", "row-a"],
            "site": ["a", "a"],
            "y_true": [10.0, 10.0],
            "y_pred": [11.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        paired_cluster_bootstrap(predictions, predictions)
