import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from warfarin_dose.data import prepare_cohort
from warfarin_dose.evaluation import (
    _aggregate_outer_rankings,
    _atomic_joblib_dump,
    _outer_statin_decisions,
    cluster_bootstrap,
    conformal_interval,
    conformal_quantile,
    inner_site_splits,
    paired_cluster_bootstrap,
    regression_metrics,
    run_primary_frame,
    select_one_se,
    site_outer_splits,
)
from warfarin_dose.features import build_feature_frame
from warfarin_dose.models import ModelSpec


def rank_feature_blocks_for_encoded_test_matrix(seed: int) -> None:
    from warfarin_dose import evaluation

    n_rows = 12
    evaluation.rank_feature_blocks(
        pd.DataFrame(
            {
                "age_decade": np.repeat([5.0, 6.0, 7.0], 4),
                "gender": np.tile(["Female", "Male"], 6),
                "weight_kg": np.arange(n_rows, dtype=float),
            }
        ),
        np.arange(n_rows),
        np.repeat(["site-a", "site-b", "site-c"], 4),
        ["age_decade", "gender", "weight_kg"],
        seed,
    )


def test_featranker_receives_disjoint_inner_training_and_validation(monkeypatch):
    calls = []

    class SpyRanker:
        def __init__(self, **_):
            pass

        def fit(self, X, y, feature_names):
            self.train = set(np.asarray(y))
            return self

        def rank_features(
            self, X_eval, y_eval, *, scoring, feature_groups, n_repeats, random_state
        ):
            evaluation = set(np.asarray(y_eval))
            calls.append((self.train, evaluation, scoring, feature_groups, n_repeats, random_state))
            return {
                "models": {
                    "spy": {
                        "evaluation_score": -1.0,
                        "importance": {
                            name: {"values": [1.0], "mean": 1.0, "std": 0.0, "rank": rank}
                            for rank, name in enumerate(feature_groups, start=1)
                        },
                    }
                },
                "consensus": [
                    {
                        "feature_group": name,
                        "median_rank": rank,
                        "mean_rank": rank,
                        "rank_std": 0.0,
                        "n_models": 1,
                    }
                    for rank, name in enumerate(feature_groups, start=1)
                ],
                "failures": [],
                "evaluation_mode": "held_out",
            }

    monkeypatch.setattr("warfarin_dose.evaluation.FeatureRanker", SpyRanker, raising=False)
    rank_feature_blocks_for_encoded_test_matrix(seed=7)

    assert calls
    assert all(train.isdisjoint(validation) for train, validation, *_ in calls)
    assert all(call[2] == "neg_mean_absolute_error" for call in calls)
    assert all(call[4] == 20 for call in calls)


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


def test_statin_gate_receives_outer_training_sites_only(raw_frame, monkeypatch):
    cohort = prepare_cohort(raw_frame)
    frame, _ = build_feature_frame(cohort.data)
    seen_sites = []

    def spy_gate(training):
        seen_sites.append(set(training["Project Site"].astype(str)))
        return pd.Series("Unknown", index=training.index), False, "test"

    monkeypatch.setattr("warfarin_dose.evaluation.statin_gate", spy_gate)
    decisions = _outer_statin_decisions(cohort.data, frame)
    all_sites = set(frame["site"])

    assert len(decisions) == len(all_sites)
    for decision, training_sites in zip(decisions, seen_sites, strict=True):
        assert training_sites == all_sites - {decision["outer_site"]}


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


def test_one_se_prefers_simpler_family_then_direct_target():
    specs = [
        ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0),
        ModelSpec("hist_gb", {"learning_rate": 0.1, "max_leaf_nodes": 15}, "sqrt", 2, 0),
    ]
    scores = pd.DataFrame(
        {
            "candidate_key": [specs[0].key] * 3 + [specs[1].key] * 3,
            "fold": [0, 1, 2, 0, 1, 2],
            "mae_mg_week": [8.4, 8.5, 8.6, 7.0, 8.1, 9.2],
        }
    )

    assert select_one_se(scores, specs) == specs[0]


def test_outer_rank_aggregation_does_not_select_best_single_fold():
    ranks = pd.DataFrame(
        {
            "feature_block": ["lucky", "lucky", "stable", "stable"],
            "outer_fold": [0, 1, 0, 1],
            "median_rank": [1.0, 10.0, 3.0, 3.0],
            "mean_rank": [1.0, 10.0, 3.0, 3.0],
            "top5_frequency": [1.0, 0.0, 1.0, 1.0],
        }
    )

    aggregate = _aggregate_outer_rankings(ranks)

    assert aggregate["feature_block"].tolist() == ["stable", "lucky"]
    assert aggregate.set_index("feature_block").loc["lucky", "top5_frequency"] == 0.5


def test_atomic_joblib_dump_preserves_existing_artifact_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "final_model.joblib"
    output.write_bytes(b"existing")

    def fail_dump(_payload, path):
        Path(path).write_bytes(b"partial")
        raise OSError("write failed")

    monkeypatch.setattr("warfarin_dose.evaluation.joblib.dump", fail_dump)
    with pytest.raises(OSError, match="write failed"):
        _atomic_joblib_dump({"new": True}, output)

    assert output.read_bytes() == b"existing"
    assert not output.with_suffix(".joblib.tmp").exists()


def test_primary_experiment_predicts_every_patient_once(raw_frame, tmp_path):
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]

    result = run_primary_frame(raw_frame, tmp_path, candidates=candidates, seed=7)
    predictions = pd.read_csv(result / "predictions.csv")
    learned = predictions[predictions["procedure"].isin(["clinical_ml", "pharmacogenomic_ml"])]

    counts = learned.groupby(["procedure", "row_key"]).size()
    assert counts.eq(1).all()
    assert learned["y_pred"].ge(0).all()
    assert learned["outer_site"].eq(learned["site"]).all()
    assert learned[["interval_lower", "interval_upper"]].notna().all().all()


def test_primary_reuse_requires_matching_provenance(raw_frame, tmp_path):
    from warfarin_dose import evaluation

    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    primary = evaluation.run_primary_frame(
        raw_frame, tmp_path / "primary", candidates=candidates, seed=7
    )

    evaluation.validate_primary_run(raw_frame, primary, seed=7)
    manifest_path = primary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 8
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        evaluation.validate_primary_run(raw_frame, primary, seed=7)


def test_primary_final_artifact_uses_one_se_feature_set(raw_frame, tmp_path, monkeypatch):
    from warfarin_dose import evaluation

    captured = {}

    def fake_fit_final_model(frame, feature_set, candidates, output_path, seed, columns=None):
        captured.update(feature_set=feature_set, columns=columns)
        Path(output_path).write_bytes(b"model")
        return {
            "feature_columns": ["age_decade"],
            "git_revision": "model-revision",
            "source_sha256": "model-source",
        }

    monkeypatch.setattr(evaluation, "_best_feature_set", lambda _: "clinical")
    monkeypatch.setattr(evaluation, "fit_final_model", fake_fit_final_model)
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]

    output = evaluation.run_primary_frame(raw_frame, tmp_path, candidates=candidates, seed=7)

    assert captured == {"feature_set": "clinical", "columns": None}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_feature_columns"] == ["age_decade"]
    assert manifest["final_model_git_revision"] == "model-revision"
    assert manifest["final_model_source_sha256"] == "model-source"


def test_random_cv_uses_fold_local_statin_gate(raw_frame, tmp_path, monkeypatch):
    from warfarin_dose import evaluation

    gate_calls = []

    def include_statin(training):
        gate_calls.append(set(training.index))
        return pd.Series("Unknown", index=training.index), True, "test"

    monkeypatch.setattr(evaluation, "statin_gate", include_statin)
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]

    output = evaluation.run_random_cv_frame(
        raw_frame, tmp_path, candidates=candidates, seed=7
    )
    selections = pd.read_csv(output / "selections.csv")

    assert len(gate_calls) == 10
    assert selections["statin_included"].all()


def test_secondary_analyses_are_separate_and_keep_complete_case_from_final_model(
    raw_frame, tmp_path, monkeypatch
):
    from warfarin_dose import evaluation

    class SpyRanker:
        def __init__(self, **_):
            pass

        def fit(self, X, y, feature_names):
            return self

        def rank_features(
            self, X_eval, y_eval, *, scoring, feature_groups, n_repeats, random_state
        ):
            return {
                "models": {
                    "spy": {
                        "importance": {
                            name: {"mean": 1.0, "std": 0.0, "rank": rank}
                            for rank, name in enumerate(feature_groups, start=1)
                        }
                    }
                },
                "consensus": [],
                "failures": [],
                "evaluation_mode": "held_out",
            }

    monkeypatch.setattr(evaluation, "FeatureRanker", SpyRanker)
    monkeypatch.setattr(
        evaluation,
        "statin_gate",
        lambda training: (pd.Series("Unknown", index=training.index), True, "test"),
    )
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]

    result = evaluation.run_all_analyses_frame(raw_frame, tmp_path, candidates=candidates, seed=7)

    assert result == tmp_path
    assert (tmp_path / "primary" / "final_model.joblib").exists()
    assert (tmp_path / "feature-selection" / "feature_selection_decision.json").exists()
    assert (tmp_path / "complete-case" / "cohort_counts.json").exists()
    assert (tmp_path / "random-cv" / "manifest.json").exists()
    assert (tmp_path / "ablation" / "manifest.json").exists()
    random_manifest = json.loads((tmp_path / "random-cv" / "manifest.json").read_text())
    assert random_manifest["analysis_label"] == "optimism_comparator"
    complete_case = json.loads((tmp_path / "complete-case" / "manifest.json").read_text())
    assert complete_case["selects_primary_artifact"] is False
    complete_selections = pd.read_csv(tmp_path / "complete-case" / "selections.csv")
    assert not complete_selections["statin_included"].any()
    ablation_metrics = pd.read_csv(tmp_path / "ablation" / "metrics.csv")
    assert not ablation_metrics["procedure"].str.contains("ablation_ranked_").any()


def test_ranked_adoption_updates_final_artifact_provenance(tmp_path, monkeypatch):
    from warfarin_dose import evaluation

    raw = pd.DataFrame({"placeholder": [1]})
    raw.attrs["source_sha256"] = "public-source-checksum"
    primary_dir = tmp_path / "primary"
    primary_dir.mkdir()
    (primary_dir / "manifest.json").write_text(
        json.dumps({"final_feature_set": "pharmacogenomic"}), encoding="utf-8"
    )
    ranked_frame = pd.DataFrame({"weekly_dose_mg": [35.0]})
    captured = {}

    monkeypatch.setattr(evaluation, "run_primary_frame", lambda *_args, **_kwargs: primary_dir)
    monkeypatch.setattr(
        evaluation,
        "run_feature_selection_frame",
        lambda *_args, **_kwargs: {
            "decision": {
                "decision": "adopt_ranked_subset",
                "selected_feature_blocks": ["weight_kg", "vkorc1"],
            },
            "frame": ranked_frame,
        },
    )
    monkeypatch.setattr(evaluation, "run_complete_case_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluation, "run_random_cv_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluation, "run_ablation_frame", lambda *_args, **_kwargs: None)

    def fake_fit(frame, feature_set, candidates, output_path, seed, columns=None):
        captured.update(
            feature_set=feature_set,
            columns=columns,
            source_sha256=frame.attrs.get("source_sha256"),
        )
        return {
            "feature_set": feature_set,
            "feature_columns": list(columns),
            "git_revision": "final-model-revision",
            "source_sha256": frame.attrs["source_sha256"],
        }

    monkeypatch.setattr(evaluation, "fit_final_model", fake_fit)

    evaluation.run_all_analyses_frame(raw, tmp_path, candidates=[], seed=7)

    assert captured == {
        "feature_set": "pharmacogenomic_ranked",
        "columns": ["weight_kg", "vkorc1"],
        "source_sha256": "public-source-checksum",
    }
    manifest = json.loads((primary_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_feature_set"] == "pharmacogenomic_ranked"
    assert manifest["final_feature_columns"] == ["weight_kg", "vkorc1"]
    assert manifest["final_model_git_revision"] == "final-model-revision"
    assert manifest["final_model_source_sha256"] == "public-source-checksum"


def test_run_experiment_defaults_to_primary_analysis():
    from warfarin_dose.cli import build_parser

    args = build_parser().parse_args(["run-experiment"])

    assert args.analysis == "primary"
