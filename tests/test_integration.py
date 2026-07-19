import json

import joblib
import pandas as pd
import pytest

from warfarin_dose.evaluation import run_primary_frame
from warfarin_dose.models import ModelSpec
from warfarin_dose.reporting import build_report, predict_patient


def test_binned_calibration_aggregates_patient_rows():
    from warfarin_dose import reporting

    predictions = pd.DataFrame(
        {
            "procedure": ["a"] * 12 + ["b"] * 12,
            "y_true": list(range(12)) * 2,
            "y_pred": list(range(1, 13)) + list(range(2, 14)),
        }
    )

    summary = reporting._binned_calibration(predictions, bins=3)

    assert set(summary) == {"procedure", "observed", "predicted", "n"}
    assert len(summary) <= 6
    assert summary["n"].sum() == len(predictions)


def test_synthetic_run_builds_report_and_safe_prediction(raw_frame, tmp_path):
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    run_dir = run_primary_frame(raw_frame, tmp_path / "run", candidates=candidates, seed=7)
    stale_selections = run_dir / "report" / "tables" / "selections.csv"
    stale_selections.parent.mkdir(parents=True)
    stale_selections.write_text("outer_fold,outer_site\n0,1\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_model_git_revision"] = "separate-final-model-revision"
    manifest["final_feature_set"] = "pharmacogenomic_ranked"
    manifest["final_feature_columns"] = ["vkorc1", "weight_kg"]
    manifest["final_model_source_sha256"] = "public-source-checksum"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = build_report(run_dir)
    artifact = joblib.load(run_dir / "final_model.joblib")
    patient = {
        "age_decade": 6,
        "gender": "Female",
        "height_cm": 165,
        "weight_kg": 70,
        "indication": artifact["categorical_training_values"]["indication"][0],
        "target_inr": 2.5,
        "diabetes": "No",
        "chf_cardiomyopathy": "No",
        "valve_replacement": "No",
        "amiodarone": "No",
        "enzyme_inducer": "No",
        "smoker": "No",
    }
    input_path = tmp_path / "patient.json"
    input_path.write_text(json.dumps(patient), encoding="utf-8")
    result = predict_patient(run_dir / "final_model.joblib", input_path)

    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "random-CV analysis is an optimism comparator" in report_text
    assert "FeatRanker importances are noncausal" in report_text
    assert "comparator sample sizes are procedure-specific" in report_text
    assert "historical population reference corresponding to 5 mg/day" in report_text
    assert "finite age, height, and weight" in report_text
    assert "[selection frequencies](tables/selection_frequencies.csv)" in report_text
    assert "../manifest.json" not in report_text
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert f'Analysis-code revision: `{run_manifest["git_revision"]}`.' in report_text
    assert "Final-model revision: `separate-final-model-revision`." in report_text
    assert "nested fold-wise ranking procedure" in report_text
    assert "not performance of the static full-cohort refit" in report_text
    assert "Final artifact label: `pharmacogenomic_ranked`" in report_text
    assert "`vkorc1`, `weight_kg`" in report_text
    assert "Final-model source SHA-256: `public-source-checksum`" in report_text
    assert (run_dir / "report" / "tables" / "overall_metrics.csv").exists()
    selection_frequencies = pd.read_csv(
        run_dir / "report" / "tables" / "selection_frequencies.csv"
    )
    assert set(selection_frequencies) == {
        "procedure",
        "candidate_key",
        "statin_included",
        "selection_count",
        "selection_rate",
    }
    assert not (run_dir / "report" / "tables" / "selections.csv").exists()
    assert (run_dir / "report" / "figures" / "observed_vs_predicted.png").exists()
    assert result["weekly_dose_mg"] >= 0
    assert result["average_daily_dose_mg"] == result["weekly_dose_mg"] / 7
    assert result["interval_90_mg_week"][0] >= 0
    assert "not prescribing guidance" in result["warning"].lower()

    interval_metrics = pd.read_csv(run_dir / "report" / "tables" / "interval_metrics.csv")
    overall_intervals = interval_metrics.loc[interval_metrics["scope"].eq("overall")]
    assert set(overall_intervals["procedure"]) == {"clinical_ml", "pharmacogenomic_ml"}
    assert (run_dir / "paired_bootstrap.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "paired_bootstrap.csv" in manifest["output_files"]

    predictions_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    missing_comparator = predictions["procedure"].eq("iwpc_pharmacogenetic")
    predictions.loc[missing_comparator, "y_pred"] = float("nan")
    predictions.loc[missing_comparator, "prediction_status"] = "missing_required_comparator_input"
    predictions.to_csv(predictions_path, index=False)
    assert build_report(run_dir).exists()


def test_build_report_accepts_all_analysis_root(raw_frame, tmp_path):
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    analysis_root = tmp_path / "all-analysis"
    primary_dir = run_primary_frame(
        raw_frame,
        analysis_root / "primary",
        candidates=candidates,
        seed=7,
    )

    report = build_report(analysis_root)

    assert report == primary_dir / "report" / "report.md"
    assert report.exists()


def test_prediction_rejects_forbidden_unknown_and_nonfinite_inputs(
    raw_frame, tmp_path, monkeypatch
):
    monkeypatch.setattr("warfarin_dose.evaluation._best_feature_set", lambda _: "clinical")
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    run_dir = run_primary_frame(raw_frame, tmp_path / "run", candidates=candidates, seed=7)
    input_path = tmp_path / "patient.json"

    input_path.write_text(json.dumps({"race": "not an input"}), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        predict_patient(run_dir / "final_model.joblib", input_path)

    input_path.write_text(json.dumps({"unexpected": "value"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        predict_patient(run_dir / "final_model.joblib", input_path)

    input_path.write_text('{"weight_kg": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        predict_patient(run_dir / "final_model.joblib", input_path)

    artifact = joblib.load(run_dir / "final_model.joblib")
    assert set(artifact["categorical_training_values"]) == {
        name
        for name in artifact["feature_columns"]
        if name not in {"age_decade", "height_cm", "weight_kg", "target_inr"}
    }

    input_path.write_text(json.dumps({"gender": "not-a-category"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unseen categorical input"):
        predict_patient(run_dir / "final_model.joblib", input_path)

    unused = next(
        name
        for name in ["statin", "vkorc1", "cyp2c9_group"]
        if name not in artifact["feature_columns"]
    )
    input_path.write_text(json.dumps({unused: "Unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        predict_patient(run_dir / "final_model.joblib", input_path)
