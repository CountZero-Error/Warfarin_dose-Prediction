import json

import pandas as pd
import pytest

from warfarin_dose.evaluation import run_primary_frame
from warfarin_dose.models import ModelSpec
from warfarin_dose.reporting import build_report, predict_patient


def test_synthetic_run_builds_report_and_safe_prediction(raw_frame, tmp_path):
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    run_dir = run_primary_frame(raw_frame, tmp_path / "run", candidates=candidates, seed=7)
    report = build_report(run_dir)
    patient = {
        "age_decade": 6,
        "gender": "Female",
        "height_cm": 165,
        "weight_kg": 70,
        "indication": "7",
        "target_inr": 2.5,
        "diabetes": "No",
        "chf_cardiomyopathy": "No",
        "valve_replacement": "No",
        "amiodarone": "No",
        "enzyme_inducer": "No",
        "smoker": "No",
        "cyp2c9_group": "Normal",
        "vkorc1": "G/G",
    }
    input_path = tmp_path / "patient.json"
    input_path.write_text(json.dumps(patient), encoding="utf-8")
    result = predict_patient(run_dir / "final_model.joblib", input_path)

    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "random-CV analysis is an optimism comparator" in report_text
    assert "FeatRanker importances are noncausal" in report_text
    assert "comparator sample sizes are procedure-specific" in report_text
    assert "[selections.csv](tables/selections.csv)" in report_text
    assert "../manifest.json" not in report_text
    assert (run_dir / "report" / "tables" / "overall_metrics.csv").exists()
    assert (run_dir / "report" / "tables" / "selections.csv").exists()
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


def test_prediction_rejects_forbidden_unknown_and_nonfinite_inputs(raw_frame, tmp_path):
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
