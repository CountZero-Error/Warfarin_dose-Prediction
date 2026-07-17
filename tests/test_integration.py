import json

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
    assert (run_dir / "report" / "tables" / "overall_metrics.csv").exists()
    assert (run_dir / "report" / "figures" / "observed_vs_predicted.png").exists()
    assert result["weekly_dose_mg"] >= 0
    assert result["average_daily_dose_mg"] == result["weekly_dose_mg"] / 7
    assert result["interval_90_mg_week"][0] >= 0
    assert "not prescribing guidance" in result["warning"].lower()


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
