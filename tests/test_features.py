import numpy as np
import pytest

from warfarin_dose.features import (
    build_feature_frame,
    feature_columns,
    make_preprocessor,
    parse_age_decade,
    select_feature_matrix,
    semantic_feature_groups,
)


def test_age_and_genotype_mapping(raw_frame):
    raw = raw_frame.iloc[:3].copy()
    raw["Age"] = ["50 - 59", "90+", "bad"]
    raw["CYP2C9 consensus"] = ["*1/*1", "*1/*3", np.nan]
    raw["VKORC1 -1639 consensus"] = ["G/A", "A/A", np.nan]

    frame, _ = build_feature_frame(raw)

    assert parse_age_decade("50 - 59") == 5
    assert frame["age_decade"].tolist()[:2] == [5.0, 9.0]
    assert np.isnan(frame.loc[2, "age_decade"])
    assert frame["cyp2c9_group"].tolist() == ["Normal", "One decreased", "Unknown"]
    assert frame["vkorc1"].tolist() == ["A/G", "A/A", "Unknown"]


def test_missing_binary_is_unknown_and_inducer_is_prespecified(raw_frame):
    raw = raw_frame.iloc[:3].copy()
    raw[["Carbamazepine (Tegretol)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin"]] = np.nan
    raw.loc[0, "Phenytoin (Dilantin)"] = 1
    raw.loc[1, ["Carbamazepine (Tegretol)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin"]] = 0

    frame, _ = build_feature_frame(raw)

    assert frame["enzyme_inducer"].tolist() == ["Yes", "No", "Unknown"]
    assert frame.loc[2, "diabetes"] in {"Yes", "No", "Unknown"}


def test_primary_feature_sets_exclude_race_site_and_outcomes(raw_frame):
    frame, metadata = build_feature_frame(raw_frame)
    columns = feature_columns("pharmacogenomic", metadata["include_statin"])
    matrix = select_feature_matrix(frame, columns)

    assert not {"race", "site", "weekly_dose_mg", "row_key"} & set(matrix.columns)
    with pytest.raises(ValueError, match="forbidden predictor"):
        select_feature_matrix(frame, ["height_cm", "weekly_dose_mg"])


def test_fold_local_imputation_and_semantic_groups(raw_frame):
    frame, metadata = build_feature_frame(raw_frame)
    columns = feature_columns("pharmacogenomic", metadata["include_statin"])
    frame.loc[0, "height_cm"] = np.nan
    preprocessor = make_preprocessor(columns, scale_numeric=False).fit(frame.iloc[1:])
    encoded = preprocessor.transform(frame.iloc[:1])
    names = preprocessor.get_feature_names_out().tolist()
    groups = semantic_feature_groups(names)

    assert np.isfinite(encoded).all()
    assert "height_cm" in groups
    assert any(name.startswith("missingindicator_height_cm") for name in groups["height_cm"])
    assert all(name.startswith("cyp2c9_group=") for name in groups["cyp2c9_group"])
