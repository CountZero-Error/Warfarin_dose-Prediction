from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age_decade", "height_cm", "weight_kg", "target_inr"]
DEMOGRAPHIC_FEATURES = ["age_decade", "gender"]
ANTHROPOMETRIC_FEATURES = ["height_cm", "weight_kg"]
CONDITION_FEATURES = [
    "indication",
    "target_inr",
    "diabetes",
    "chf_cardiomyopathy",
    "valve_replacement",
]
MEDICATION_FEATURES = ["amiodarone", "enzyme_inducer", "smoker"]
PGX_FEATURES = ["cyp2c9_group", "vkorc1"]

CLINICAL_FEATURES = (
    DEMOGRAPHIC_FEATURES + ANTHROPOMETRIC_FEATURES + CONDITION_FEATURES + MEDICATION_FEATURES
)
PHARMACOGENOMIC_FEATURES = CLINICAL_FEATURES + PGX_FEATURES
FORBIDDEN = {
    "weekly_dose_mg",
    "Therapeutic Dose of Warfarin",
    "INR on Reported Therapeutic Dose of Warfarin",
    "Subject Reached Stable Dose of Warfarin",
    "PharmGKB Subject ID",
    "PharmGKB Sample ID",
    "Project Site",
    "site",
    "row_key",
    "patient_key",
    "race",
    "Comments regarding Project Site Dataset",
}
STATIN_SOURCE_COLUMNS = [
    "Simvastatin (Zocor)",
    "Atorvastatin (Lipitor)",
    "Fluvastatin (Lescol)",
    "Lovastatin (Mevacor)",
    "Pravastatin (Pravachol)",
    "Rosuvastatin (Crestor)",
    "Cerivastatin (Baycol)",
]
INDUCER_SOURCE_COLUMNS = [
    "Carbamazepine (Tegretol)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
]


def parse_age_decade(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"^\s*(\d{1,3})", str(value))
    return float(int(match.group(1)) // 10) if match else np.nan


def normalize_binary(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"1", "1.0", "yes", "true", "present"}:
        return "Yes"
    if text in {"0", "0.0", "no", "false", "not present"}:
        return "No"
    return "Unknown"


def combine_binary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    normalized = frame[list(columns)].map(normalize_binary)
    return pd.Series(
        np.where(
            normalized.eq("Yes").any(axis=1),
            "Yes",
            np.where(normalized.eq("No").any(axis=1), "No", "Unknown"),
        ),
        index=frame.index,
    )


def normalize_cyp2c9(value: object) -> tuple[str, str]:
    if pd.isna(value) or str(value).replace(" ", "").lower() in {"", "unknown", "nan", "nocall"}:
        return "Unknown", "Unknown"
    diplotype = str(value).replace(" ", "")
    diplotype = "/".join(sorted(diplotype.split("/")))
    if diplotype == "*1/*1":
        group = "Normal"
    elif diplotype in {"*1/*2", "*1/*3", "*1/*5", "*1/*13", "*1/*14"}:
        group = "One decreased"
    elif diplotype in {"*2/*2", "*2/*3", "*3/*3"}:
        group = "Two decreased"
    else:
        group = "Other observed"
    return diplotype, group


def normalize_vkorc1(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    alleles = str(value).replace(" ", "").upper().split("/")
    if len(alleles) != 2 or any(allele not in {"A", "G"} for allele in alleles):
        return "Unknown"
    return "/".join(sorted(alleles))


def parse_target_inr(value: object, estimated_range: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(numeric):
        return float(numeric)
    numbers = re.findall(r"\d+(?:\.\d+)?", "" if pd.isna(estimated_range) else str(estimated_range))
    return float(np.mean([float(item) for item in numbers[:2]])) if numbers else np.nan


def statin_gate(raw: pd.DataFrame) -> tuple[pd.Series, bool, str]:
    values = raw[STATIN_SOURCE_COLUMNS].map(normalize_binary)
    composite = combine_binary(raw, STATIN_SOURCE_COLUMNS)
    source_nonmissing = raw[STATIN_SOURCE_COLUMNS].notna()
    observed = int(source_nonmissing.sum().sum())
    mapped = int((values.ne("Unknown") & source_nonmissing).sum().sum())
    nonmissing_fraction = float(composite.ne("Unknown").mean())
    mapping_fraction = mapped / observed if observed else 0.0
    known = composite[composite.ne("Unknown")]
    prevalence = known.value_counts(normalize=True)
    class_ok = bool({"Yes", "No"}.issubset(prevalence.index) and prevalence.min() >= 0.01)
    include = bool(nonmissing_fraction >= 0.50 and mapping_fraction >= 0.95 and class_ok)
    reason = (
        "included"
        if include
        else (
            "excluded: "
            f"nonmissing={nonmissing_fraction:.3f}, mapping={mapping_fraction:.3f}, "
            f"both_classes_1pct={class_ok}"
        )
    )
    return composite, include, reason


def build_feature_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    statin, include_statin, statin_reason = statin_gate(raw)
    cyp = raw["CYP2C9 consensus"].map(normalize_cyp2c9)
    frame = pd.DataFrame(index=raw.index)
    for passthrough in ["row_key", "patient_key", "site", "weekly_dose_mg"]:
        if passthrough in raw:
            frame[passthrough] = raw[passthrough]
    frame["age_decade"] = raw["Age"].map(parse_age_decade)
    frame["gender"] = raw["Gender"].fillna("Unknown").astype(str).str.strip().str.title()
    frame["height_cm"] = pd.to_numeric(raw["Height (cm)"], errors="coerce")
    frame["weight_kg"] = pd.to_numeric(raw["Weight (kg)"], errors="coerce")
    frame["indication"] = (
        raw["Indication for Warfarin Treatment"].fillna("Unknown").astype(str).str.strip()
    )
    frame["target_inr"] = [
        parse_target_inr(value, estimated)
        for value, estimated in zip(
            raw["Target INR"], raw["Estimated Target INR Range Based on Indication"], strict=True
        )
    ]
    frame["diabetes"] = raw["Diabetes"].map(normalize_binary)
    frame["chf_cardiomyopathy"] = raw["Congestive Heart Failure and/or Cardiomyopathy"].map(
        normalize_binary
    )
    frame["valve_replacement"] = raw["Valve Replacement"].map(normalize_binary)
    frame["amiodarone"] = raw["Amiodarone (Cordarone)"].map(normalize_binary)
    frame["enzyme_inducer"] = combine_binary(raw, INDUCER_SOURCE_COLUMNS)
    frame["smoker"] = raw["Current Smoker"].map(normalize_binary)
    frame["statin"] = statin
    frame["cyp2c9_diplotype"] = [item[0] for item in cyp]
    frame["cyp2c9_group"] = [item[1] for item in cyp]
    frame["vkorc1"] = raw["VKORC1 -1639 consensus"].map(normalize_vkorc1)
    frame["race"] = raw["Race (OMB)"].fillna("Unknown").astype(str).str.strip()
    return frame, {"include_statin": include_statin, "statin_reason": statin_reason}


def feature_columns(
    name: Literal["clinical", "pharmacogenomic"], include_statin: bool
) -> list[str]:
    columns = list(CLINICAL_FEATURES if name == "clinical" else PHARMACOGENOMIC_FEATURES)
    if include_statin:
        columns.append("statin")
    return columns


def select_feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    forbidden = sorted(set(columns) & FORBIDDEN)
    missing = sorted(set(columns) - set(frame.columns))
    if forbidden:
        raise ValueError(f"forbidden predictor columns: {forbidden}")
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate feature names")
    if missing:
        raise ValueError(f"missing canonical feature columns: {missing}")
    return frame.loc[:, list(columns)].copy()


def _combine_name(feature: str, category: object) -> str:
    return f"{feature}={category}"


def make_preprocessor(columns: Sequence[str], scale_numeric: bool) -> ColumnTransformer:
    numeric = [name for name in columns if name in NUMERIC_FEATURES]
    categorical = [name for name in columns if name not in NUMERIC_FEATURES]
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    feature_name_combiner=_combine_name,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("missingindicator", MissingIndicator(features="all", sparse=False), numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        verbose_feature_names_out=False,
    )


def semantic_feature_groups(encoded_names: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in encoded_names:
        source = name.removeprefix("missingindicator_").split("=", 1)[0]
        groups[source].append(name)
    return dict(groups)
