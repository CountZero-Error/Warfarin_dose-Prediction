from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://api.pharmgkb.org/v1/download/submission/553247439"
SOURCE_SHA256 = "0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d"
SOURCE_SIZE = 5_083_136
RAW_PATH = Path("data/raw/PS206767-553247439.xls")
SOURCE_SHEET = "Subject Data"
DOWNLOAD_USER_AGENT = "warfarin-dose-research/0.1"

ID_COLUMNS = ["PharmGKB Subject ID", "PharmGKB Sample ID"]
SITE_COLUMN = "Project Site"
STABLE_COLUMN = "Subject Reached Stable Dose of Warfarin"
TARGET_COLUMN = "Therapeutic Dose of Warfarin"
POST_TREATMENT_INR = "INR on Reported Therapeutic Dose of Warfarin"
CYP2C9_COLUMN = "CYP2C9 consensus"
VKORC1_COLUMN = "VKORC1 -1639 consensus"

REQUIRED_COLUMNS = [
    *ID_COLUMNS,
    SITE_COLUMN,
    "Gender",
    "Race (Reported)",
    "Race (OMB)",
    "Age",
    "Height (cm)",
    "Weight (kg)",
    "Indication for Warfarin Treatment",
    "Diabetes",
    "Congestive Heart Failure and/or Cardiomyopathy",
    "Valve Replacement",
    "Simvastatin (Zocor)",
    "Atorvastatin (Lipitor)",
    "Fluvastatin (Lescol)",
    "Lovastatin (Mevacor)",
    "Pravastatin (Pravachol)",
    "Rosuvastatin (Crestor)",
    "Cerivastatin (Baycol)",
    "Amiodarone (Cordarone)",
    "Carbamazepine (Tegretol)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
    "Target INR",
    "Estimated Target INR Range Based on Indication",
    STABLE_COLUMN,
    TARGET_COLUMN,
    POST_TREATMENT_INR,
    "Current Smoker",
    CYP2C9_COLUMN,
    VKORC1_COLUMN,
    "Comments regarding Project Site Dataset",
]

LEGACY_COMPLETE_CASE_COLUMNS = [
    "Height (cm)",
    "Weight (kg)",
    POST_TREATMENT_INR,
    "Current Smoker",
    "Diabetes",
    "Amiodarone (Cordarone)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
    TARGET_COLUMN,
    CYP2C9_COLUMN,
    VKORC1_COLUMN,
    "Gender",
    "Age",
]


@dataclass(frozen=True)
class Cohort:
    data: pd.DataFrame
    exclusions: pd.DataFrame
    flow: pd.DataFrame
    issues: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(raw: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    duplicates = raw.columns[raw.columns.duplicated()].tolist()
    if missing:
        raise ValueError(f"missing required IWPC columns: {missing}")
    if duplicates:
        raise ValueError(f"duplicate IWPC column names: {duplicates}")


def read_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("raw IWPC checksum does not match the reviewed dataset version")
    raw = pd.read_excel(path, sheet_name=SOURCE_SHEET, engine="xlrd")
    validate_schema(raw)
    return raw


def _row_keys(raw: pd.DataFrame) -> pd.Series:
    values = raw[ID_COLUMNS].fillna("Unknown").astype(str)
    return pd.Series(
        [
            hashlib.sha256(f"{subject}|{sample}|{index}".encode()).hexdigest()[:20]
            for index, (subject, sample) in enumerate(values.itertuples(index=False, name=None))
        ],
        index=raw.index,
        name="row_key",
    )


def _patient_keys(raw: pd.DataFrame) -> pd.Series:
    values = raw[ID_COLUMNS].fillna("Unknown").astype(str)
    return pd.Series(
        [
            hashlib.sha256(f"{subject}|{sample}".encode()).hexdigest()[:20]
            for subject, sample in values.itertuples(index=False, name=None)
        ],
        index=raw.index,
        name="patient_key",
    )


def prepare_cohort(raw: pd.DataFrame) -> Cohort:
    validate_schema(raw)
    target = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    stable = pd.to_numeric(raw[STABLE_COLUMN], errors="coerce").eq(1)
    finite_positive = pd.Series(np.isfinite(target) & target.gt(0), index=raw.index)
    normalized_site = raw[SITE_COLUMN].astype("string").str.strip()
    has_site = normalized_site.notna() & normalized_site.ne("")
    reason = np.select(
        [~stable, ~finite_positive, ~has_site],
        ["not_stable", "invalid_target", "missing_site"],
        default="eligible",
    )
    keys = _row_keys(raw)
    patient_keys = _patient_keys(raw)
    eligible = reason == "eligible"
    data = raw.loc[eligible].copy()
    data.insert(0, "row_key", keys.loc[eligible].to_numpy())
    data.insert(1, "patient_key", patient_keys.loc[eligible].to_numpy())
    data["site"] = normalized_site.loc[eligible].astype(str)
    data["weekly_dose_mg"] = target.loc[eligible].astype(float)
    exclusions = pd.DataFrame({"row_key": keys.loc[~eligible], "reason": reason[~eligible]})
    flow = pd.DataFrame(
        {
            "stage": ["source", "eligible", "excluded"],
            "count": [len(raw), int(eligible.sum()), int((~eligible).sum())],
        }
    )
    duplicate_ids = int(raw.duplicated(ID_COLUMNS, keep=False).sum())
    conflicting_targets = int(
        raw.assign(_target=target)
        .groupby(ID_COLUMNS, dropna=False)["_target"]
        .nunique(dropna=True)
        .gt(1)
        .sum()
    )
    height = pd.to_numeric(raw["Height (cm)"], errors="coerce")
    weight = pd.to_numeric(raw["Weight (kg)"], errors="coerce")
    issues = pd.DataFrame(
        {
            "issue": [
                "duplicate_subject_sample_rows",
                "conflicting_duplicate_target",
                "impossible_height",
                "impossible_weight",
                "dose_above_200_mg_week",
            ],
            "count": [
                duplicate_ids,
                conflicting_targets,
                int(((height <= 0) | (height > 250)).sum()),
                int(((weight <= 0) | (weight > 500)).sum()),
                int((target > 200).sum()),
            ],
        }
    )
    return Cohort(data=data, exclusions=exclusions, flow=flow, issues=issues)


def _label_counts(values: pd.Series) -> pd.DataFrame:
    labels = values.fillna("Missing").astype(str).str.strip()
    return labels.value_counts().rename_axis("label").reset_index(name="count")


def _invalid_cyp2c9_labels(values: pd.Series) -> pd.DataFrame:
    observed = values.dropna().astype(str).str.strip()
    labels = observed.str.replace(" ", "", regex=False)
    invalid = observed[
        ~labels.str.lower().isin({"", "unknown", "nan", "no call"})
        & ~labels.str.fullmatch(r"\*\d+/\*\d+")
    ]
    return _label_counts(invalid)


def _invalid_vkorc1_labels(values: pd.Series) -> pd.DataFrame:
    observed = values.dropna().astype(str).str.strip()
    labels = observed.str.replace(" ", "", regex=False).str.upper()
    invalid = observed[
        ~labels.str.lower().isin({"", "unknown", "nan", "no call"})
        & ~labels.str.fullmatch(r"[AG]/[AG]")
    ]
    return _label_counts(invalid)


def build_audit(raw: pd.DataFrame, cohort: Cohort) -> dict[str, pd.DataFrame]:
    from warfarin_dose.features import build_feature_frame

    eligible = cohort.data
    features, feature_metadata = build_feature_frame(eligible)
    missingness = (
        eligible[REQUIRED_COLUMNS]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "source_feature"})
    )
    missing_by_site = (
        eligible.groupby("site")[REQUIRED_COLUMNS]
        .apply(lambda frame: frame.isna().mean())
        .stack()
        .rename("missing_fraction")
        .reset_index()
    )
    missing_by_site.columns = ["site", "source_feature", "missing_fraction"]
    site_summary = eligible.groupby("site")["weekly_dose_mg"].agg(
        n="size", mean="mean", median="median", minimum="min", maximum="max"
    ).reset_index()
    distributions = eligible[["weekly_dose_mg", "Height (cm)", "Weight (kg)"]].describe().T
    legacy_complete = eligible[LEGACY_COMPLETE_CASE_COLUMNS].notna().all(axis=1)
    population = pd.DataFrame(
        {
            "cohort": ["eligible", "legacy_complete_case"],
            "n": [len(eligible), int(legacy_complete.sum())],
        }
    )
    age_parse_failures = int(eligible["Age"].notna().sum() - features["age_decade"].notna().sum())
    feature_quality = pd.DataFrame(
        {
            "measure": ["statin_decision", "statin_reason", "age_parse_failures"],
            "value": [
                "included" if feature_metadata["include_statin"] else "excluded",
                feature_metadata["statin_reason"],
                age_parse_failures,
            ],
        }
    )
    cyp2c9_source = eligible[CYP2C9_COLUMN].fillna("Missing").astype(str).str.strip()
    vkorc1_source = eligible[VKORC1_COLUMN].fillna("Missing").astype(str).str.strip()
    genotype_labels = pd.concat(
        [
            pd.DataFrame(
                {
                    "source_feature": CYP2C9_COLUMN,
                    "source_value": cyp2c9_source,
                    "normalized_value": features["cyp2c9_diplotype"],
                    "normalized_group": features["cyp2c9_group"],
                }
            ),
            pd.DataFrame(
                {
                    "source_feature": VKORC1_COLUMN,
                    "source_value": vkorc1_source,
                    "normalized_value": features["vkorc1"],
                    "normalized_group": features["vkorc1"],
                }
            ),
        ],
        ignore_index=True,
    )
    genotype_labels = (
        genotype_labels.value_counts(sort=False)
        .rename("count")
        .reset_index()
        .sort_values(["source_feature", "source_value"], kind="stable")
        .reset_index(drop=True)
    )
    canonical_columns = [
        column
        for column in features.columns
        if column not in {"row_key", "patient_key", "site", "weekly_dose_mg"}
    ]
    feature_missingness = pd.DataFrame(
        {
            "feature": canonical_columns,
            "missing_fraction": [
                float(features[column].isna().mean()) for column in canonical_columns
            ],
            "unknown_count": [
                int(features[column].eq("Unknown").sum()) for column in canonical_columns
            ],
        }
    )
    return {
        "cohort_flow": cohort.flow,
        "exclusions": cohort.exclusions,
        "issues": cohort.issues,
        "missingness": missingness,
        "missingness_by_site": missing_by_site,
        "site_summary": site_summary,
        "distributions": distributions.reset_index(names="measure"),
        "population_comparison": population,
        "cyp2c9_observed_labels": _label_counts(eligible[CYP2C9_COLUMN]),
        "vkorc1_observed_labels": _label_counts(eligible[VKORC1_COLUMN]),
        "cyp2c9_invalid_labels": _invalid_cyp2c9_labels(eligible[CYP2C9_COLUMN]),
        "vkorc1_invalid_labels": _invalid_vkorc1_labels(eligible[VKORC1_COLUMN]),
        "feature_quality": feature_quality,
        "genotype_labels": genotype_labels,
        "feature_missingness": feature_missingness,
    }


def write_audit(
    raw_path: Path = RAW_PATH, output_dir: Path = Path("artifacts/audit")
) -> dict[str, object]:
    raw = read_raw(raw_path)
    cohort = prepare_cohort(raw)
    tables = build_audit(raw, cohort)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    summary = {
        "source_rows": len(raw),
        "eligible_rows": len(cohort.data),
        "sites": int(cohort.data["site"].nunique()),
        "source_sha256": sha256_file(raw_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sections = ["# IWPC Data Audit", "", "Research use only; no prescribing guidance.", ""]
    for name, table in tables.items():
        sections.extend(
            [
                f"## {name.replace('_', ' ').title()}",
                "",
                "```text",
                table.to_string(index=False),
                "```",
                "",
            ]
        )
    (output_dir / "audit.md").write_text("\n".join(sections), encoding="utf-8")
    return summary


def download_data(
    destination: Path = RAW_PATH,
    url: str = SOURCE_URL,
    expected_sha256: str = SOURCE_SHA256,
    expected_size: int | None = SOURCE_SIZE,
) -> dict[str, object]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    resolved_url = url
    request = urllib.request.Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            resolved_url = response.geturl()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        checksum = sha256_file(partial)
        if checksum != expected_sha256:
            raise ValueError(
                f"IWPC checksum changed: expected {expected_sha256}, observed {checksum}. "
                "Review and explicitly update the dataset version before continuing."
            )
        size_bytes = partial.stat().st_size
        if expected_size is not None and size_bytes != expected_size:
            raise ValueError(
                f"IWPC file size changed: expected {expected_size}, observed {size_bytes}"
            )
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    manifest = {
        "source_url": url,
        "resolved_url": resolved_url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "path": str(destination),
        "size_bytes": size_bytes,
        "sha256": expected_sha256,
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
