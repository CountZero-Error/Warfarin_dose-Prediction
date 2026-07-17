# Warfarin Dose Research Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible clinical-ML research package that estimates stable therapeutic warfarin dose in mg/week from public IWPC clinical and pharmacogenomic data and evaluates it with leakage-safe site-held-out validation.

**Architecture:** A small `src/warfarin_dose/` package owns data acquisition, cohort construction, feature definitions, models, nested evaluation, and deterministic reporting. Commands use only those package functions; patient-level source data and generated runs remain untracked. Primary learned models use all prespecified features, while FeatRanker, ablations, complete-case analysis, and random-CV are explicitly secondary experiments.

**Tech Stack:** Python 3.11+, NumPy, pandas, scikit-learn, matplotlib, joblib, xlrd, FeatRanker 0.2.0, pytest, Ruff, setuptools.

## Global Constraints

- The canonical target and every metric, table, saved prediction, and model output use `mg/week`; average `mg/day` is display-only and equals `mg/week / 7`.
- The source is only `https://api.pharmgkb.org/v1/download/submission/553247439`; expected file size is `5,083,136` bytes and SHA-256 is `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`.
- A checksum or required-schema change stops execution; there is no credentialed-data path and no synthetic fallback outside tests.
- Primary validation is leave-one-site-out; inner selection is grouped by site with `GroupKFold(n_splits=min(5, n_training_sites))`; fewer than three training sites is a hard failure.
- Site, identifiers, stable-dose status, post-treatment INR, therapeutic dose, comments, and downstream outcomes never enter learned model matrices.
- Race is used only by exact IWPC comparators and subgroup auditing, never by primary learned models.
- Primary learned analyses use all prespecified clinical or pharmacogenomic features; FeatRanker 0.2.0 is secondary and may select only top-5, top-10, or all semantic feature blocks inside outer-training data.
- Primary metric is MAE in mg/week; secondary metrics are RMSE, R2, PW20, dose-category, site, subgroup, and empirical 90% conformal-interval results.
- Aggregate intervals and paired differences use 2,000 seeded site-cluster bootstrap replicates; subgroup performance is suppressed below 30 outer predictions.
- Research outputs must say they are not prescribing guidance, a medical device, or a substitute for clinician-guided INR monitoring.
- No web application, PyTorch framework, AutoML layer, private data, causal claims, arbitrary upper-dose cap, or silent model/data fallback is added.
- Execution starts in an isolated worktree created from commit `229cb21`; do not alter, clean, reset, or commit the current checkout's unfinished archive migration.

---

## File Map

- `pyproject.toml`: package metadata, pinned FeatRanker dependency, development tools, and command entry point.
- `.gitignore`: raw data, experiment artifacts, fitted models, caches, and local OS files.
- `README.md`: portfolio-facing purpose, reproducibility commands, results boundary, and safety statement.
- `src/warfarin_dose/__init__.py`: version only.
- `src/warfarin_dose/__main__.py`: module entry point.
- `src/warfarin_dose/cli.py`: thin `argparse` command routing.
- `src/warfarin_dose/data.py`: download, fingerprint, workbook schema, cohort, audit, and run-manifest helpers.
- `src/warfarin_dose/features.py`: canonical features, normalization, statin gate, leakage rejection, preprocessing, and semantic groups.
- `src/warfarin_dose/models.py`: fixed/fold benchmarks, exact IWPC equations, learned candidates, target transform, and fitted pipeline creation.
- `src/warfarin_dose/evaluation.py`: split integrity, nested selection, FeatRanker, metrics, bootstrap, conformal intervals, sensitivities, and artifacts.
- `src/warfarin_dose/reporting.py`: tables, figures, manuscript-style report, and inference output.
- `tests/conftest.py`: deterministic multi-site synthetic source-shaped data.
- `tests/test_data.py`: download, schema, cohort, and audit checks.
- `tests/test_features.py`: mappings, missingness, preprocessing, semantic blocks, and leakage checks.
- `tests/test_models.py`: units, published equations, candidate pipelines, and nonnegative predictions.
- `tests/test_evaluation.py`: grouped splits, selection, metrics, bootstrap, conformal, FeatRanker, and prediction coverage.
- `tests/test_integration.py`: synthetic audit-to-report-to-predict workflow.
- `docs/DATA_CARD.md`: source, fields, cohort, missingness, intended use, and limitations.
- `docs/MODEL_CARD.md`: evaluation, subgroup/uncertainty interpretation, intended use, and safety limits.
- `.github/workflows/tests.yml`: lint, test, and build checks without downloading patient data.

### Task 1: Preserve Legacy Work and Add Verified Public-Data Acquisition

**Files:**
- Move: `README.md`, `datasets/`, `feature_importance.ipynb`, `neural_network/`, `old_version/`, `requirements.txt` to matching paths under `archive/`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/warfarin_dose/__init__.py`
- Create: `src/warfarin_dose/__main__.py`
- Create: `src/warfarin_dose/data.py`
- Create: `src/warfarin_dose/cli.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `download_data(destination: Path, url: str = SOURCE_URL, expected_sha256: str = SOURCE_SHA256, expected_size: int | None = SOURCE_SIZE) -> dict[str, object]`
- Produces: `sha256_file(path: Path) -> str`
- Produces: CLI command `python -m warfarin_dose download-data`

- [ ] **Step 1: Create an isolated execution worktree**

Use the `superpowers:using-git-worktrees` skill, branch from `229cb21`, and verify the original dirty checkout is unchanged.

Run:

```bash
rtk git status --short --branch
rtk git rev-parse HEAD
```

Expected: the execution worktree is clean at `229cb21`; the original checkout still contains its pre-existing `RD`/untracked migration state.

- [ ] **Step 2: Move only tracked legacy project content under `archive/`**

Run in the isolated worktree:

```bash
rtk mkdir -p archive
rtk git mv README.md archive/README.md
rtk git mv datasets archive/datasets
rtk git mv feature_importance.ipynb archive/feature_importance.ipynb
rtk git mv neural_network archive/neural_network_legacy
rtk git mv old_version archive/old_version
rtk git mv requirements.txt archive/requirements.txt
```

Remove tracked bytecode if present; do not archive caches:

```bash
rtk git rm -r --ignore-unmatch archive/old_version/UI/__pycache__
```

Expected: all historical source, notebooks, data snapshots, and model files remain available under `archive/`; no archive module is imported by the new package.

- [ ] **Step 3: Add package metadata and ignore rules**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "warfarin-dose-research"
version = "0.1.0"
description = "Site-aware clinical ML research on public IWPC warfarin data"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "featranker==0.2.0",
  "joblib>=1.4",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "pandas>=2.2",
  "scikit-learn>=1.5",
  "xlrd>=2.0.1,<3",
]

[project.optional-dependencies]
dev = ["build>=1.2", "pytest>=8.2", "ruff>=0.5"]

[project.scripts]
warfarin-dose = "warfarin_dose.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Create `.gitignore`:

```gitignore
.DS_Store
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
data/raw/
artifacts/
*.joblib
```

Create `src/warfarin_dose/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write failing checksum and atomic-download tests**

Add to `tests/test_data.py`:

```python
from io import BytesIO

import pytest

from warfarin_dose import data


class Response(BytesIO):
    def geturl(self) -> str:
        return "https://s3.pgkb.org/submission/example.xls"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_verifies_checksum_and_records_manifest(tmp_path, monkeypatch):
    payload = b"public-iwpc-data"
    expected = data.hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *_: Response(payload))

    manifest = data.download_data(
        tmp_path / "raw.xls", expected_sha256=expected, expected_size=len(payload)
    )

    assert (tmp_path / "raw.xls").read_bytes() == payload
    assert manifest["sha256"] == expected
    assert manifest["resolved_url"].endswith("example.xls")
    assert not (tmp_path / "raw.xls.part").exists()


def test_download_removes_partial_file_on_checksum_change(tmp_path, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *_: Response(b"changed"))

    with pytest.raises(ValueError, match="checksum"):
        data.download_data(tmp_path / "raw.xls", expected_sha256="0" * 64)

    assert not (tmp_path / "raw.xls").exists()
    assert not (tmp_path / "raw.xls.part").exists()
```

- [ ] **Step 5: Run the focused tests and confirm failure**

Run:

```bash
rtk conda run -n DL python -m pip install -e '.[dev]'
rtk conda run -n DL python -m pytest tests/test_data.py -v
```

Expected: FAIL because `warfarin_dose.data` and its functions do not exist.

- [ ] **Step 6: Implement verified atomic download**

Create `src/warfarin_dose/data.py` with these acquisition primitives:

```python
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://api.pharmgkb.org/v1/download/submission/553247439"
SOURCE_SHA256 = "0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d"
SOURCE_SIZE = 5_083_136
RAW_PATH = Path("data/raw/PS206767-553247439.xls")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
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
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    manifest = {
        "source_url": url,
        "resolved_url": resolved_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": expected_sha256,
    }
    if expected_size is not None and manifest["size_bytes"] != expected_size:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"IWPC file size changed: expected {expected_size}, observed {manifest['size_bytes']}"
        )
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
```

Keep checksum as the authoritative content identity; the size check provides an additional actionable diagnostic.

- [ ] **Step 7: Add the first thin CLI command**

Create `src/warfarin_dose/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .data import RAW_PATH, download_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warfarin-dose")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data", help="download and verify public IWPC data")
    download.add_argument("--output", type=str, default=str(RAW_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        manifest = download_data(destination=Path(args.output))
        print(f"verified {manifest['sha256']} at {manifest['path']}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
```

Create `src/warfarin_dose/__main__.py`:

```python
from .cli import main

raise SystemExit(main())
```

The CLI test monkeypatches the downloader and uses no network.

- [ ] **Step 8: Verify package, tests, and command help**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_data.py -v
rtk conda run -n DL python -m warfarin_dose --help
rtk conda run -n DL ruff check src tests
```

Expected: focused tests PASS; help lists `download-data`; Ruff reports no errors.

- [ ] **Step 9: Commit the acquisition slice**

```bash
rtk git add archive pyproject.toml .gitignore src tests/test_data.py
rtk git commit -m "feat: add verified IWPC data acquisition"
```

### Task 2: Build the Stable-Dose Cohort and Data Audit

**Files:**
- Modify: `src/warfarin_dose/data.py`
- Modify: `src/warfarin_dose/cli.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: verified `.xls` from `download_data`
- Produces: `read_raw(path: Path) -> pd.DataFrame`
- Produces: `prepare_cohort(raw: pd.DataFrame) -> Cohort`
- Produces: `build_audit(raw: pd.DataFrame, cohort: Cohort) -> dict[str, pd.DataFrame]`
- Produces: `write_audit(raw_path: Path, output_dir: Path) -> dict[str, object]`
- Produces: CLI command `python -m warfarin_dose audit-data`

- [ ] **Step 1: Add a deterministic source-shaped test fixture**

Create `tests/conftest.py`. Import `REQUIRED_COLUMNS`, initialize every required field to `numpy.nan`, then populate six sites with eight rows each. Every row must have unique subject/sample IDs, stable flag `1`, positive weekly dose, age bands, height, weight, clinical fields, and canonical genotype consensus values. Use this dose generator so tests have signal without reproducing real patients:

```python
dose = 10.0 + 2.0 * site + 0.1 * weight - 3.0 * (vkorc1 == "A/A")
```

Cycle CYP2C9 through `*1/*1`, `*1/*2`, and `*1/*3`; cycle VKORC1 through `G/G`, `A/G`, and `A/A`. The fixture returns the 48-row DataFrame and contains no source patient values.

- [ ] **Step 2: Write failing cohort tests**

Add to `tests/test_data.py`:

```python
import numpy as np


def test_cohort_uses_stable_positive_weekly_dose_and_site(raw_frame):
    raw = raw_frame.copy()
    raw.loc[0, "Subject Reached Stable Dose of Warfarin"] = 0
    raw.loc[1, "Therapeutic Dose of Warfarin"] = -1
    raw.loc[2, "Project Site"] = np.nan

    cohort = data.prepare_cohort(raw)

    assert len(cohort.data) == len(raw) - 3
    assert set(cohort.exclusions["reason"]) == {"not_stable", "invalid_target", "missing_site"}
    assert cohort.data["weekly_dose_mg"].gt(0).all()
    assert cohort.data["row_key"].is_unique
    assert "PharmGKB Subject ID" not in cohort.exclusions.columns


def test_unusual_positive_dose_is_audited_not_removed(raw_frame):
    raw = raw_frame.copy()
    raw.loc[0, "Therapeutic Dose of Warfarin"] = 315.0

    cohort = data.prepare_cohort(raw)

    assert 315.0 in cohort.data["weekly_dose_mg"].to_numpy()
    assert cohort.issues.set_index("issue").loc["dose_above_200_mg_week", "count"] == 1


def test_required_schema_change_stops(raw_frame):
    with pytest.raises(ValueError, match="missing required IWPC columns"):
        data.prepare_cohort(raw_frame.drop(columns=["Project Site"]))
```

- [ ] **Step 3: Run tests and confirm missing cohort API**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_data.py -v
```

Expected: FAIL because `REQUIRED_COLUMNS`, `Cohort`, and cohort functions do not exist.

- [ ] **Step 4: Declare and validate the exact source schema**

Add a frozen dataclass and exact column constants to `data.py`:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    "Height (cm)", "Weight (kg)", POST_TREATMENT_INR, "Current Smoker", "Diabetes",
    "Amiodarone (Cordarone)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin",
    TARGET_COLUMN, CYP2C9_COLUMN, VKORC1_COLUMN, "Gender", "Age",
]


@dataclass(frozen=True)
class Cohort:
    data: pd.DataFrame
    exclusions: pd.DataFrame
    flow: pd.DataFrame
    issues: pd.DataFrame


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
    raw = pd.read_excel(path, engine="xlrd")
    validate_schema(raw)
    return raw
```

The workbook may contain additional reviewed source columns; exact required columns must remain present.

- [ ] **Step 5: Implement exclusive cohort reasons and non-excluding issue counts**

Implement `prepare_cohort` with this precedence:

```python
def _row_keys(raw: pd.DataFrame) -> pd.Series:
    values = raw[ID_COLUMNS].fillna("Unknown").astype(str)
    return pd.Series(
        [hashlib.sha256(f"{a}|{b}|{i}".encode()).hexdigest()[:20]
         for i, (a, b) in enumerate(values.itertuples(index=False, name=None))],
        index=raw.index,
        name="row_key",
    )


def _patient_keys(raw: pd.DataFrame) -> pd.Series:
    values = raw[ID_COLUMNS].fillna("Unknown").astype(str)
    return pd.Series(
        [hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:20]
         for a, b in values.itertuples(index=False, name=None)],
        index=raw.index,
        name="patient_key",
    )


def prepare_cohort(raw: pd.DataFrame) -> Cohort:
    validate_schema(raw)
    target = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    stable = pd.to_numeric(raw[STABLE_COLUMN], errors="coerce").eq(1)
    finite_positive = pd.Series(np.isfinite(target) & target.gt(0), index=raw.index)
    has_site = raw[SITE_COLUMN].notna()
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
    data["site"] = data[SITE_COLUMN].astype(str)
    data["weekly_dose_mg"] = target.loc[eligible].astype(float)
    exclusions = pd.DataFrame({"row_key": keys.loc[~eligible], "reason": reason[~eligible]})
    flow = pd.DataFrame(
        {"stage": ["source", "eligible", "excluded"],
         "count": [len(raw), int(eligible.sum()), int((~eligible).sum())]}
    )
    duplicate_ids = int(raw.duplicated(ID_COLUMNS, keep=False).sum())
    conflicting_targets = int(
        raw.assign(_target=target).groupby(ID_COLUMNS, dropna=False)["_target"]
        .nunique(dropna=True).gt(1).sum()
    )
    height = pd.to_numeric(raw["Height (cm)"], errors="coerce")
    weight = pd.to_numeric(raw["Weight (kg)"], errors="coerce")
    issues = pd.DataFrame(
        {
            "issue": [
                "duplicate_subject_sample_rows", "conflicting_duplicate_target",
                "impossible_height", "impossible_weight", "dose_above_200_mg_week",
            ],
            "count": [
                duplicate_ids, conflicting_targets,
                int(((height <= 0) | (height > 250)).sum()),
                int(((weight <= 0) | (weight > 500)).sum()),
                int((target > 200).sum()),
            ],
        }
    )
    return Cohort(data=data, exclusions=exclusions, flow=flow, issues=issues)
```

Do not exclude issue rows beyond the three eligibility rules.

- [ ] **Step 6: Build machine-readable and human-readable audit artifacts**

Implement:

```python
def build_audit(raw: pd.DataFrame, cohort: Cohort) -> dict[str, pd.DataFrame]:
    eligible = cohort.data
    missingness = (
        eligible[REQUIRED_COLUMNS].isna().mean().rename("missing_fraction").reset_index()
        .rename(columns={"index": "source_feature"})
    )
    missing_by_site = (
        eligible.groupby("site")[REQUIRED_COLUMNS].apply(lambda frame: frame.isna().mean())
        .stack().rename("missing_fraction").reset_index(names=["site", "source_feature"])
    )
    site_summary = eligible.groupby("site")["weekly_dose_mg"].agg(
        n="size", mean="mean", median="median", minimum="min", maximum="max"
    ).reset_index()
    distributions = eligible[["weekly_dose_mg", "Height (cm)", "Weight (kg)"]].describe().T
    legacy_complete = eligible[LEGACY_COMPLETE_CASE_COLUMNS].notna().all(axis=1)
    population = pd.DataFrame(
        {"cohort": ["eligible", "legacy_complete_case"],
         "n": [len(eligible), int(legacy_complete.sum())]}
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
    }


def write_audit(raw_path: Path = RAW_PATH, output_dir: Path = Path("artifacts/audit")) -> dict[str, object]:
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
        sections.extend([f"## {name.replace('_', ' ').title()}", "", "```text", table.to_string(index=False), "```", ""])
    (output_dir / "audit.md").write_text("\n".join(sections), encoding="utf-8")
    return summary
```

Add genotype-label tables for observed CYP2C9/VKORC1 values and invalid-label counts; include age parsing failures after Task 3 exposes `parse_age_decade`.

- [ ] **Step 7: Add `audit-data` CLI routing**

Add `--input` defaulting to `RAW_PATH` and `--output` defaulting to `artifacts/audit`. Route to `write_audit(Path(args.input), Path(args.output))` and print only source/eligible/site counts and output path.

- [ ] **Step 8: Verify audit behavior**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_data.py -v
rtk conda run -n DL ruff check src tests
```

Expected: cohort, schema, unusual-dose, and audit tests PASS; no patient identifiers appear in exclusion/audit outputs.

- [ ] **Step 9: Commit cohort and audit**

```bash
rtk git add src/warfarin_dose/data.py src/warfarin_dose/cli.py tests
rtk git commit -m "feat: add stable-dose cohort and audit"
```

### Task 3: Define Canonical Clinical and Pharmacogenomic Features

**Files:**
- Create: `src/warfarin_dose/features.py`
- Modify: `src/warfarin_dose/data.py`
- Create: `tests/test_features.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: `Cohort.data`
- Produces: `build_feature_frame(cohort_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]`
- Produces: `feature_columns(name: Literal["clinical", "pharmacogenomic"], include_statin: bool) -> list[str]`
- Produces: `select_feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame`
- Produces: `make_preprocessor(columns: Sequence[str], scale_numeric: bool) -> ColumnTransformer`
- Produces: `semantic_feature_groups(encoded_names: Sequence[str]) -> dict[str, list[str]]`

- [ ] **Step 1: Write failing mapping, leakage, and preprocessing tests**

Create `tests/test_features.py`:

```python
import numpy as np
import pandas as pd
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
```

- [ ] **Step 2: Run tests and confirm the feature module is absent**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_features.py -v
```

Expected: FAIL on import of `warfarin_dose.features`.

- [ ] **Step 3: Implement source normalization and exact feature sets**

Create `features.py` with these constants and normalizers:

```python
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age_decade", "height_cm", "weight_kg", "target_inr"]
DEMOGRAPHIC_FEATURES = ["age_decade", "gender"]
ANTHROPOMETRIC_FEATURES = ["height_cm", "weight_kg"]
CONDITION_FEATURES = ["indication", "target_inr", "diabetes", "chf_cardiomyopathy", "valve_replacement"]
MEDICATION_FEATURES = ["amiodarone", "enzyme_inducer", "smoker"]
PGX_FEATURES = ["cyp2c9_group", "vkorc1"]

CLINICAL_FEATURES = DEMOGRAPHIC_FEATURES + ANTHROPOMETRIC_FEATURES + CONDITION_FEATURES + MEDICATION_FEATURES
PHARMACOGENOMIC_FEATURES = CLINICAL_FEATURES + PGX_FEATURES
FORBIDDEN = {
    "weekly_dose_mg", "Therapeutic Dose of Warfarin",
    "INR on Reported Therapeutic Dose of Warfarin",
    "Subject Reached Stable Dose of Warfarin", "PharmGKB Subject ID",
    "PharmGKB Sample ID", "Project Site", "site", "row_key", "patient_key",
    "Comments regarding Project Site Dataset",
}
STATIN_SOURCE_COLUMNS = [
    "Simvastatin (Zocor)", "Atorvastatin (Lipitor)", "Fluvastatin (Lescol)",
    "Lovastatin (Mevacor)", "Pravastatin (Pravachol)", "Rosuvastatin (Crestor)",
    "Cerivastatin (Baycol)",
]
INDUCER_SOURCE_COLUMNS = [
    "Carbamazepine (Tegretol)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin",
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
    if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "nan"}:
        return "Unknown", "Unknown"
    diplotype = str(value).replace(" ", "")
    alleles = sorted(diplotype.split("/"))
    diplotype = "/".join(alleles)
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
```

The CYP2C9 activity grouping is an analysis grouping, not a CPIC phenotype assignment; retain normalized `cyp2c9_diplotype` separately for the exact published equation.

- [ ] **Step 4: Implement the statin QA gate and canonical frame**

Add:

```python
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
    class_ok = {"Yes", "No"}.issubset(prevalence.index) and prevalence.min() >= 0.01
    include = nonmissing_fraction >= 0.50 and mapping_fraction >= 0.95 and class_ok
    reason = (
        "included"
        if include
        else f"excluded: nonmissing={nonmissing_fraction:.3f}, mapping={mapping_fraction:.3f}, both_classes_1pct={class_ok}"
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
    frame["indication"] = raw["Indication for Warfarin Treatment"].fillna("Unknown").astype(str).str.strip()
    frame["target_inr"] = [
        parse_target_inr(value, estimated)
        for value, estimated in zip(raw["Target INR"], raw["Estimated Target INR Range Based on Indication"], strict=True)
    ]
    frame["diabetes"] = raw["Diabetes"].map(normalize_binary)
    frame["chf_cardiomyopathy"] = raw["Congestive Heart Failure and/or Cardiomyopathy"].map(normalize_binary)
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
```

- [ ] **Step 5: Implement fold-local preprocessing and semantic encoded groups**

Add:

```python
def _combine_name(feature: str, category: object) -> str:
    return f"{feature}={category}"


def make_preprocessor(columns: Sequence[str], scale_numeric: bool) -> ColumnTransformer:
    numeric = [name for name in columns if name in NUMERIC_FEATURES]
    categorical = [name for name in columns if name not in NUMERIC_FEATURES]
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("encode", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, feature_name_combiner=_combine_name
            )),
        ]
    )
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), numeric), ("categorical", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )


def semantic_feature_groups(encoded_names: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in encoded_names:
        source = name.removeprefix("missingindicator_").split("=", 1)[0]
        groups[source].append(name)
    return dict(groups)
```

Ensure `ColumnTransformer.get_feature_names_out()` is unique and every encoded feature appears in exactly one semantic group.

- [ ] **Step 6: Connect feature QA to the data audit**

In `build_audit`, call `build_feature_frame(cohort.data)` and add:

- `feature_quality.csv` with statin decision/reason and age parse failure count;
- `genotype_labels.csv` with source value, normalized value/group, and count;
- `feature_missingness.csv` from canonical features, including explicit `Unknown` counts.

This import is one-way (`data.py` imports `features.py` inside `build_audit`) to avoid a module import cycle.

- [ ] **Step 7: Run focused feature and data tests**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_features.py tests/test_data.py -v
rtk conda run -n DL ruff check src tests
```

Expected: all mapping, leakage, fold-local preprocessing, semantic-group, and audit tests PASS.

- [ ] **Step 8: Commit canonical features**

```bash
rtk git add src/warfarin_dose/data.py src/warfarin_dose/features.py tests
rtk git commit -m "feat: define leakage-safe clinical features"
```

### Task 4: Implement Published Comparators and Learned Model Candidates

**Files:**
- Create: `src/warfarin_dose/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: canonical feature DataFrames and `make_preprocessor`
- Produces: `iwpc_clinical(frame: pd.DataFrame) -> np.ndarray`
- Produces: `iwpc_pharmacogenetic(frame: pd.DataFrame) -> np.ndarray`
- Produces: `model_candidates(seed: int) -> list[ModelSpec]`
- Produces: `make_model_pipeline(columns: Sequence[str], spec: ModelSpec) -> Pipeline`
- Produces: fitted `DoseRegressor.predict(X) -> nonnegative np.ndarray`

- [ ] **Step 1: Write failing unit and published-equation tests**

Create `tests/test_models.py`:

```python
import numpy as np
import pandas as pd
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
            "age_decade": [5.0], "height_cm": [175.0], "weight_kg": [80.0],
            "race": ["White"], "enzyme_inducer": ["No"], "amiodarone": ["No"],
            "vkorc1": ["G/G"], "cyp2c9_diplotype": ["*1/*1"],
        }
    )

    np.testing.assert_allclose(iwpc_clinical(frame), [34.82888256], rtol=0, atol=1e-8)
    np.testing.assert_allclose(iwpc_pharmacogenetic(frame), [46.83896721], rtol=0, atol=1e-8)
    frame.loc[0, ["vkorc1", "cyp2c9_diplotype"]] = ["A/G", "*1/*2"]
    np.testing.assert_allclose(iwpc_pharmacogenetic(frame), [29.75811601], rtol=0, atol=1e-8)
```

Also add:

```python
def test_iwpc_requires_age_height_weight_but_supports_unknown_genotype():
    frame = pd.DataFrame({
        "age_decade": [np.nan, 5.0], "height_cm": [175.0, 175.0], "weight_kg": [80.0, 80.0],
        "race": ["Unknown", "Unknown"], "enzyme_inducer": ["Unknown", "Unknown"],
        "amiodarone": ["Unknown", "Unknown"], "vkorc1": ["Unknown", "Unknown"],
        "cyp2c9_diplotype": ["Unknown", "Unknown"],
    })
    assert np.isnan(iwpc_pharmacogenetic(frame)[0])
    assert np.isfinite(iwpc_pharmacogenetic(frame)[1])


def test_dose_regressor_clips_before_inverse_square_root():
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, 4.0])
    model = DoseRegressor(DummyRegressor(strategy="constant", constant=-2), target_mode="sqrt").fit(X, y)
    assert model.predict(X).tolist() == [0.0, 0.0]


def test_candidate_grid_is_small_and_deterministic():
    candidates = model_candidates(seed=42)
    assert {item.family for item in candidates} == {"ridge", "elasticnet", "hist_gb", "random_forest", "mlp"}
    assert {item.target_mode for item in candidates} == {"direct", "sqrt"}
    assert len(candidates) == 38
```

- [ ] **Step 2: Run tests and confirm model API is missing**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_models.py -v
```

Expected: FAIL on import of `warfarin_dose.models`.

- [ ] **Step 3: Implement exact IWPC clinical and pharmacogenetic equations**

Use coefficients from the corrected IWPC supplementary appendix and CPIC 2017 supplement:

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

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
        4.0376 - 0.2546 * age + 0.0118 * height + 0.0134 * weight
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
        5.6044 - 0.2614 * age + 0.0087 * height + 0.0128 * weight
        - 0.8677 * vkor.eq("A/G") - 1.6974 * vkor.eq("A/A") - 0.4854 * vkor.eq("Unknown")
        - 0.5211 * cyp.eq("*1/*2") - 0.9357 * cyp.eq("*1/*3")
        - 1.0616 * cyp.eq("*2/*2") - 1.9206 * cyp.eq("*2/*3")
        - 2.3312 * cyp.eq("*3/*3") - 0.2188 * cyp.eq("Unknown")
        + _race_terms(frame["race"], clinical=False)
        + 1.1816 * frame["enzyme_inducer"].eq("Yes").astype(float)
        - 0.5503 * frame["amiodarone"].eq("Yes").astype(float)
    )
    return np.where(valid, np.square(linear.astype(float)), np.nan)
```

Do not impute age, height, or weight for these published comparators. Mark unavailable comparator predictions explicitly and compute paired comparisons only on shared finite rows.

- [ ] **Step 4: Implement target handling as one serializable estimator**

Add:

```python
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
        prediction = np.clip(prediction, 0.0, None)
        prediction = np.square(prediction) if self.target_mode == "sqrt" else prediction
        if not np.isfinite(prediction).all():
            raise ValueError("model produced nonfinite weekly-dose predictions")
        return prediction
```

- [ ] **Step 5: Enumerate the fixed small candidate grid**

Add:

```python
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
    result = []
    for family, family_order, grid in GRIDS:
        for complexity_order, params in enumerate(ParameterGrid(grid)):
            for target_mode in ("direct", "sqrt"):
                result.append(ModelSpec(family, params, target_mode, family_order, complexity_order))
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
            early_stopping=True, max_iter=500, random_state=seed, learning_rate_init=0.001,
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
```

The grid has 19 hyperparameter configurations and two target modes, for 38 candidates. Preserve the family order Ridge, Elastic Net, histogram gradient boosting, random forest, MLP; direct target precedes square root within ties.

- [ ] **Step 6: Run focused model tests**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_models.py -v
rtk conda run -n DL ruff check src tests
```

Expected: published worked examples, unit conversion, target clipping, and exact candidate-count tests PASS.

- [ ] **Step 7: Commit models**

```bash
rtk git add src/warfarin_dose/models.py tests/test_models.py
rtk git commit -m "feat: add warfarin model comparators"
```

### Task 5: Add Split, Metric, Bootstrap, and Conformal Primitives

**Files:**
- Create: `src/warfarin_dose/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `site_outer_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]`
- Produces: `inner_site_splits(sites: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]`
- Produces: `regression_metrics(y_true, y_pred) -> dict[str, float]`
- Produces: `conformal_quantile(residuals, coverage: float = 0.90) -> float`
- Produces: `cluster_bootstrap(predictions: pd.DataFrame, iterations: int = 2000, seed: int = 20260717) -> pd.DataFrame`

- [ ] **Step 1: Write failing split and statistical tests**

Create `tests/test_evaluation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from warfarin_dose.evaluation import (
    cluster_bootstrap,
    conformal_interval,
    conformal_quantile,
    inner_site_splits,
    regression_metrics,
    site_outer_splits,
)


def test_site_splits_are_disjoint_and_cover_each_row_once():
    frame = pd.DataFrame({
        "site": np.repeat(["a", "b", "c", "d"], 3),
        "patient_key": [f"patient-{index}" for index in range(12)],
    })
    seen = np.zeros(len(frame), dtype=int)
    for train, test in site_outer_splits(frame):
        assert set(frame.iloc[train]["site"]).isdisjoint(frame.iloc[test]["site"])
        seen[test] += 1
    assert seen.tolist() == [1] * len(frame)


def test_inner_site_splits_require_three_sites():
    with pytest.raises(ValueError, match="at least three"):
        inner_site_splits(["a", "a", "b", "b"])


def test_metrics_and_dose_categories():
    metrics = regression_metrics(np.array([10.0, 35.0, 60.0]), np.array([12.0, 28.0, 66.0]))
    assert metrics["mae_mg_week"] == 5.0
    assert metrics["rmse_mg_week"] == pytest.approx(np.sqrt(89 / 3))
    assert metrics["pw20"] == pytest.approx(2 / 3)


def test_finite_sample_conformal_quantile_and_nonnegative_lower_bound():
    assert conformal_quantile([1, 2, 3, 4], coverage=0.80) == 4
    lower, upper = conformal_interval(np.array([2.0]), radius=4.0)
    assert lower.tolist() == [0.0]
    assert upper.tolist() == [6.0]


def test_site_cluster_bootstrap_is_seeded():
    predictions = pd.DataFrame({
        "site": ["a", "a", "b", "b"], "y_true": [10.0, 20.0, 30.0, 40.0],
        "y_pred": [11.0, 18.0, 33.0, 36.0],
    })
    first = cluster_bootstrap(predictions, iterations=20, seed=7)
    second = cluster_bootstrap(predictions, iterations=20, seed=7)
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Run tests and confirm evaluation module is absent**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement strict grouped splits**

Create `evaluation.py`:

```python
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

DEFAULT_SEED = 20260717
BOOTSTRAP_ITERATIONS = 2_000


def _validate_split(
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    patient_keys: np.ndarray | None = None,
) -> None:
    if set(train) & set(test):
        raise ValueError("overlapping train/test row positions")
    if set(groups[train]) & set(groups[test]):
        raise ValueError("overlapping train/test sites")
    if patient_keys is not None and set(patient_keys[train]) & set(patient_keys[test]):
        raise ValueError("overlapping train/test patients")


def site_outer_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["site"].astype(str).to_numpy()
    patient_keys = frame["patient_key"].astype(str).to_numpy()
    splits = list(LeaveOneGroupOut().split(frame, groups=groups))
    coverage = np.zeros(len(frame), dtype=int)
    for train, test in splits:
        _validate_split(train, test, groups, patient_keys)
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ValueError("every eligible row must have exactly one outer site fold")
    return splits


def inner_site_splits(sites: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(sites, dtype=str)
    n_sites = len(np.unique(groups))
    if n_sites < 3:
        raise ValueError("inner grouped validation requires at least three training sites")
    splits = list(GroupKFold(n_splits=min(5, n_sites)).split(np.zeros(len(groups)), groups=groups))
    for train, validation in splits:
        _validate_split(train, validation, groups)
    return splits
```

- [ ] **Step 4: Implement finite metrics and dose categories**

Add:

```python
def dose_category(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values <= 21, "low", np.where(values >= 49, "high", "intermediate"))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require same-shaped finite truth and predictions")
    if len(truth) == 0:
        raise ValueError("metrics require at least one prediction")
    return {
        "n": int(len(truth)),
        "mae_mg_week": float(mean_absolute_error(truth, prediction)),
        "rmse_mg_week": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "pw20": float(np.mean(np.abs(prediction - truth) <= 0.20 * truth)),
    }
```

For category/site/subgroup outputs, call this function per group only when `n >= 30`; otherwise save `n` and null metric fields. Aggregate evaluation must always have finite MAE/RMSE/PW20.

- [ ] **Step 5: Implement finite-sample empirical conformal intervals**

Add:

```python
def conformal_quantile(residuals: Sequence[float], coverage: float = 0.90) -> float:
    values = np.asarray(residuals, dtype=float)
    if not 0 < coverage < 1 or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("conformal residuals and coverage must be finite and valid")
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_interval(prediction: Sequence[float], radius: float) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(radius) or radius < 0:
        raise ValueError("nonfinite prediction or conformal radius")
    return np.clip(prediction - radius, 0.0, None), prediction + radius
```

- [ ] **Step 6: Implement seeded site-cluster bootstrap**

Add:

```python
def cluster_bootstrap(
    predictions: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    required = {"site", "y_true", "y_pred"}
    if required - set(predictions):
        raise ValueError(f"bootstrap missing columns: {sorted(required - set(predictions))}")
    sites = predictions["site"].drop_duplicates().to_numpy()
    if len(sites) < 2:
        raise ValueError("site-cluster bootstrap requires at least two sites")
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        chosen = rng.choice(sites, size=len(sites), replace=True)
        sampled = pd.concat(
            [predictions.loc[predictions["site"].eq(site)] for site in chosen],
            ignore_index=True,
        )
        rows.append({"iteration": iteration, **regression_metrics(sampled["y_true"], sampled["y_pred"])})
    return pd.DataFrame(rows)


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("bootstrap interval contains nonfinite values")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)
```

Add `paired_cluster_bootstrap` that inner-joins two procedure predictions by `row_key` and `site`, resamples sites once per iteration, and saves metric A minus metric B. Never compare unpaired rows.

- [ ] **Step 7: Run focused statistical tests**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -v
rtk conda run -n DL ruff check src tests
```

Expected: split coverage, metric, conformal, and deterministic cluster-bootstrap tests PASS.

- [ ] **Step 8: Commit evaluation primitives**

```bash
rtk git add src/warfarin_dose/evaluation.py tests/test_evaluation.py
rtk git commit -m "feat: add site-aware evaluation primitives"
```

### Task 6: Run the Primary Nested Leave-One-Site-Out Experiment

**Files:**
- Modify: `src/warfarin_dose/evaluation.py`
- Modify: `src/warfarin_dose/cli.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: canonical feature frame, `ModelSpec`, grouped split/metric/conformal functions
- Produces: `score_candidates(X, y, sites, columns, candidates, seed) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `select_one_se(scores: pd.DataFrame, candidates: Sequence[ModelSpec]) -> ModelSpec`
- Produces: `run_primary_experiment(raw_path: Path, output_dir: Path, seed: int = DEFAULT_SEED, candidates: Sequence[ModelSpec] | None = None) -> Path`
- Produces: `fit_final_model(frame, feature_set, candidates, output_path, seed) -> dict[str, object]`
- Produces: CLI command `python -m warfarin_dose run-experiment --analysis primary`

- [ ] **Step 1: Write failing one-standard-error and primary-coverage tests**

Add to `tests/test_evaluation.py`:

```python
from warfarin_dose.evaluation import run_primary_frame, select_one_se
from warfarin_dose.models import ModelSpec


def test_one_se_prefers_simpler_family_then_direct_target():
    specs = [
        ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0),
        ModelSpec("hist_gb", {"learning_rate": 0.1, "max_leaf_nodes": 15}, "sqrt", 2, 0),
    ]
    scores = pd.DataFrame({
        "candidate_key": [specs[0].key] * 3 + [specs[1].key] * 3,
        "fold": [0, 1, 2, 0, 1, 2],
        "mae_mg_week": [8.4, 8.5, 8.6, 7.0, 8.1, 9.2],
    })
    assert select_one_se(scores, specs) == specs[0]


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
```

- [ ] **Step 2: Run tests and confirm runner functions are absent**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL because nested scoring and experiment functions are not defined.

- [ ] **Step 3: Implement candidate scoring and structured failures**

Add to `evaluation.py`:

```python
from pathlib import Path
import json
import platform
import subprocess
from importlib.metadata import version

import joblib
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from .data import RAW_PATH, prepare_cohort, read_raw, sha256_file
from .features import build_feature_frame, feature_columns, select_feature_matrix
from .models import ModelSpec, iwpc_clinical, iwpc_pharmacogenetic, make_model_pipeline, model_candidates


def score_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    candidates: Sequence[ModelSpec],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    splits = inner_site_splits(sites)
    for spec in candidates:
        for fold, (train, validation) in enumerate(splits):
            try:
                pipeline = make_model_pipeline(columns, spec, seed + fold)
                pipeline.fit(X.iloc[train], y[train])
                prediction = pipeline.predict(X.iloc[validation])
                score_rows.append({
                    "candidate_key": spec.key,
                    "fold": fold,
                    "mae_mg_week": float(mean_absolute_error(y[validation], prediction)),
                })
            except Exception as error:
                failures.append({
                    "stage": "inner_fit", "candidate_key": spec.key, "fold": fold,
                    "error_type": type(error).__name__, "message": str(error),
                })
    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise RuntimeError(f"no successful candidate model; failures={failures}")
    return scores, pd.DataFrame(failures)


def select_one_se(scores: pd.DataFrame, candidates: Sequence[ModelSpec]) -> ModelSpec:
    summary = scores.groupby("candidate_key")["mae_mg_week"].agg(["mean", "std", "count"])
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    best_key = summary["mean"].idxmin()
    threshold = float(summary.loc[best_key, "mean"] + summary.loc[best_key, "se"])
    eligible = {key for key, row in summary.iterrows() if row["mean"] <= threshold}
    successful = [spec for spec in candidates if spec.key in eligible]
    if not successful:
        raise RuntimeError("one-standard-error selection found no successful candidate")
    return min(
        successful,
        key=lambda spec: (
            spec.family_order, 0 if spec.target_mode == "direct" else 1, spec.complexity_order,
        ),
    )
```

A candidate is eligible only if it succeeded in every inner fold. Add `expected_fold_count` filtering before summarizing; partial failures remain in `failures.csv` and cannot win selection.

- [ ] **Step 4: Generate inner out-of-fold residuals for conformal calibration**

Add:

```python
def calibration_residuals(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    spec: ModelSpec,
    seed: int,
) -> np.ndarray:
    residuals = np.full(len(X), np.nan)
    for fold, (train, validation) in enumerate(inner_site_splits(sites)):
        pipeline = make_model_pipeline(columns, spec, seed + fold)
        pipeline.fit(X.iloc[train], y[train])
        residuals[validation] = np.abs(y[validation] - pipeline.predict(X.iloc[validation]))
    if not np.isfinite(residuals).all():
        raise ValueError("inner grouped conformal calibration did not cover every training row")
    return residuals
```

- [ ] **Step 5: Implement one learned site-held-out procedure**

Add:

```python
def _run_learned_procedure(
    frame: pd.DataFrame,
    procedure: str,
    columns: Sequence[str],
    candidates: Sequence[ModelSpec],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = select_feature_matrix(frame, columns)
    y = frame["weekly_dose_mg"].to_numpy(float)
    predictions, selections, failures = [], [], []
    for outer_fold, (train, test) in enumerate(site_outer_splits(frame)):
        training_sites = frame.iloc[train]["site"].astype(str).to_numpy()
        scores, fold_failures = score_candidates(
            X.iloc[train].reset_index(drop=True), y[train], training_sites,
            columns, candidates, seed + outer_fold * 100,
        )
        selected = select_one_se(scores, candidates)
        residuals = calibration_residuals(
            X.iloc[train].reset_index(drop=True), y[train], training_sites,
            columns, selected, seed + outer_fold * 100,
        )
        radius = conformal_quantile(residuals, coverage=0.90)
        pipeline = make_model_pipeline(columns, selected, seed + outer_fold)
        pipeline.fit(X.iloc[train], y[train])
        predicted = pipeline.predict(X.iloc[test])
        lower, upper = conformal_interval(predicted, radius)
        target_min, target_max = float(y[train].min()), float(y[train].max())
        for position, prediction, low, high in zip(test, predicted, lower, upper, strict=True):
            predictions.append({
                "row_key": frame.iloc[position]["row_key"], "site": frame.iloc[position]["site"],
                "outer_site": frame.iloc[position]["site"], "outer_fold": outer_fold,
                "procedure": procedure, "y_true": y[position], "y_pred": prediction,
                "interval_lower": low, "interval_upper": high,
                "extrapolated_target": bool(prediction < target_min or prediction > target_max),
                "model_family": selected.family, "target_mode": selected.target_mode,
                "candidate_key": selected.key, "prediction_status": "ok",
                "gender": frame.iloc[position]["gender"],
                "age_group": (
                    "Unknown" if not np.isfinite(frame.iloc[position]["age_decade"])
                    else "<50" if frame.iloc[position]["age_decade"] < 5
                    else "50-69" if frame.iloc[position]["age_decade"] < 7 else "70+"
                ),
                "race_audit": frame.iloc[position]["race"],
                "cyp2c9_group": frame.iloc[position]["cyp2c9_group"],
                "vkorc1": frame.iloc[position]["vkorc1"],
                "dose_category": dose_category([y[position]])[0],
            })
        selections.append({
            "procedure": procedure, "outer_fold": outer_fold,
            "outer_site": frame.iloc[test[0]]["site"], "candidate_key": selected.key,
            "conformal_radius": radius,
        })
        if not fold_failures.empty:
            fold_failures = fold_failures.assign(
                procedure=procedure, outer_fold=outer_fold,
                outer_site=frame.iloc[test[0]]["site"],
            )
            failures.extend(fold_failures.to_dict("records"))
    result = pd.DataFrame(predictions)
    counts = result.groupby("row_key").size()
    if len(result) != len(frame) or not counts.eq(1).all():
        raise ValueError(f"{procedure} did not produce exactly one outer prediction per patient")
    return result, pd.DataFrame(selections), pd.DataFrame(failures)
```

Never catch an outer-fold failure and continue with missing primary predictions. Partial candidate failures are tolerated only before a successful selected model exists.

- [ ] **Step 6: Add fixed, fold-summary, and exact published comparator predictions**

For every outer split, append long-format rows for:

- `fixed_35_mg_week`: always `35.0`;
- `training_mean`: mean outer-training dose;
- `training_median`: median outer-training dose;
- `iwpc_clinical`: exact equation or `prediction_status="missing_required_comparator_input"` with null `y_pred`;
- `iwpc_pharmacogenetic`: exact equation or the same explicit missing status.

Comparator rows carry no conformal interval. Published comparator metrics use finite rows only; paired differences inner-join shared finite `row_key` values. Fixed and learned procedures cover all eligible rows.
Every benchmark/comparator row copies the same `gender`, `age_group`, `race_audit`, `cyp2c9_group`, `vkorc1`, and actual `dose_category` audit fields as learned rows so subgroup comparisons are derived from saved predictions without reopening patient data.

- [ ] **Step 7: Persist the complete primary run contract**

Implement `run_primary_frame(raw, output_dir, candidates=None, seed=DEFAULT_SEED)`:

1. `prepare_cohort(raw)` then `build_feature_frame(cohort.data)`.
2. Run `clinical_ml` with clinical columns and `pharmacogenomic_ml` with clinical plus CYP2C9/VKORC1 columns.
3. Append benchmark/comparator rows.
4. Save `predictions.csv`, `selections.csv`, `failures.csv`, `cohort_flow.csv`, `issues.csv`, `feature_metadata.json`, `metrics.csv`, `site_metrics.csv`, `dose_category_metrics.csv`, `bootstrap.csv`, and `paired_differences.csv`.
5. Save `manifest.json` with seed, UTC timestamp, Git revision, Python/platform, package versions, source checksum, cohort/site counts, analysis name, model grid, and all output filenames.

Use:

```python
def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "featranker", "xlrd", "joblib", "matplotlib"]
    return {name: version(name) for name in names}
```

`run_primary_experiment(raw_path, output_dir, seed=DEFAULT_SEED, candidates=None)` calls `read_raw(raw_path)` and delegates to `run_primary_frame(raw, output_dir, candidates=candidates, seed=seed)` so tests never need the real workbook.

- [ ] **Step 8: Fit the final research artifact without reporting training performance**

After outer predictions are frozen:

1. Compare `clinical_ml` and `pharmacogenomic_ml` by mean site MAE.
2. Compute the best procedure's standard error across site MAEs.
3. If clinical ML is within that one-standard-error threshold, choose clinical ML; otherwise choose the lower-error procedure.
4. On the full cohort, rerun grouped candidate selection for that feature set, generate grouped out-of-fold residuals, fit the pipeline on all rows, and save `final_model.joblib`.

The joblib payload is exactly:

```python
{
    "pipeline": fitted_pipeline,
    "feature_columns": selected_columns,
    "feature_set": selected_feature_set,
    "model_spec": selected_spec,
    "conformal_radius": full_data_radius,
    "numeric_training_ranges": numeric_ranges,
    "target_training_range": [float(y.min()), float(y.max())],
    "source_sha256": source_sha256,
    "git_revision": git_revision,
    "research_warning": "Research use only; not prescribing guidance or a medical device.",
}
```

- [ ] **Step 9: Add `run-experiment --analysis primary`**

Extend `cli.py` with arguments:

```text
run-experiment --analysis {primary,feature-selection,complete-case,random-cv,ablation,all}
               --input data/raw/PS206767-553247439.xls
               --output artifacts/run
               --seed 20260717
```

The default analysis is `primary`. Fail if the output directory already contains a manifest, preventing silent overwrite.

- [ ] **Step 10: Verify the primary nested workflow on synthetic data**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -v
rtk conda run -n DL ruff check src tests
```

Expected: one-standard-error selection, every-patient coverage, nonnegative prediction, outer-site identity, and interval tests PASS using only one injected Ridge candidate.

- [ ] **Step 11: Commit primary nested evaluation**

```bash
rtk git add src/warfarin_dose/evaluation.py src/warfarin_dose/cli.py tests/test_evaluation.py
rtk git commit -m "feat: add nested site-held-out experiment"
```

### Task 7: Add Leakage-Safe FeatRanker and Prespecified Sensitivities

**Files:**
- Modify: `src/warfarin_dose/evaluation.py`
- Modify: `src/warfarin_dose/cli.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: FeatRanker 0.2.0 `fit`/`rank_features`, inner grouped splits, model selection
- Produces: `rank_feature_blocks(X, y, sites, columns, seed) -> tuple[pd.DataFrame, list[dict[str, object]]]`
- Produces: feature-selection, complete-case, random-CV, and ablation artifact subdirectories
- Produces: `run_all_analyses(raw_path, output_dir, seed) -> Path`

- [ ] **Step 1: Write a failing FeatRanker separation test**

Monkeypatch `evaluation.FeatureRanker` with a spy whose `fit` records unique synthetic target IDs and whose `rank_features` records evaluation target IDs. Add:

```python
def test_featranker_receives_disjoint_inner_training_and_validation(monkeypatch):
    calls = []

    class SpyRanker:
        def __init__(self, **_):
            pass

        def fit(self, X, y, feature_names):
            self.train = set(np.asarray(y))
            return self

        def rank_features(self, X_eval, y_eval, *, scoring, feature_groups, n_repeats, random_state):
            evaluation = set(np.asarray(y_eval))
            calls.append((self.train, evaluation, scoring, feature_groups, n_repeats, random_state))
            return {
                "models": {"spy": {"evaluation_score": -1.0, "importance": {
                    name: {"values": [1.0], "mean": 1.0, "std": 0.0, "rank": rank}
                    for rank, name in enumerate(feature_groups, start=1)
                }}},
                "consensus": [
                    {"feature_group": name, "median_rank": rank, "mean_rank": rank,
                     "rank_std": 0.0, "n_models": 1}
                    for rank, name in enumerate(feature_groups, start=1)
                ],
                "failures": [], "evaluation_mode": "held_out",
            }

    monkeypatch.setattr("warfarin_dose.evaluation.FeatureRanker", SpyRanker)
    rank_feature_blocks_for_encoded_test_matrix(seed=7)

    assert calls
    assert all(train.isdisjoint(validation) for train, validation, *_ in calls)
    assert all(call[2] == "neg_mean_absolute_error" for call in calls)
    assert all(call[4] == 20 for call in calls)
```

The helper used by this test builds three ordinary feature columns, uses `y=np.arange(n_rows)` only as unique test IDs, supplies three site groups, and calls the same internal fold-ranking function used by production.

- [ ] **Step 2: Run the test and confirm the ranking path is absent**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -k featranker -v
```

Expected: FAIL because no ranking function exists.

- [ ] **Step 3: Implement inner-fold semantic grouped permutation ranking**

Add imports and implementation:

```python
from featranker import FeatureRanker

from .features import make_preprocessor, semantic_feature_groups


def rank_feature_blocks(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    columns: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rank_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for fold, (train, validation) in enumerate(inner_site_splits(sites)):
        preprocessor = make_preprocessor(columns, scale_numeric=False)
        X_train = preprocessor.fit_transform(X.iloc[train])
        X_validation = preprocessor.transform(X.iloc[validation])
        names = preprocessor.get_feature_names_out().tolist()
        groups = semantic_feature_groups(names)
        ranker = FeatureRanker(task="reg", group="all")
        ranker.fit(X_train, y[train], feature_names=names)
        report = ranker.rank_features(
            X_validation,
            y[validation],
            scoring="neg_mean_absolute_error",
            feature_groups=groups,
            n_repeats=20,
            random_state=seed + fold,
        )
        if report.get("evaluation_mode") != "held_out":
            raise ValueError("FeatRanker must report held_out evaluation mode")
        for model, model_report in report["models"].items():
            for block, importance in model_report["importance"].items():
                rank_rows.append({
                    "fold": fold, "model": model, "feature_block": block,
                    "rank": importance["rank"], "importance_mean": importance["mean"],
                    "importance_std": importance["std"],
                })
        reports.append(report)
    ranks = pd.DataFrame(rank_rows)
    if ranks.empty:
        raise RuntimeError("FeatRanker produced no successful model rankings")
    aggregate = ranks.groupby("feature_block")["rank"].agg(
        median_rank="median", mean_rank="mean", rank_std="std", observations="size"
    ).reset_index()
    top5 = ranks.groupby("feature_block")["rank"].apply(lambda values: float((values <= 5).mean()))
    aggregate["top5_frequency"] = aggregate["feature_block"].map(top5)
    aggregate = aggregate.sort_values(["median_rank", "mean_rank", "feature_block"]).reset_index(drop=True)
    return aggregate, reports
```

Save every raw FeatRanker report as JSON so initialization, fit, and ranking failures remain auditable. Aggregate ranks and selection frequencies, never raw importance magnitudes across heterogeneous models.

- [ ] **Step 4: Implement top-5/top-10/all nested subset selection**

Within each outer site:

1. Rank blocks using only outer-training rows and grouped inner folds.
2. Define deterministic block lists from aggregate order: first 5, first 10, and all.
3. For each subset, run the same `score_candidates` grouped inner validation.
4. Summarize the selected candidate's fold MAE for each subset.
5. Let the best subset's standard error define the threshold; choose the smallest subset within threshold in fixed order top-5, top-10, all.
6. Refit that subset on all outer-training rows, calibrate intervals from grouped inner residuals, and predict the untouched outer site once.

Save this procedure as `pharmacogenomic_ranked`. The unchanged `pharmacogenomic_ml` all-feature result remains primary.

- [ ] **Step 5: Prespecify the feature-selection adoption rule**

After all outer predictions are frozen, adopt the ranked subset in `final_model.joblib` only when either:

- its mean site MAE is lower than all-feature pharmacogenomic ML; or
- its mean site MAE is within one all-feature site-MAE standard error and it uses at least 30% fewer semantic blocks.

Otherwise retain all features and report ranking only as an interpretation/stability result. This decision and both error summaries go into `feature_selection_decision.json`.

- [ ] **Step 6: Add complete-case sensitivity**

Define complete case on the canonical pharmacogenomic feature columns: every numeric value finite and every categorical value not `Unknown`. Run the same leave-one-site-out nested procedure on that reduced cohort. Save cohort counts and site retention; call the procedure `pharmacogenomic_complete_case`. Do not use complete-case results to select the primary final artifact.

- [ ] **Step 7: Add nested random-CV optimism comparator**

Add deterministic random split helpers:

```python
from sklearn.model_selection import KFold


def random_outer_splits(n_rows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(n_splits=10, shuffle=True, random_state=seed).split(np.arange(n_rows)))


def random_inner_splits(n_rows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_rows)))
```

Make the existing scorers choose an explicit inner mode without changing the site-aware default:

```python
def resolve_inner_splits(sites: np.ndarray, seed: int, mode: str):
    if mode == "site":
        return inner_site_splits(sites)
    if mode == "random":
        return random_inner_splits(len(sites), seed)
    raise ValueError(f"unknown inner split mode: {mode}")
```

Add `inner_mode: str = "site"` to `score_candidates` and `calibration_residuals`, and replace each direct `inner_site_splits(sites)` call with `resolve_inner_splits(sites, seed, inner_mode)`. Add `outer_splits=None` and `inner_mode: str = "site"` to `_run_learned_procedure`, resolving `outer_splits = site_outer_splits(frame)` when omitted and forwarding `inner_mode` to scoring/calibration. Run the full pharmacogenomic procedure with `random_outer_splits(len(frame), seed)` and `inner_mode="random"`, and save `pharmacogenomic_random_cv`. Label every artifact and report table `optimism_comparator`; never merge these predictions into primary site-held-out estimates.

- [ ] **Step 8: Add prespecified block ablations**

Run leave-one-site-out nested pharmacogenomic procedures after removing each block:

- `demographics`: age decade, gender;
- `anthropometrics`: height, weight;
- `clinical_conditions`: indication, target INR, diabetes, CHF/cardiomyopathy, valve replacement;
- `medications`: amiodarone, enzyme inducer, smoking, and statin when admitted by its QA gate;
- `pharmacogenomics`: CYP2C9 and VKORC1.

After feature ranking, also remove at most three individual blocks satisfying both median rank `<=5` and top-5 frequency `>=0.70`, ordered by median rank then name. This rule is fixed before viewing ablation outer-test errors.

- [ ] **Step 9: Route all secondary analyses and verify**

Each CLI analysis writes under its own subdirectory of the run. `--analysis all` runs primary first, then feature selection, complete case, random CV, and ablation, and finally reevaluates the final-model adoption rule.

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_evaluation.py -v
rtk conda run -n DL ruff check src tests
```

Expected: spy proves disjoint FeatRanker train/evaluation rows and semantic groups; synthetic sensitivity paths use injected Ridge-only candidates and finish quickly.

- [ ] **Step 10: Commit ranking and sensitivities**

```bash
rtk git add src/warfarin_dose/evaluation.py src/warfarin_dose/cli.py tests/test_evaluation.py
rtk git commit -m "feat: add nested feature and sensitivity analyses"
```

### Task 8: Build Deterministic Reports and Safe Research Inference

**Files:**
- Create: `src/warfarin_dose/reporting.py`
- Modify: `src/warfarin_dose/cli.py`
- Create: `docs/DATA_CARD.md`
- Create: `docs/MODEL_CARD.md`
- Create: `README.md`
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: saved run CSV/JSON and `final_model.joblib`
- Produces: `build_report(run_dir: Path) -> Path`
- Produces: `predict_patient(model_path: Path, input_path: Path) -> dict[str, object]`
- Produces: CLI commands `build-report` and `predict`

- [ ] **Step 1: Write a failing report/inference integration test**

Create `tests/test_integration.py`:

```python
import json

import pandas as pd

from warfarin_dose.evaluation import run_primary_frame
from warfarin_dose.models import ModelSpec
from warfarin_dose.reporting import build_report, predict_patient


def test_synthetic_run_builds_report_and_safe_prediction(raw_frame, tmp_path):
    candidates = [ModelSpec("ridge", {"alpha": 1.0}, "direct", 0, 0)]
    run_dir = run_primary_frame(raw_frame, tmp_path / "run", candidates=candidates, seed=7)
    report = build_report(run_dir)
    patient = {
        "age_decade": 6, "gender": "Female", "height_cm": 165, "weight_kg": 70,
        "indication": "7", "target_inr": 2.5, "diabetes": "No",
        "chf_cardiomyopathy": "No", "valve_replacement": "No", "amiodarone": "No",
        "enzyme_inducer": "No", "smoker": "No", "cyp2c9_group": "Normal",
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
```

- [ ] **Step 2: Run the integration test and confirm report functions are absent**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_integration.py -v
```

Expected: FAIL on import of `warfarin_dose.reporting`.

- [ ] **Step 3: Implement report tables from saved outer predictions**

Create `reporting.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import dose_category, regression_metrics
from .features import PHARMACOGENOMIC_FEATURES

RESEARCH_WARNING = (
    "Research use only; this estimate is not prescribing guidance, a medical device, "
    "or a substitute for clinician-guided INR monitoring."
)


def _finite_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.loc[
        predictions["prediction_status"].eq("ok") & predictions["y_pred"].notna()
    ].copy()


def _metric_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        rows.append({**dict(zip(group_columns, keys, strict=True)), **metrics})
    return pd.DataFrame(rows)
```

`build_report` must independently derive and save:

- `overall_metrics.csv` by procedure;
- `site_metrics.csv` by procedure/site;
- `dose_category_metrics.csv` using actual-dose cutoffs `<=21`, `>21 and <49`, `>=49`;
- `subgroup_metrics.csv` for gender, age group, race audit field, CYP2C9/VKORC1 availability/group, and dose category, suppressing metrics under 30 rows;
- `interval_metrics.csv` with 90% coverage and mean width overall, by site, and eligible subgroup;
- `paired_differences.csv` and bootstrap 95% intervals from saved paired bootstrap artifacts;
- `feature_stability.csv`, `ablation_metrics.csv`, and sensitivity tables when those analysis artifacts exist.

The report never recomputes or fits a model.

- [ ] **Step 4: Implement four publication figures with matplotlib only**

Use a deterministic `plt.style.context("seaborn-v0_8-whitegrid")`, 300 DPI, labeled mg/week axes, and close every figure:

1. `observed_vs_predicted.png`: primary learned procedures plus IWPC PGx on finite paired rows, identity line.
2. `mae_by_site.png`: horizontal site MAE for clinical ML, pharmacogenomic ML, and IWPC PGx.
3. `feature_rank_stability.png`: median rank with rank-standard-deviation error bars when ranking exists.
4. `interval_coverage_by_site.png`: empirical 90% interval coverage with a nominal 0.90 reference line.

Do not add decorative dashboards or interactive plotting dependencies.

- [ ] **Step 5: Generate a manuscript-style Markdown report**

`build_report(run_dir)` creates `run_dir/report/report.md` with these exact sections:

```markdown
# Site-Aware Warfarin Dose Prediction from Public IWPC Data

## Research question
## Public data and cohort
## Pre-treatment clinical and pharmacogenomic features
## Leakage-safe validation and model selection
## Primary site-held-out performance
## Comparison with fixed and published IWPC algorithms
## Prediction uncertainty
## Feature stability, ablations, and sensitivity analyses
## Subgroup and site audit
## Limitations
## Reproducibility
## Research-use warning
```

Populate counts, metrics, intervals, selected model information, and artifact links from saved files. State explicitly that performance under site shift may differ, conformal coverage is empirical rather than guaranteed under hospital shift, permutation importance is associational/correlation-sensitive, and no result is a dose recommendation.

- [ ] **Step 6: Implement strict JSON research inference**

Add:

```python
def predict_patient(model_path: Path, input_path: Path) -> dict[str, object]:
    artifact = joblib.load(model_path)
    patient = json.loads(Path(input_path).read_text(encoding="utf-8"))
    expected = artifact["feature_columns"]
    allowed = set(PHARMACOGENOMIC_FEATURES) | {"statin"}
    forbidden = {"weekly_dose_mg", "site", "row_key", "race"}
    unknown = sorted(set(patient) - allowed - forbidden)
    supplied_forbidden = sorted(set(patient) & forbidden)
    if unknown or supplied_forbidden:
        raise ValueError(
            f"incompatible inference schema; unknown={unknown}, forbidden={supplied_forbidden}"
        )
    missing = [name for name in expected if name not in patient or patient[name] is None]
    row = pd.DataFrame([{name: patient.get(name, np.nan) for name in expected}])
    prediction = float(artifact["pipeline"].predict(row)[0])
    radius = float(artifact["conformal_radius"])
    lower, upper = max(0.0, prediction - radius), prediction + radius
    numeric_flags = []
    for name, limits in artifact["numeric_training_ranges"].items():
        value = patient.get(name)
        if value is not None and not limits[0] <= float(value) <= limits[1]:
            numeric_flags.append(name)
    target_limits = artifact["target_training_range"]
    return {
        "weekly_dose_mg": prediction,
        "average_daily_dose_mg": prediction / 7,
        "interval_90_mg_week": [lower, upper],
        "missing_inputs": missing,
        "extrapolation_flags": {
            "numeric_inputs_outside_training_range": numeric_flags,
            "prediction_outside_training_target_range": not target_limits[0] <= prediction <= target_limits[1],
        },
        "model_version": artifact["git_revision"],
        "source_sha256": artifact["source_sha256"],
        "warning": RESEARCH_WARNING,
    }
```

Before prediction, coerce numeric inputs to finite floats and categorical inputs to strings; reject nonfinite supplied numbers. Missing inputs remain allowed and are handled by the fitted pipeline.

- [ ] **Step 7: Complete CLI routing**

Add:

```text
build-report --run-dir artifacts/run
predict --model artifacts/run/final_model.joblib --input patient.json
```

`predict` prints sorted, indented JSON. Every command returns nonzero on validation failure through an actionable exception message; no command catches and hides data/model errors.

- [ ] **Step 8: Write portfolio README and research cards**

`README.md` contains, in order:

- one-paragraph biomedical-informatics research question;
- why mg/week and `/7` matter;
- public dataset citation and checksum;
- cohort/feature summary and forbidden predictors;
- leave-one-site-out diagram in plain text;
- exact setup and five CLI commands;
- explanation of fixed 35 mg/week versus individual patient dose;
- primary/secondary experiment distinction;
- result links generated after the public run;
- limitations and research-only warning;
- legacy archive note.

`docs/DATA_CARD.md` documents source provenance, 68-field workbook context, 21 sites as an audit expectation, eligibility, identifiers, missingness, genotype/race handling, stable-dose heterogeneity across sites, checksum update process, and prohibition on redistribution claims beyond the public source.

`docs/MODEL_CARD.md` documents model families, all-feature primary analysis, FeatRanker secondary role, site-held-out validation, metrics, conformal interpretation, subgroup suppression, race exclusion from learned inputs, intended research use, prohibited clinical use, and known limitations for rare genotypes/high doses/site shift.

- [ ] **Step 9: Run integration and focused tests**

Run:

```bash
rtk conda run -n DL python -m pytest tests/test_integration.py tests/test_evaluation.py -v
rtk conda run -n DL ruff check src tests
```

Expected: synthetic audit-to-nested-run-to-report-to-predict path PASS; generated daily dose is exactly weekly/7; report figures/tables exist.

- [ ] **Step 10: Commit reporting, inference, and documentation**

```bash
rtk git add README.md docs/DATA_CARD.md docs/MODEL_CARD.md src/warfarin_dose tests/test_integration.py
rtk git commit -m "feat: add research report and safe inference"
```

### Task 9: Add CI and Verify Full Public-Data Reproducibility

**Files:**
- Create: `.github/workflows/tests.yml`
- Modify: `README.md`
- Modify: curated result links only after the real run

**Interfaces:**
- Consumes: complete package and public source endpoint
- Produces: clean build, CI checks, public-data smoke audit, full analysis artifacts, and final report

- [ ] **Step 1: Write CI without a public-data download**

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[dev]'
      - run: ruff check src tests
      - run: python -m pytest
      - run: python -m build
```

CI uses only synthetic data; it never downloads or uploads patient-level rows.

- [ ] **Step 2: Run all local verification before touching the public endpoint**

Run:

```bash
rtk conda run -n DL ruff check src tests
rtk conda run -n DL python -m pytest
rtk conda run -n DL python -m build
rtk git diff --check
```

Expected: Ruff clean; all tests PASS; wheel and source distribution build; diff check clean.

- [ ] **Step 3: Commit CI**

```bash
rtk git add .github/workflows/tests.yml
rtk git commit -m "ci: test and build research package"
```

- [ ] **Step 4: Download and verify the reviewed public dataset**

Run:

```bash
rtk conda run -n DL python -m warfarin_dose download-data
rtk shasum -a 256 data/raw/PS206767-553247439.xls
```

Expected SHA-256:

```text
0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d
```

If it differs, stop. Do not update the constant or continue without a separate reviewed dataset-version change.

- [ ] **Step 5: Run the real-data audit smoke check**

Run:

```bash
rtk conda run -n DL python -m warfarin_dose audit-data
```

Expected structural checks: about 5,700 source rows, about 5,410 eligible stable-dose rows, 21 sites, no missing required columns, and an explicitly reported legacy complete-case count near 1,477. Treat these as audit comparisons, not forced assertions. Investigate material differences before modeling.

- [ ] **Step 6: Run the complete prespecified analysis**

Run:

```bash
rtk conda run -n DL python -m warfarin_dose run-experiment --analysis all --output artifacts/run
```

Expected files include primary and secondary manifests, all eligible learned outer predictions exactly once, benchmark/comparator predictions, feature-ranking raw reports, sensitivity/ablation outputs, 2,000-replicate bootstrap artifacts, and `final_model.joblib`. No nonfinite successful learned predictions or intervals are allowed.

- [ ] **Step 7: Build and inspect the final report**

Run:

```bash
rtk conda run -n DL python -m warfarin_dose build-report --run-dir artifacts/run
```

Verify:

- every performance claim can be traced to `predictions.csv`;
- primary tables use site-held-out predictions, not random CV;
- random-CV tables are labeled optimism comparator;
- fixed 35 mg/week is labeled historical 5 mg/day benchmark;
- published comparator sample sizes disclose missing required inputs;
- feature ranking does not claim causality or guaranteed performance gain;
- subgroup estimates under 30 rows are suppressed;
- uncertainty language says empirical under site shift;
- no prescribing language appears.

- [ ] **Step 8: Curate only small non-patient outputs**

Copy only the generated aggregate report, aggregate CSV tables, and publication PNGs chosen for the portfolio into `results/`. Do not commit raw data, row-level predictions, split assignments, model files, or manifests containing local paths. Add `results/README.md` recording the run Git revision and source checksum.

Run:

```bash
rtk mkdir -p results
rtk cp artifacts/run/report/report.md results/README.md
rtk cp -R artifacts/run/report/tables results/tables
rtk cp -R artifacts/run/report/figures results/figures
rtk git status --short
rtk git diff --check
```

Expected: only intentionally curated aggregate result files are untracked/modified; `data/raw/` and `artifacts/` remain ignored.

- [ ] **Step 9: Commit reproducible aggregate results**

```bash
rtk git add README.md results
rtk git commit -m "docs: add site-held-out research results"
```

- [ ] **Step 10: Perform final acceptance verification**

Run:

```bash
rtk conda run -n DL ruff check src tests
rtk conda run -n DL python -m pytest
rtk conda run -n DL python -m build
rtk git diff --check
rtk git status --short --branch
```

Expected: all checks pass; execution worktree is clean; the original checkout's pre-existing migration remains untouched; the final result does not need to beat IWPC or any complex model to satisfy the scientific acceptance criteria.
