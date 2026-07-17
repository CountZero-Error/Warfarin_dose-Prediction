# Warfarin Dose Research Rebuild

This biomedical-informatics research project asks whether pre-treatment clinical and pharmacogenomic information can estimate a patient's stable warfarin dose when tested on a previously unseen clinical site. It is a reproducible, research-only analysis of public IWPC data—not a prescribing tool.

## Dose unit

Weekly mg/week is the canonical target and reporting unit. The displayed average daily dose is exactly `weekly_dose_mg / 7`; it is not a separately modeled outcome.

## Public data

The input is the public International Warfarin Pharmacogenetics Consortium (IWPC) workbook distributed through PharmGKB: <https://api.pharmgkb.org/v1/download/submission/553247439>. The reviewed workbook SHA-256 is `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`.

## Cohort and features

Eligible records have a stable, positive therapeutic dose and a site. Learned models use pre-treatment clinical features with optional CYP2C9/VKORC1 features; dose, post-treatment INR, identifiers, site, and race are forbidden predictors. Missing expected predictors are imputed inside each fitted pipeline.

## Validation

```text
hold out site S ──> train/select/calibrate on all other sites ──> evaluate once on S
repeat for every site ──> combine out-of-site predictions
```

## Setup and commands

```bash
conda run -n DL python -m pip install -e '.[dev]'
warfarin-dose download-data --output data/raw/PS206767-553247439.xls
warfarin-dose audit-data --input data/raw/PS206767-553247439.xls --output artifacts/audit
warfarin-dose run-experiment --analysis primary --input data/raw/PS206767-553247439.xls --output artifacts/run
warfarin-dose build-report --run-dir artifacts/run/primary
warfarin-dose predict --model artifacts/run/primary/final_model.joblib --input patient.json
```

## Comparators and analyses

Fixed 35 mg/week is a population-reference comparator, not an individual patient dose. The primary analysis evaluates all-feature clinical and pharmacogenomic models with site-held-out validation. Feature selection, complete-case, random-CV, and ablation analyses are secondary and do not replace the primary analysis.

## Results

Run-specific links are generated after a verified public-data run: `artifacts/run/primary/report/report.md`, its tables, and its figures. No public-run performance claim is made here before that run exists.

## Limitations and research-use warning

Hospital/site shift, rare genotypes, high doses, missingness, and stable-dose definition differences may limit generalization. **Research use only: estimates are not prescribing guidance, a medical device, or a substitute for clinician-guided INR monitoring.**

## Legacy archive

`archive/` preserves historical notebooks and models; it is not part of the reproducible research pipeline.
