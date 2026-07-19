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
warfarin-dose run-experiment --analysis all --input data/raw/PS206767-553247439.xls --output artifacts/run
warfarin-dose build-report --run-dir artifacts/run
warfarin-dose predict --model artifacts/run/primary/final_model.joblib --input patient.json
```

## Comparators and analyses

Fixed 35 mg/week is a population-reference comparator, not an individual patient dose. The primary analysis evaluates all-feature clinical and pharmacogenomic models with site-held-out validation. Feature selection, complete-case, random-CV, and ablation analyses are secondary and do not replace the primary analysis.

## Results

The verified public-data run retained 5,410 eligible patients from 21 sites. Under primary leave-one-site-out evaluation, the all-feature pharmacogenomic model achieved MAE 9.48 mg/week (90% conformal coverage 90.4%), the clinical-only model achieved MAE 11.69 mg/week, and the fixed 35 mg/week historical reference (5 mg/day) achieved MAE 13.23 mg/week. The published IWPC pharmacogenetic equation achieved MAE 8.65 mg/week on the smaller 4,302-patient subset with its required inputs, so its overall number is not a same-cohort replacement for the primary comparison.

The nested fold-wise FeatRanker procedure achieved out-of-site MAE 9.38 mg/week; each outer training fold selected its own ranked subset. The separately refitted final artifact uses the aggregate-ranked static inputs `vkorc1`, `weight_kg`, `age_decade`, `cyp2c9_group`, and `height_cm`. The 9.38 value evaluates the nested procedure, not that static full-cohort refit. Random-CV MAE was 8.77 mg/week and is labeled only as an optimism comparator.

The frozen evaluation was generated at sanitized revision `7ea3e03dc5b2bedea4e0af48421c0e6474c24283`. The selected five-feature final artifact was provenance-corrected and deterministically refit at sanitized revision `ec753167d82242a8c176f3f2f7b1e09ad4a22dea` without changing the frozen outer predictions.

- [Curated research report](results/README.md)
- [Overall metrics](results/tables/overall_metrics.csv)
- [Paired bootstrap comparisons](results/tables/paired_differences.csv)
- [Sensitivity analyses](results/tables/sensitivity_metrics.csv)
- [Feature stability](results/tables/feature_stability.csv)
- [Observed versus predicted figure](results/figures/observed_vs_predicted.png)
- [Site-level MAE figure](results/figures/mae_by_site.png)

## Limitations and research-use warning

Hospital/site shift, rare genotypes, high doses, missingness, and stable-dose definition differences may limit generalization. **Research use only: estimates are not prescribing guidance, a medical device, or a substitute for clinician-guided INR monitoring.**

## Legacy archive

`archive/` preserves historical source code only; patient-level files, fitted models, and notebooks with embedded outputs were removed. The archive is not part of the reproducible research pipeline.
