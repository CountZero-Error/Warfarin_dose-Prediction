# Warfarin Dose Prediction Research Rebuild Design

**Status:** Draft for written review  
**Date:** 2026-07-17  
**Primary domain:** Biomedical informatics / clinical pharmacogenomics  

## 1. Goal

Rebuild the repository as a reproducible clinical-ML research project that estimates a patient's stable therapeutic **weekly warfarin dose in mg/week** from information available before treatment: demographics, anthropometrics, clinical history, medications, and pharmacogenomic variables.

The project must demonstrate more than model fitting. It must show correct clinical target definition, data provenance, leakage control, site-aware validation, uncertainty analysis, subgroup auditing, reproducible experiments, and honest comparison with published dosing algorithms.

This is a research and education project. It is not a prescribing system, medical device, or substitute for clinician-guided INR monitoring.

## 2. Scope and non-goals

### In scope

- A fully reproducible pipeline from public raw data to cohort, models, tables, figures, and saved predictions.
- Stable-dose regression using clinical and pharmacogenomic data.
- Published clinical and pharmacogenetic IWPC comparators.
- Leakage-safe grouped validation across data-collection sites.
- Missing-data, feature-ranking, ablation, uncertainty, and subgroup experiments.
- A command-line research prediction demonstration using the final frozen pipeline.
- A manuscript-style report, data card, model card, tests, and continuous integration.

### Out of scope

- A web application or Streamlit presentation.
- Real-time dose adjustment, longitudinal INR control, or treatment recommendations.
- Claims of causal feature effects.
- Credential-gated or private clinical data.
- A general AutoML framework.
- Deep architectures whose complexity is not justified by this small tabular dataset.

The legacy application and neural-network implementation will remain available under `archive/` for provenance but will not be imported by the rebuilt package.

## 3. Data source, target, and cohort

### 3.1 Source

The primary source is the publicly downloadable International Warfarin Pharmacogenetics Consortium (IWPC) patient-level dataset already used by the repository. The canonical endpoint is `https://api.pharmgkb.org/v1/download/submission/553247439`, which currently redirects to PharmGKB's public object store. The acquisition command saves `data/raw/PS206767-553247439.xls` and reads the legacy Excel format with `xlrd`.

The source retrieved on 2026-07-17 is 5,083,136 bytes with SHA-256 `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`. The downloader verifies this checksum and records the source URL, retrieval time, file size, and checksum in a run manifest. An upstream checksum change must stop the run and require an explicit, reviewed dataset-version update.

No credentialed dataset or silent synthetic fallback is allowed. A failed download, changed checksum, or incompatible schema must stop with an actionable error.

The currently inspected workbook contains approximately 5,700 patients, 68 source fields, and 21 sites. Approximately 5,410 records are expected to have a stable-dose indicator and a nonmissing therapeutic dose. These are audit expectations, not values to force: the pipeline will report the observed counts and fail only when required columns or invariants are absent.

### 3.2 Target

The canonical target is the source field representing therapeutic stable warfarin dose in **mg/week**.

- All training, metrics, tables, and saved predictions use mg/week.
- Average mg/day may be displayed only as `weekly_dose / 7` and must be labeled as a derived display value.
- The legacy `/6` conversion is a documented historical bug and must not be reproduced.
- A fixed 35 mg/week value is a historical `5 mg/day` benchmark, not a patient recommendation.

### 3.3 Eligibility

The primary cohort includes records that:

1. are marked as having reached a stable warfarin dose;
2. have a finite, positive therapeutic weekly dose;
3. contain a usable site label for primary site-grouped validation.

Unusual but valid doses remain in the primary cohort. No dose-range exclusion is planned; adding one later requires a reviewed specification amendment and can only be a sensitivity analysis. Duplicate patient identifiers, impossible values, and conflicting records are surfaced in the data audit rather than silently removed.

### 3.4 Data-quality report

The audit produces machine-readable and human-readable outputs covering:

- cohort flow and exclusion reasons;
- missingness by feature and site;
- dose, age, height, and weight distributions;
- genotype availability and invalid genotype labels;
- site sizes and target distributions;
- duplicate identifiers and impossible values;
- comparison of the full eligible cohort with the historical complete-case cohort.

Identifiers, free-text comments, and direct site labels are removed before model matrices are created.

## 4. Features

Feature definitions are source-variable definitions, not whichever encoded columns happen to be produced by preprocessing.

### 4.1 Candidate clinical features

- Age decade parsed from the source band: for example, `50-59` becomes `5` and `90+` becomes `9`; unparseable or missing bands remain missing for fold-local handling.
- Sex/gender as represented by the source dataset.
- Height and weight.
- Indication group and target INR range.
- Diabetes.
- Congestive heart failure or cardiomyopathy.
- Valve replacement.
- Amiodarone use.
- Enzyme-inducer use as a prespecified composite of carbamazepine, phenytoin, and rifampin.
- Smoking status.
- Statin use only if at least 50% of eligible records are nonmissing, at least 95% of nonmissing values map unambiguously to yes/no, and both classes have at least 1% prevalence. Otherwise it is excluded with the audit reason recorded.

### 4.2 Pharmacogenomic features

- CYP2C9 diplotype, with a documented mapping to clinically meaningful genotype or activity groups.
- VKORC1 -1639 genotype.

Correlated alternative VKORC1 assays are not included beside the selected canonical assay. CYP4F2 is not fabricated or inferred when absent from the source data.

### 4.3 Prespecified feature sets

1. **Published IWPC comparators:** exact variables and coding required by the published clinical and pharmacogenetic equations, including race where the historical formula requires it.
2. **Clinical ML:** eligible clinical variables without genotype and without race.
3. **Pharmacogenomic ML:** the clinical ML set plus CYP2C9 and VKORC1.

Race is retained only to reproduce and audit the historical comparator and to describe subgroup performance. It is not a candidate input to the primary learned models. Site is used only for grouping and auditing, never as a predictor.

### 4.4 Forbidden predictors

The feature builder must reject rather than merely ignore:

- therapeutic dose or transformations of the target;
- INR measured after treatment or on the stable dose;
- the stable-dose eligibility flag after cohort construction;
- patient, family, and site identifiers;
- free-text comments;
- downstream treatment outcomes;
- sparse medication fields not explicitly approved by the feature dictionary.

## 5. Missing data and preprocessing

All learned preprocessing is fitted inside the relevant training fold.

- Continuous variables use training-fold median imputation plus missingness indicators.
- Binary and categorical clinical variables are normalized to documented categories, encode missing values as an explicit `Unknown` category, and never treat missing as `No`.
- Genotype missingness is represented as `Unknown`; genotypes are never statistically invented.
- Categorical encoding uses training-fold categories and handles unseen evaluation categories without failure.
- Scaling is applied only to models that require it.

The primary analysis uses fold-local imputation. Complete-case modeling is a sensitivity analysis because the legacy `dropna()` path retained only about 1,477 of 5,700 records and changed the study population.

Preprocessing and estimator are saved as one fitted pipeline so inference cannot bypass training-time transformations.

## 6. Validation and experiment data flow

### 6.1 Primary validation

The primary estimate of transportability uses leave-one-site-out outer validation. In each outer fold:

1. Hold one site completely untouched as the outer test set.
2. Use only the remaining sites for preprocessing choices, feature ranking, target-transform selection, hyperparameter selection, and interval calibration.
3. Perform inner grouped cross-validation by site using `GroupKFold(n_splits=min(5, number_of_outer_training_sites))`; fewer than three available outer-training sites is a hard failure.
4. Refit the selected complete pipeline on all outer-training patients.
5. Predict the outer site once and save patient-level predictions with anonymized row keys and fold metadata.

Every eligible patient must receive exactly one outer prediction. Split-integrity checks must prove patient and site disjointness. If the available sites cannot support the required grouped folds, the experiment stops rather than falling back silently to random splitting.

### 6.2 Secondary validation

A nested random-CV experiment is retained only to quantify how much random splitting overstates performance relative to site-held-out validation. It is labeled an optimism comparator and does not replace the primary result.

### 6.3 Final fitted model

After all outer-fold results are frozen, the final research artifact uses the prespecified selection rule and is fitted on the full eligible cohort. Its reported expected performance remains the outer site-held-out estimate, not its training score.

## 7. Models and target transformations

### 7.1 Benchmarks

- Fixed 35 mg/week benchmark.
- Published IWPC clinical equation.
- Published IWPC pharmacogenetic equation.
- Training-fold mean and median predictors as sanity checks.

Published equations are implemented from their documented coefficients and protected by unit tests using worked examples. They are not refitted.

### 7.2 Learned models

- Regularized linear regression: Ridge and Elastic Net.
- Random forest regression.
- Histogram gradient boosting regression.
- A small scikit-learn MLP regressor as a deliberately limited neural-network benchmark.

The MLP is not a privileged model. A custom PyTorch training framework is excluded unless evidence from the simpler benchmark shows a concrete need.

Hyperparameter grids remain small and are enumerated in the implementation plan. Inner mean MAE is the selection score. When configurations are within one standard error of the best, choose the simplest using the fixed order Ridge, Elastic Net, histogram gradient boosting, random forest, then MLP. No broad AutoML or unconstrained optimization is used.

### 7.3 Target representation

Each eligible learned model compares direct weekly-dose prediction with prediction of `sqrt(weekly_dose)` using inner grouped validation. Square-root predictions are clipped at zero before squaring. The chosen representation is part of the nested pipeline and cannot be selected from outer-test results.

Final predictions are constrained to be nonnegative. They are not capped at an arbitrary upper dose; predictions beyond the outer-training target range are flagged as extrapolations.

## 8. FeatRanker and feature selection

FeatRanker `0.2.0` is pinned as an analysis dependency. The project uses only its leakage-safe interface:

```python
ranker.fit(X_inner_train, y_inner_train, feature_names=...)
ranker.rank_features(
    X_inner_validation,
    y_inner_validation,
    scoring="neg_mean_absolute_error",
    feature_groups=...,
    random_state=...,
)
```

The deprecated in-sample `rankFeatures()` method is forbidden.

Encoded columns from one source variable are ranked as a semantic block. This includes genotype one-hot columns, categorical levels, and a source variable with its missingness indicator where applicable.

Feature ranking is a secondary experiment, not a prerequisite for the primary models:

1. Train the full prespecified feature-set pipeline as the primary analysis.
2. Within each inner grouped fold, rank feature blocks on the inner-validation data.
3. Aggregate median rank, rank variability, and selection frequency across successful models and inner folds; do not average heterogeneous raw importance magnitudes.
4. Evaluate prespecified top-5, top-10, and all-feature-block subsets through the same inner grouped validation.
5. Select the smallest subset within one standard error of the best inner mean MAE.
6. Evaluate the entire selection procedure on the untouched outer site.

A reduced feature set is adopted only if the nested experiment improves held-out MAE or preserves performance with materially fewer inputs. Otherwise the final conclusion is that ranking improved scientific understanding but not predictive performance.

## 9. Ablation, metrics, and statistical analysis

### 9.1 Ablations

Prespecified block ablations compare:

- demographics;
- height and weight;
- clinical conditions;
- medication exposures;
- pharmacogenomics.

Ablations use the same nested site-aware evaluation. Outer results are not used iteratively to invent new ablations. FeatRanker stability is reported separately; individual-block ablations identified after observing this cohort are excluded from confirmatory results and require a separately preregistered analysis.

### 9.2 Metrics

The primary metric is MAE in mg/week.

Secondary metrics are:

- RMSE in mg/week;
- R²;
- percentage of predictions within 20% of the observed stable dose;
- low (`<=21 mg/week`), intermediate (`>21 and <49 mg/week`), and high (`>=49 mg/week`) dose-category performance;
- site-level MAE and error distributions.

MAPE is not a primary metric because it is unstable for low doses.

### 9.3 Comparisons and confidence intervals

Model comparisons use paired outer predictions. Ninety-five-percent confidence intervals for aggregate metrics and metric differences use 2,000 iterations of a seeded bootstrap that resamples sites as clusters, not individual patients. The report emphasizes effect sizes and intervals rather than significance labels.

Subgroup reporting covers sex/gender, age group, race as an audit attribute, genotype availability or group, dose category, and site. Groups with fewer than 30 outer predictions report counts but no standalone performance estimate; no causal interpretation is made.

## 10. Prediction uncertainty

The project reports nominal 90% symmetric conformal intervals using absolute out-of-fold residuals generated only within the outer-training sites. The finite-sample quantile is computed from pooled grouped inner-CV residuals, the final estimator is refitted on the outer-training data, and the interval is applied to the outer site.

The lower interval bound is clipped at zero. Coverage, mean width, site-level coverage, and subgroup coverage are reported. Because hospitals are not exchangeable in a strict statistical sense, the report describes these as empirical cross-validated conformal intervals and does not claim guaranteed coverage under site shift.

## 11. Package architecture and interfaces

The rebuilt code lives in a small `src/warfarin_dose/` package:

- `data.py`: public download, checksum, schema validation, cohort construction, and audit tables.
- `features.py`: feature dictionary, leakage guards, fold-local preprocessing, and semantic groups.
- `models.py`: published equations, benchmark estimators, learned pipelines, and target transforms.
- `evaluation.py`: grouped splits, nested selection, FeatRanker integration, metrics, conformal intervals, and saved outer predictions.
- `reporting.py`: deterministic publication tables and figures from saved results.
- `cli.py`: thin commands that call these modules.

Notebooks may explore generated artifacts but cannot contain the only implementation of data cleaning, model fitting, or evaluation.

Primary commands:

```text
python -m warfarin_dose download-data
python -m warfarin_dose audit-data
python -m warfarin_dose run-experiment
python -m warfarin_dose build-report
python -m warfarin_dose predict --input patient.json
```

The prediction command returns weekly dose, derived average daily dose, uncertainty interval, missing-input summary, extrapolation flags, model version, and a research-only warning. It does not use prescribing language.

Generated artifacts are excluded from source control except intentionally curated small tables or figures. Each experiment writes its configuration, Git revision, package versions, seed, data checksum, cohort counts, split assignments, failures, metrics, and predictions.

## 12. Error handling

The pipeline fails loudly for:

- target-unit ambiguity;
- missing or renamed required columns;
- changed raw-data checksum without explicit acceptance;
- duplicate feature names or leakage columns;
- nonfinite targets, model scores, predictions, or intervals;
- overlapping train/test patients or sites;
- fewer than the required grouped folds;
- an experiment with no successful candidate model;
- an inference schema incompatible with the fitted pipeline.

Partial model failures are recorded in experiment artifacts and may be tolerated only when at least one prespecified candidate completes. Data exclusions always include machine-readable reasons and counts.

## 13. Testing and continuous integration

### Unit tests

- Weekly-dose target and `/7` daily conversion.
- Published IWPC equation coefficients and worked examples.
- Leakage-column rejection.
- Cohort inclusion and exclusion reasons.
- Feature mappings, genotype groups, and enzyme-inducer composite.
- Fold-local imputation and unseen categories.
- Site-disjoint nested splits and exactly-one outer prediction per patient.
- Metric and clustered-bootstrap calculations.
- Conformal quantile, nonnegative lower bound, and coverage calculations.
- FeatRanker calls use inner-validation data and semantic groups.
- Saved pipeline inference matches evaluation-time preprocessing.

### Integration tests

A tiny synthetic multi-site dataset runs the complete audit, nested experiment, report, and prediction path without downloading the real workbook. A separate local smoke test runs against the public dataset. CI does not commit or expose patient-level source data.

CI runs formatting or lint checks, unit tests, the synthetic integration test, and package building on supported Python versions.

## 14. Deliverables

- Reproducible Python package and command-line workflow.
- Public-data acquisition and data-quality report.
- Saved outer-fold predictions and run manifests.
- Baseline, model, target-transform, complete-case, random-CV, feature-selection, and ablation results.
- Publication-quality cohort, performance, calibration, uncertainty, subgroup, site, and feature-stability figures and tables.
- Manuscript-style report describing methods, results, limitations, and clinical context.
- Data card and model card.
- Tests and CI.
- Archived legacy code with its known methodological limitations documented.

## 15. Acceptance criteria

The rebuild is complete when a fresh environment can obtain the public data and reproduce the primary report with documented commands; all eligible patients receive one and only one site-held-out prediction; no post-treatment or identifier field reaches a learned model; target units remain mg/week end to end; published and fixed-dose comparators are included; uncertainty and subgroup results are reported; FeatRanker is used only inside inner validation; tests and CI pass; and all performance claims are supported by saved outer predictions.

Beating the published algorithm is not an acceptance criterion. A scientifically valid result may show that a simpler model, the full feature set, or a published comparator performs best.
