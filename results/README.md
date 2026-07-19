# Site-Aware Warfarin Dose Prediction from Public IWPC Data

## Research question
Can pre-treatment clinical and pharmacogenomic information estimate stable weekly warfarin dose
under site-held-out evaluation? This report describes saved research artifacts only; it is not a
dose recommendation.

## Public data and cohort
The saved run records 5410 eligible rows across
21 sites from the reviewed public IWPC source
(SHA-256: `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`).

## Pre-treatment clinical and pharmacogenomic features
Learned inputs exclude dose, post-treatment INR, identifiers, site, and race. Missing expected
inputs are handled by the fitted preprocessing pipeline.

## Leakage-safe validation and model selection
Primary results use leave-one-site-out outer validation with training-site-only model selection
and conformal calibration. Selected models are recorded in
[selection frequencies](tables/selection_frequencies.csv).

## Primary site-held-out performance
Saved overall performance is available in [overall metrics](tables/overall_metrics.csv), [site metrics](tables/site_metrics.csv). All doses and errors are mg/week.

## Comparison with fixed and published IWPC algorithms
The fixed 35 mg/week comparator is a historical population reference corresponding to 5 mg/day,
not an individual recommendation. Published-IWPC comparator sample sizes are procedure-specific
because both equations require finite age, height, and weight; their documented missing
race/genotype terms remain supported. Exact shared finite counts and paired saved-prediction
comparisons are in [paired differences](tables/paired_differences.csv).

## Prediction uncertainty
Conformal interval coverage is empirical rather than guaranteed under hospital shift. See
[interval metrics](tables/interval_metrics.csv).

## Feature stability, ablations, and sensitivity analyses
Feature ranking, ablation, and sensitivity tables are included only when corresponding saved
analysis artifacts exist. FeatRanker importances are noncausal, associational, and
correlation-sensitive. The random-CV analysis is an optimism comparator, not primary evidence;
the site-held-out analysis remains primary.

The held-out ranked-subset metrics evaluate a nested fold-wise ranking procedure; outer folds may
select different feature blocks. They are not performance of the static full-cohort refit.
Final artifact label: `pharmacogenomic_ranked`. Final artifact inputs: `vkorc1`, `weight_kg`, `age_decade`, `cyp2c9_group`, `height_cm`.
Final-model source SHA-256: `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`.


## Subgroup and site audit
Subgroup, site, and dose-category metrics are suppressed when n < 30. Race is an audit field,
not a learned input.

## Limitations
Performance under site shift may differ. Rare genotypes, high doses, missingness, and
source-specific stable-dose definitions can limit transportability. No result is a dose
recommendation.

## Reproducibility
This report was generated from saved CSV/JSON artifacts; it neither fits models nor recomputes
predictions. Sanitized analysis-code revision: `7ea3e03dc5b2bedea4e0af48421c0e6474c24283`.
Sanitized final-model revision: `ec753167d82242a8c176f3f2f7b1e09ad4a22dea`. The source run retains a machine-readable
manifest outside this curated report.

## Research-use warning
Research use only; this estimate is not prescribing guidance, a medical device, or a substitute for clinician-guided INR monitoring.
