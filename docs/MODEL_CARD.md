# Model Card: Site-Aware Warfarin Dose Research

## Model families and primary analysis

Candidate families are ridge, elastic net, histogram gradient boosting, random forest, and MLP regressors with direct or square-root target modes. The primary analysis compares all-feature clinical and pharmacogenomic models with leave-one-site-out outer validation, inner grouped selection, and training-only conformal calibration. Weekly mg/week is the canonical target.

## Secondary analyses and metrics

FeatRanker is a secondary feature-selection analysis, not the default primary model. Complete-case, random-CV, and ablation analyses are sensitivity analyses. Reports provide MAE, RMSE, R², percent within 20%, interval coverage, and interval width; subgroup, site, and dose-category metrics are suppressed for n < 30.

## Fairness, uncertainty, and safety

Race is excluded from learned inputs and retained only for audit. Conformal intervals summarize empirical 90% coverage in the evaluation setting; they do not guarantee coverage after hospital/site shift. Feature permutation rankings are associational and sensitive to correlated predictors.

## Intended and prohibited use

Intended use is reproducible biomedical-informatics research on the reviewed public data. It is prohibited to use the model for prescribing, clinical decision support, dose changes, or replacing clinician-guided INR monitoring.

## Known limitations

Performance may not transport across sites, data-collection practices, patient populations, rare CYP2C9/VKORC1 genotypes, or high-dose ranges. Missingness and stable-dose definitions are source-specific. **Research use only; no result or prediction is a dose recommendation.**
