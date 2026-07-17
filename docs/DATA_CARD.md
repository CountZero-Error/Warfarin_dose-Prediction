# Data Card: Public IWPC Workbook

## Provenance and scope

This project uses the public IWPC workbook distributed by PharmGKB at <https://api.pharmgkb.org/v1/download/submission/553247439>. The reviewed source checksum is `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`. The source workbook has a 68-field context; this project validates and uses only the documented fields required for cohort construction and pre-treatment features.

## Cohort and sites

The eligible cohort requires a recorded stable-dose flag, finite positive therapeutic weekly dose, and project site. The original public data are expected to contain 21 sites for audit; this is an expectation to check, not a guaranteed property of any derived subset. Stable-dose distributions and missingness may differ across sites.

## Identifiers, missingness, and sensitive fields

Subject/sample identifiers are used only to construct internal deduplication and split keys; they are never learned features or inference inputs. Missing clinical and genotype values are retained for pipeline imputation/audit rather than silently complete-case filtered in the primary analysis. CYP2C9 and VKORC1 labels are normalized with unknown/no-call values retained as such. Race is retained only as an audit field and excluded from learned inputs.

## Data handling

The checksum is verified before a public workbook is read. To update the reviewed source, record the public URL, file size, SHA-256, schema review, and any downstream reproducibility impact in a committed change before updating constants. Do not claim rights to redistribute the workbook beyond the public source's own terms; direct users to PharmGKB for access.
