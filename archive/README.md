# Legacy source archive

This directory preserves selected historical source code for context only. It is not part of the
tested research pipeline and must not be used for clinical decisions.

The original project explored neural-network and classical-regression approaches to warfarin
dose prediction. Patient-row datasets, fitted models/scalers, generated performance files, and
notebooks with embedded outputs were deliberately removed from the rebuilt branch. Retained
Python/UI source may reference those unavailable artifacts and is not expected to run.

For the reproducible implementation, use the package and commands documented in the repository
root [README](../README.md). The reviewed public IWPC workbook is downloaded at runtime from
PharmGKB and verified against the pinned SHA-256; it is never committed.

See [ARTIFACTS_REMOVED.md](ARTIFACTS_REMOVED.md) for the governance rationale and source
provenance.
