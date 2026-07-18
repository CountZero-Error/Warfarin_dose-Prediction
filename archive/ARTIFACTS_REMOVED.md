# Removed legacy artifacts

The research rebuild excludes legacy patient-row files, transformed datasets, fitted model and
scaler binaries, generated performance files, and notebooks containing embedded data/output.
Only selected historical source code remains under `archive/`.

The public source can be reacquired through the tested downloader:

```bash
warfarin-dose download-data --output data/raw/PS206767-553247439.xls
```

- Source: International Warfarin Pharmacogenetics Consortium submission distributed by PharmGKB
- Endpoint: <https://api.pharmgkb.org/v1/download/submission/553247439>
- Reviewed SHA-256: `0d95eacbcaf747638825c50a0c81ab1932a450b85e88a02990c98d26e7da5a6d`

Removing files from the current tree does not erase them from pre-existing Git history. Before
making a repository with that history public, create a sanitized history (or a clean new public
repository) that contains only the rebuilt source and curated aggregate results. History rewriting
is intentionally not performed automatically because it changes published commit identities.
