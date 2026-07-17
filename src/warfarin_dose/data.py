from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
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
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
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
