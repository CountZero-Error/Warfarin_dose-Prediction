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
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *_, **__: Response(payload))

    manifest = data.download_data(
        tmp_path / "raw.xls", expected_sha256=expected, expected_size=len(payload)
    )

    assert (tmp_path / "raw.xls").read_bytes() == payload
    assert manifest["sha256"] == expected
    assert manifest["resolved_url"].endswith("example.xls")
    assert not (tmp_path / "raw.xls.part").exists()


def test_download_removes_partial_file_on_checksum_change(tmp_path, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *_, **__: Response(b"changed"))

    with pytest.raises(ValueError, match="checksum"):
        data.download_data(tmp_path / "raw.xls", expected_sha256="0" * 64)

    assert not (tmp_path / "raw.xls").exists()
    assert not (tmp_path / "raw.xls.part").exists()


def test_size_mismatch_preserves_existing_destination_and_manifest(tmp_path, monkeypatch):
    destination = tmp_path / "raw.xls"
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    destination.write_bytes(b"existing verified data")
    manifest.write_bytes(b'{"sha256": "existing"}\n')
    payload = b"changed"
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *_, **__: Response(payload))

    with pytest.raises(ValueError, match="file size"):
        data.download_data(
            destination,
            expected_sha256=data.hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload) + 1,
        )

    assert destination.read_bytes() == b"existing verified data"
    assert manifest.read_bytes() == b'{"sha256": "existing"}\n'
    assert not destination.with_suffix(destination.suffix + ".part").exists()


def test_download_data_command_uses_downloader_without_network(monkeypatch, capsys):
    from warfarin_dose import cli

    monkeypatch.setattr(
        cli,
        "download_data",
        lambda destination: {"sha256": "verified", "path": str(destination)},
    )

    assert cli.main(["download-data", "--output", "custom.xls"]) == 0
    assert capsys.readouterr().out == "verified verified at custom.xls\n"
