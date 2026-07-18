from io import BytesIO

import numpy as np
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
    captured = {}

    def open_download(request, **_):
        captured["user_agent"] = request.get_header("User-agent")
        return Response(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", open_download)

    manifest = data.download_data(
        tmp_path / "raw.xls", expected_sha256=expected, expected_size=len(payload)
    )

    assert (tmp_path / "raw.xls").read_bytes() == payload
    assert manifest["sha256"] == expected
    assert manifest["resolved_url"].endswith("example.xls")
    assert captured["user_agent"] == data.DOWNLOAD_USER_AGENT
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


def test_cohort_uses_stable_positive_weekly_dose_and_site(raw_frame):
    raw = raw_frame.copy()
    raw.loc[0, "Subject Reached Stable Dose of Warfarin"] = 0
    raw.loc[1, "Therapeutic Dose of Warfarin"] = -1
    raw.loc[2, "Project Site"] = np.nan

    cohort = data.prepare_cohort(raw)

    assert len(cohort.data) == len(raw) - 3
    assert set(cohort.exclusions["reason"]) == {"not_stable", "invalid_target", "missing_site"}
    assert cohort.data["weekly_dose_mg"].gt(0).all()
    assert cohort.data["row_key"].is_unique
    assert "PharmGKB Subject ID" not in cohort.exclusions.columns


def test_unusual_positive_dose_is_audited_not_removed(raw_frame):
    raw = raw_frame.copy()
    raw.loc[0, "Therapeutic Dose of Warfarin"] = 315.0

    cohort = data.prepare_cohort(raw)

    assert 315.0 in cohort.data["weekly_dose_mg"].to_numpy()
    assert cohort.issues.set_index("issue").loc["dose_above_200_mg_week", "count"] == 1


def test_required_schema_change_stops(raw_frame):
    with pytest.raises(ValueError, match="missing required IWPC columns"):
        data.prepare_cohort(raw_frame.drop(columns=["Project Site"]))


def test_audit_includes_hashed_exclusions_and_genotype_label_tables(raw_frame):
    raw = raw_frame.copy()
    raw.loc[0, "Subject Reached Stable Dose of Warfarin"] = 0
    raw.loc[1, "CYP2C9 consensus"] = "not-a-cyp2c9-label"
    raw.loc[2, "VKORC1 -1639 consensus"] = "not-a-vkorc1-label"

    tables = data.build_audit(raw, data.prepare_cohort(raw))

    assert "PharmGKB Subject ID" not in tables["exclusions"].columns
    assert tables["cyp2c9_observed_labels"]["count"].sum() == len(raw) - 1
    assert tables["vkorc1_observed_labels"]["count"].sum() == len(raw) - 1
    assert tables["cyp2c9_invalid_labels"].to_dict("records") == [
        {"label": "not-a-cyp2c9-label", "count": 1}
    ]
    assert tables["vkorc1_invalid_labels"].to_dict("records") == [
        {"label": "not-a-vkorc1-label", "count": 1}
    ]
    assert set(tables) >= {"feature_quality", "genotype_labels", "feature_missingness"}
    assert set(tables["feature_quality"]["measure"]) == {
        "statin_decision",
        "statin_reason",
        "age_parse_failures",
    }
    assert {
        "source_feature",
        "source_value",
        "normalized_value",
        "normalized_group",
        "count",
    } == set(tables["genotype_labels"].columns)
    assert {"feature", "missing_fraction", "unknown_count"} == set(
        tables["feature_missingness"].columns
    )


def test_audit_data_command_prints_counts_and_output_path(monkeypatch, capsys):
    from warfarin_dose import cli

    monkeypatch.setattr(
        cli,
        "write_audit",
        lambda raw_path, output_dir: {
            "source_rows": 48,
            "eligible_rows": 45,
            "sites": 6,
            "source_sha256": "not-printed",
        },
    )

    assert cli.main(["audit-data", "--input", "source.xls", "--output", "audit-output"]) == 0
    assert capsys.readouterr().out == (
        "source_rows: 48\neligible_rows: 45\nsites: 6\noutput: audit-output\n"
    )
