"""Tests for snapshot verification and separate Nasdaq refresh downloads."""

import asyncio
import csv
import hashlib
import io
import json
import sys
from datetime import date
from pathlib import Path
from urllib.error import URLError

import pytest

from src.download_qqq_data import (
    DataContractError,
    _fetch_json_sync,
    build_latest_download_url,
    fetch_nasdaq_history,
    load_snapshot_manifest,
    main,
    normalize_latest_payload,
    profile_csv,
    refresh_latest_data,
    validate_latest_rows,
    verify_snapshot,
    write_csv_atomically,
    write_json_atomically,
)


def _csv_bytes(rows: list[list[str]]) -> bytes:
    """Build deterministic CRLF CSV fixture bytes."""
    lines = ["Date,Close,Volume,Open,High,Low", *[",".join(row) for row in rows]]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _write_manifest(path: Path, content: bytes, rows: int) -> None:
    """Write a manifest matching a small snapshot fixture."""
    payload = {
        "source": "Nasdaq QQQ Historical Data",
        "symbol": "QQQ",
        "dataset": {
            "date_min": "2024-01-02",
            "date_max": "2024-01-03",
            "row_count": rows,
            "columns": ["Date", "Close", "Volume", "Open", "High", "Low"],
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _api_payload() -> dict[str, object]:
    """Return a minimal mocked Nasdaq response."""
    return {
        "status": {"rCode": 200},
        "data": {
            "tradesTable": {
                "rows": [
                    {
                        "date": "01/03/2024",
                        "close": "$402.00",
                        "volume": "11,000,000",
                        "open": "401.00",
                        "high": "403.00",
                        "low": "400.00",
                    },
                    {
                        "date": "01/02/2024",
                        "close": "400.00",
                        "volume": "10,000,000",
                        "open": "399.00",
                        "high": "401.00",
                        "low": "398.00",
                    },
                ]
            }
        },
    }


def test_snapshot_checksum_and_contract_pass(tmp_path: Path) -> None:
    """Snapshot verification covers bytes, shape, dates, duplicates, and missing data."""
    content = _csv_bytes(
        [
            ["01/03/2024", "402", "11000000", "401", "403", "400"],
            ["01/02/2024", "400", "10000000", "399", "401", "398"],
        ]
    )
    snapshot = tmp_path / "snapshot.csv"
    manifest = tmp_path / "manifest.json"
    snapshot.write_bytes(content)
    _write_manifest(manifest, content, rows=2)

    profile = verify_snapshot(snapshot, manifest)

    assert profile.row_count == 2
    assert profile.date_min == "2024-01-02"
    assert profile.date_max == "2024-01-03"
    assert profile.duplicate_dates == 0
    assert profile.missing_ohlcv == 0


def test_snapshot_checksum_change_is_rejected(tmp_path: Path) -> None:
    """A byte-level change fails even when the CSV values remain parseable."""
    original = _csv_bytes(
        [
            ["01/03/2024", "402", "11000000", "401", "403", "400"],
            ["01/02/2024", "400", "10000000", "399", "401", "398"],
        ]
    )
    snapshot = tmp_path / "snapshot.csv"
    manifest = tmp_path / "manifest.json"
    snapshot.write_bytes(original.replace(b"402", b"402.0", 1))
    _write_manifest(manifest, original, rows=2)

    with pytest.raises(DataContractError, match="SHA-256"):
        verify_snapshot(snapshot, manifest)


def test_snapshot_duplicate_and_missing_values_are_rejected(tmp_path: Path) -> None:
    """Semantic checks fail independently of file identity."""
    content = _csv_bytes(
        [
            ["01/02/2024", "400", "10000000", "399", "401", ""],
            ["01/02/2024", "401", "11000000", "400", "402", "399"],
        ]
    )
    snapshot = tmp_path / "snapshot.csv"
    manifest = tmp_path / "manifest.json"
    snapshot.write_bytes(content)
    _write_manifest(manifest, content, rows=2)

    with pytest.raises(DataContractError, match=r"duplicate dates.*missing OHLCV"):
        verify_snapshot(snapshot, manifest)


def test_latest_contract_has_no_fixed_snapshot_size_or_start_date() -> None:
    """A small mocked rolling response is valid without 2,512 rows or a 2016 start."""
    rows = normalize_latest_payload(_api_payload())

    assert len(rows) == 2
    assert rows[-1]["Date"] == "2024-01-02"
    assert rows[0]["Close"] == "402.00"
    assert rows[0]["Volume"] == "11000000"


def test_latest_url_uses_a_rolling_ten_year_request() -> None:
    """Refresh requests are based on their run date rather than snapshot dates."""
    url = build_latest_download_url(date(2026, 10, 5))

    assert "fromdate=2016-10-05" in url
    assert "todate=2026-10-05" in url
    assert "limit=5000" in url


def test_latest_url_handles_leap_day() -> None:
    """Rolling requests map leap day safely into a non-leap start year."""
    url = build_latest_download_url(date(2024, 2, 29))

    assert "fromdate=2014-02-28" in url


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": {"rCode": 400}}, "rejected"),
        ({"status": {"rCode": 200}, "data": None}, "no historical rows"),
        (
            {
                "status": {"rCode": 200},
                "data": {"tradesTable": {"rows": ["invalid"]}},
            },
            "invalid historical row",
        ),
        (
            {
                "status": {"rCode": 200},
                "data": {"tradesTable": {"rows": [{"date": "bad"}]}},
            },
            "invalid date",
        ),
    ],
)
def test_latest_payload_rejects_invalid_api_shapes(
    payload: dict[str, object], message: str
) -> None:
    """Malformed and unsuccessful mocked API responses fail clearly."""
    with pytest.raises(DataContractError, match=message):
        normalize_latest_payload(payload)


def test_latest_payload_rejects_missing_and_invalid_numbers() -> None:
    """Missing and nonnumeric OHLCV values cannot enter a refresh file."""
    missing = _api_payload()
    missing_data = missing["data"]
    assert isinstance(missing_data, dict)
    missing_rows = missing_data["tradesTable"]["rows"]  # type: ignore[index]
    missing_rows[0]["close"] = None  # type: ignore[index]
    with pytest.raises(DataContractError, match="no close"):
        normalize_latest_payload(missing)

    invalid = _api_payload()
    invalid_data = invalid["data"]
    assert isinstance(invalid_data, dict)
    invalid_rows = invalid_data["tradesTable"]["rows"]  # type: ignore[index]
    invalid_rows[0]["volume"] = "unknown"  # type: ignore[index]
    with pytest.raises(DataContractError, match="invalid volume"):
        normalize_latest_payload(invalid)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "zero rows"),
        ([{"Date": "2024-01-02"}, {"Date": "2024-01-02"}], "duplicate dates"),
        (
            [{"Date": "2024-01-02"}, {"Date": "2024-01-03"}],
            "not reverse chronological",
        ),
        ([{"Date": "2024-01-02"}], "is missing"),
    ],
)
def test_latest_row_contract_rejects_invalid_rows(
    rows: list[dict[str, str]], message: str
) -> None:
    """Latest validation enforces quality without a fixed historical size."""
    with pytest.raises(DataContractError, match=message):
        validate_latest_rows(rows)


def test_refresh_never_writes_over_snapshot_even_with_overwrite() -> None:
    """The canonical snapshot path is permanently protected from refreshes."""
    with pytest.raises(DataContractError, match="snapshot"):
        write_csv_atomically([], Path("data/raw/qqq_daily.csv"), overwrite=True)


def test_refresh_writer_refuses_existing_file_without_overwrite(tmp_path: Path) -> None:
    """A latest file requires explicit overwrite authorization."""
    output = tmp_path / "latest.csv"
    output.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_csv_atomically([], output)

    assert output.read_text(encoding="utf-8") == "original"


def test_json_writer_replaces_report_atomically(tmp_path: Path) -> None:
    """A refresh report can replace its previous generated version."""
    report = tmp_path / "reports" / "latest.json"

    write_json_atomically({"rows": 2}, report)
    write_json_atomically({"rows": 3}, report)

    assert json.loads(report.read_text(encoding="utf-8")) == {"rows": 3}


def test_profile_and_manifest_errors_are_clear(tmp_path: Path) -> None:
    """Missing files, bad schemas, empty CSVs, dates, and manifests are rejected."""
    with pytest.raises(FileNotFoundError):
        profile_csv(tmp_path / "missing.csv")

    bad_schema = tmp_path / "bad_schema.csv"
    bad_schema.write_text("Date,Close\n2024-01-02,100\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="missing required columns"):
        profile_csv(bad_schema)

    empty = tmp_path / "empty.csv"
    empty.write_text("Date,Close,Volume,Open,High,Low\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="no data rows"):
        profile_csv(empty)

    bad_date = tmp_path / "bad_date.csv"
    bad_date.write_text(
        "Date,Close,Volume,Open,High,Low\ninvalid,100,1,99,101,98\n",
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="Invalid Date"):
        profile_csv(bad_date)

    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(DataContractError, match="Invalid snapshot manifest"):
        load_snapshot_manifest(bad_manifest)


def test_fetch_helpers_use_mocked_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch helpers are covered without network I/O."""
    response = json.dumps(_api_payload()).encode("utf-8")
    monkeypatch.setattr(
        "src.download_qqq_data.urlopen",
        lambda request, timeout: io.BytesIO(response),
    )

    assert _fetch_json_sync("https://example.test", 1.0)["status"] == {"rCode": 200}
    assert asyncio.run(fetch_nasdaq_history("https://example.test"))["data"]


def test_fetch_rejects_network_error_and_non_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors and JSON arrays become domain-specific errors."""

    def raise_url_error(request: object, timeout: float) -> io.BytesIO:
        raise URLError("offline")

    monkeypatch.setattr("src.download_qqq_data.urlopen", raise_url_error)
    with pytest.raises(DataContractError, match="Could not download"):
        _fetch_json_sync("https://example.test", 1.0)

    monkeypatch.setattr(
        "src.download_qqq_data.urlopen",
        lambda request, timeout: io.BytesIO(b"[]"),
    )
    with pytest.raises(DataContractError, match="non-object"):
        _fetch_json_sync("https://example.test", 1.0)


def test_refresh_uses_mock_response_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refresh workflow is tested without making a real Nasdaq request."""

    async def fake_fetch(url: str, timeout_seconds: float = 60.0) -> dict[str, object]:
        assert "fromdate=2014-01-03" in url
        assert timeout_seconds == 60.0
        return _api_payload()

    monkeypatch.setattr("src.download_qqq_data.fetch_nasdaq_history", fake_fetch)
    output = tmp_path / "qqq_daily_latest.csv"
    report = tmp_path / "download_report.json"

    profile = asyncio.run(
        refresh_latest_data(
            output_path=output,
            report_path=report,
            as_of_date=date(2024, 1, 3),
        )
    )

    assert profile.row_count == 2
    assert profile.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    saved_report = json.loads(report.read_text(encoding="utf-8"))
    assert saved_report["workflow"] == "refresh_latest"
    assert saved_report["dataset"]["row_count"] == 2
    assert saved_report["snapshot_modified"] is False
    with output.open(encoding="utf-8", newline="") as output_file:
        assert next(csv.reader(output_file)) == [
            "Date",
            "Close",
            "Volume",
            "Open",
            "High",
            "Low",
        ]


def test_cli_runs_both_workflows_with_mocked_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both CLI subcommands dispatch correctly without live network access."""
    content = _csv_bytes(
        [
            ["01/03/2024", "402", "11000000", "401", "403", "400"],
            ["01/02/2024", "400", "10000000", "399", "401", "398"],
        ]
    )
    snapshot = tmp_path / "snapshot.csv"
    manifest = tmp_path / "manifest.json"
    snapshot.write_bytes(content)
    _write_manifest(manifest, content, rows=2)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_qqq_data.py",
            "verify-snapshot",
            "--snapshot",
            str(snapshot),
            "--manifest",
            str(manifest),
        ],
    )

    main()

    assert '"row_count": 2' in capsys.readouterr().out

    async def fake_fetch(url: str, timeout_seconds: float = 60.0) -> dict[str, object]:
        return _api_payload()

    monkeypatch.setattr("src.download_qqq_data.fetch_nasdaq_history", fake_fetch)
    latest = tmp_path / "latest.csv"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_qqq_data.py",
            "refresh-latest",
            "--output",
            str(latest),
            "--report",
            str(report),
        ],
    )

    main()

    assert '"row_count": 2' in capsys.readouterr().out


def test_cli_converts_contract_error_to_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI validation failures return a concise nonzero exit."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_qqq_data.py",
            "verify-snapshot",
            "--snapshot",
            str(tmp_path / "missing.csv"),
        ],
    )

    with pytest.raises(SystemExit, match="ERROR"):
        main()
