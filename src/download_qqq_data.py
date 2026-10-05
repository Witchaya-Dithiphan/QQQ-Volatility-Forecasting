"""Verify the historical QQQ snapshot or download a separate latest dataset."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in {None, ""}:  # pragma: no cover - direct-script import bootstrap
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import (
    LATEST_DATA_PATH,
    RAW_DATA_DOWNLOAD_REPORT_PATH,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
)

NASDAQ_API_URL = "https://api.nasdaq.com/api/quote/QQQ/historical"
NASDAQ_SOURCE_PAGE = "https://www.nasdaq.com/market-activity/etf/qqq/historical"
OUTPUT_COLUMNS = ("Date", "Close", "Volume", "Open", "High", "Low")
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
DEFAULT_LATEST_LIMIT = 5_000


class DataContractError(RuntimeError):
    """Raised when market data fails its snapshot or latest-data contract."""


@dataclass(frozen=True)
class DatasetProfile:
    """Structural facts calculated without modifying a CSV file."""

    row_count: int
    columns: tuple[str, ...]
    date_min: str
    date_max: str
    duplicate_dates: int
    missing_ohlcv: int
    sha256: str


@dataclass(frozen=True)
class SnapshotManifest:
    """Expected immutable facts for the experiment's historical snapshot."""

    source: str
    symbol: str
    date_min: str
    date_max: str
    row_count: int
    columns: tuple[str, ...]
    sha256: str


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest of the exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_csv_date(value: str) -> date:
    """Parse Nasdaq-style or ISO calendar dates."""
    try:
        if "/" in value:
            month, day, year = (int(part) for part in value.split("/"))
            return date(year, month, day)
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Invalid Date value: {value!r}") from exc


def profile_csv(path: Path) -> DatasetProfile:
    """Inspect snapshot structure and content without changing its bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"QQQ dataset not found: {path}")

    dates: list[date] = []
    missing_ohlcv = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise DataContractError(f"CSV has no header: {path}")
        columns = tuple(reader.fieldnames)
        missing_columns = [
            column
            for column in ("Date", *REQUIRED_OHLCV_COLUMNS)
            if column not in columns
        ]
        if missing_columns:
            raise DataContractError(
                "CSV is missing required columns: " + ", ".join(missing_columns)
            )

        for row_number, row in enumerate(reader, start=2):
            raw_date = row.get("Date")
            if raw_date is None or not raw_date.strip():
                raise DataContractError(f"Missing Date at CSV row {row_number}")
            dates.append(_parse_csv_date(raw_date.strip()))
            missing_ohlcv += sum(
                row.get(column) is None or not str(row[column]).strip()
                for column in REQUIRED_OHLCV_COLUMNS
            )

    if not dates:
        raise DataContractError(f"CSV has no data rows: {path}")
    return DatasetProfile(
        row_count=len(dates),
        columns=columns,
        date_min=min(dates).isoformat(),
        date_max=max(dates).isoformat(),
        duplicate_dates=len(dates) - len(set(dates)),
        missing_ohlcv=missing_ohlcv,
        sha256=sha256_file(path),
    )


def load_snapshot_manifest(path: Path = RAW_DATA_MANIFEST_PATH) -> SnapshotManifest:
    """Load the committed snapshot contract from JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        return SnapshotManifest(
            source=str(payload["source"]),
            symbol=str(payload["symbol"]),
            date_min=str(dataset["date_min"]),
            date_max=str(dataset["date_max"]),
            row_count=int(dataset["row_count"]),
            columns=tuple(dataset["columns"]),
            sha256=str(dataset["sha256"]).lower(),
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise DataContractError(f"Invalid snapshot manifest {path}: {exc}") from exc


def verify_snapshot(
    snapshot_path: Path = RAW_DATA_PATH,
    manifest_path: Path = RAW_DATA_MANIFEST_PATH,
) -> DatasetProfile:
    """Verify byte identity and the complete CSV contract of the old snapshot."""
    manifest = load_snapshot_manifest(manifest_path)
    profile = profile_csv(snapshot_path)
    failures: list[str] = []
    comparisons: tuple[tuple[str, object, object], ...] = (
        ("SHA-256", profile.sha256, manifest.sha256),
        ("row count", profile.row_count, manifest.row_count),
        ("columns", profile.columns, manifest.columns),
        ("minimum date", profile.date_min, manifest.date_min),
        ("maximum date", profile.date_max, manifest.date_max),
        ("duplicate dates", profile.duplicate_dates, 0),
        ("missing OHLCV values", profile.missing_ohlcv, 0),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            failures.append(f"{label}: actual={actual!r}, expected={expected!r}")
    if failures:
        raise DataContractError("Snapshot verification failed: " + "; ".join(failures))
    return profile


def _subtract_years(day: date, years: int) -> date:
    """Subtract whole years, mapping leap day to February 28 when necessary."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def build_latest_download_url(as_of_date: date) -> str:
    """Build a rolling-window Nasdaq request for a refresh dataset."""
    query = urlencode(
        {
            "assetclass": "etf",
            "fromdate": _subtract_years(as_of_date, 10).isoformat(),
            "todate": as_of_date.isoformat(),
            "limit": DEFAULT_LATEST_LIMIT,
        }
    )
    return f"{NASDAQ_API_URL}?{query}"


def _fetch_json_sync(url: str, timeout_seconds: float) -> dict[str, Any]:
    """Fetch and decode one Nasdaq response in a worker thread."""
    request = Request(
        url,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
            "User-Agent": "Mozilla/5.0 (compatible; QQQ-Volatility-Forecasting/1.0)",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Could not download Nasdaq history: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataContractError("Nasdaq returned a non-object JSON response")
    return payload


async def fetch_nasdaq_history(
    url: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Fetch Nasdaq history without blocking the calling event loop."""
    return await asyncio.to_thread(_fetch_json_sync, url, timeout_seconds)


def _plain_number(value: Any, field: str) -> str:
    """Normalize a Nasdaq price or volume while preserving exact precision."""
    if value is None:
        raise DataContractError(f"Nasdaq row has no {field} value")
    normalized = str(value).strip().replace("$", "").replace(",", "")
    try:
        Decimal(normalized)
    except InvalidOperation as exc:
        raise DataContractError(f"Nasdaq row has invalid {field}: {value!r}") from exc
    return normalized


def normalize_latest_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a successful Nasdaq response into canonical latest-data rows."""
    status = payload.get("status")
    if not isinstance(status, dict) or status.get("rCode") != 200:
        raise DataContractError(f"Nasdaq rejected the request: {status!r}")
    data = payload.get("data")
    table = data.get("tradesTable") if isinstance(data, dict) else None
    source_rows = table.get("rows") if isinstance(table, dict) else None
    if not isinstance(source_rows, list) or not source_rows:
        raise DataContractError("Nasdaq response contains no historical rows")

    rows: list[dict[str, str]] = []
    for source_row in source_rows:
        if not isinstance(source_row, dict):
            raise DataContractError("Nasdaq returned an invalid historical row")
        try:
            parsed_date = _parse_csv_date(str(source_row["date"]))
        except (KeyError, DataContractError) as exc:
            raise DataContractError(
                f"Nasdaq row has an invalid date: {source_row!r}"
            ) from exc
        rows.append(
            {
                "Date": parsed_date.isoformat(),
                "Close": _plain_number(source_row.get("close"), "close"),
                "Volume": _plain_number(source_row.get("volume"), "volume"),
                "Open": _plain_number(source_row.get("open"), "open"),
                "High": _plain_number(source_row.get("high"), "high"),
                "Low": _plain_number(source_row.get("low"), "low"),
            }
        )
    validate_latest_rows(rows)
    return rows


def validate_latest_rows(rows: list[dict[str, str]]) -> None:
    """Validate refresh data without requiring historical snapshot boundaries."""
    if not rows:
        raise DataContractError("Latest download contains zero rows")
    dates = [_parse_csv_date(row["Date"]) for row in rows]
    if len(dates) != len(set(dates)):
        raise DataContractError("Latest download contains duplicate dates")
    if dates != sorted(dates, reverse=True):
        raise DataContractError("Latest download is not reverse chronological")
    for row in rows:
        missing = [
            column for column in OUTPUT_COLUMNS if not row.get(column, "").strip()
        ]
        if missing:
            raise DataContractError(
                f"Latest row {row.get('Date', '<unknown>')} is missing: {missing}"
            )


def _same_path(first: Path, second: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return first.resolve(strict=False) == second.resolve(strict=False)


def write_csv_atomically(
    rows: list[dict[str, str]],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Write refresh rows atomically while permanently protecting the snapshot."""
    if _same_path(output_path, RAW_DATA_PATH):
        raise DataContractError(
            f"Refusing to write refresh data over the snapshot: {RAW_DATA_PATH}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_atomically(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


async def refresh_latest_data(
    output_path: Path = LATEST_DATA_PATH,
    report_path: Path = RAW_DATA_DOWNLOAD_REPORT_PATH,
    overwrite: bool = False,
    as_of_date: date | None = None,
) -> DatasetProfile:
    """Download a current rolling dataset without mutating the old snapshot."""
    requested_as_of = as_of_date or datetime.now(timezone.utc).date()
    url = build_latest_download_url(requested_as_of)
    payload = await fetch_nasdaq_history(url)
    rows = normalize_latest_payload(payload)
    write_csv_atomically(rows, output_path, overwrite=overwrite)
    profile = profile_csv(output_path)
    report: dict[str, Any] = {
        "workflow": "refresh_latest",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Nasdaq QQQ Historical Data",
        "source_page": NASDAQ_SOURCE_PAGE,
        "source_endpoint": url,
        "symbol": "QQQ",
        "output_path": str(output_path),
        "dataset": asdict(profile),
        "snapshot_modified": False,
    }
    write_json_atomically(report, report_path)
    return profile


def _build_parser() -> argparse.ArgumentParser:
    """Build the two-workflow command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    verify_parser = subparsers.add_parser(
        "verify-snapshot", help="Verify the archived experiment snapshot."
    )
    verify_parser.add_argument("--snapshot", type=Path, default=RAW_DATA_PATH)
    verify_parser.add_argument("--manifest", type=Path, default=RAW_DATA_MANIFEST_PATH)
    refresh_parser = subparsers.add_parser(
        "refresh-latest", help="Download current rolling history to a separate file."
    )
    refresh_parser.add_argument("--output", type=Path, default=LATEST_DATA_PATH)
    refresh_parser.add_argument(
        "--report", type=Path, default=RAW_DATA_DOWNLOAD_REPORT_PATH
    )
    refresh_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run snapshot verification or the separate latest-data refresh."""
    args = _build_parser().parse_args()
    try:
        if args.workflow == "verify-snapshot":
            profile = verify_snapshot(args.snapshot, args.manifest)
            print(json.dumps(asdict(profile), indent=2))
        else:
            profile = asyncio.run(
                refresh_latest_data(
                    output_path=args.output,
                    report_path=args.report,
                    overwrite=args.overwrite,
                )
            )
            print(json.dumps(asdict(profile), indent=2))
    except (DataContractError, FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
