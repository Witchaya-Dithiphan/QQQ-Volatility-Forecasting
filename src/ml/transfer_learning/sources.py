"""Other-ETF source data for pretraining, fetched from the same Nasdaq endpoint as QQQ.

Same vendor means the same price policy as the accepted QQQ snapshot (no dividend
adjustment, no Adj Close), and the accepted cleaning, feature and target functions are
reused unchanged. Every source is cut at the last QQQ Train date *before* targets are
computed, so no source label looks past the day QQQ Train ends: market-wide volatility
in the Validation/Test period would otherwise leak into pretraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from src.build_features import build_all_features
from src.build_targets import add_regression_target
from src.clean_data import clean_qqq_data
from src.download_qqq_data import _fetch_json_sync, _subtract_years

from ..core.contracts import FEATURE_COLUMNS, REGRESSION_TARGET

ROOT = PROJECT_ROOT / "outputs" / "experiments" / "transfer_learning"
SOURCE_DIR = ROOT / "sources"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"
TICKERS = ("SPY", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "EFA", "EEM")
# Last accepted QQQ Train date; source rows whose 5-day label window ends later are dropped.
SOURCE_CUTOFF = np.datetime64("2023-08-18")
API_URL = "https://api.nasdaq.com/api/quote/{ticker}/historical"
COLUMNS = ("Date", "Close", "Volume", "Open", "High", "Low")
HORIZON = 5
# A one-day move this large in a broad ETF means an unadjusted split, not a market move.
SPLIT_SUSPECT_RETURN = 0.25


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF.

    Git stores these CSVs with LF while the Windows run wrote CRLF, so hashing raw bytes
    would make every source look changed on a fresh clone (as core.artifacts does for JSON).
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def download_url(ticker: str, as_of: date) -> str:
    query = urlencode({"assetclass": "etf", "fromdate": _subtract_years(as_of, 10).isoformat(),
                       "todate": as_of.isoformat(), "limit": 5000})
    return API_URL.format(ticker=ticker) + "?" + query


def payload_rows(payload: dict) -> list[dict[str, str]]:
    """Raw Nasdaq rows as strings, oldest first; 'N/A' is left for clean_qqq_data to drop."""
    status = payload.get("status")
    if not isinstance(status, dict) or status.get("rCode") != 200:
        raise ValueError(f"Nasdaq rejected the request: {status!r}")
    rows = payload["data"]["tradesTable"]["rows"]
    out = []
    for row in rows:
        month, day, year = (int(part) for part in row["date"].split("/"))
        out.append({"Date": date(year, month, day).isoformat(),
                    **{c: str(row[c.lower()]).replace("$", "").replace(",", "") for c in COLUMNS[1:]}})
    return sorted(out, key=lambda r: r["Date"])


def download(ticker: str, as_of: date, directory: Path = SOURCE_DIR) -> dict:
    url = download_url(ticker, as_of)
    rows = payload_rows(_fetch_json_sync(url, 60.0))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticker}_daily.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"ticker": ticker, "url": url, "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "rows": len(rows), "date_min": rows[0]["Date"], "date_max": rows[-1]["Date"],
            "sha256": text_sha256(path)}


def featurize(raw: pd.DataFrame, cutoff: np.datetime64 | None = None) -> tuple[pd.DataFrame, dict]:
    """Accepted cleaning + features (+ forward target when cut) on one ticker's OHLCV.

    With a cutoff, rows after it are removed *before* the target is computed, so the
    last five rows lose their label instead of borrowing returns from after the cutoff.
    """
    clean, report = clean_qqq_data(raw)
    if cutoff is not None:
        clean = clean.loc[clean["Date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
    # The target validator recomputes return_1d from Close, so warm-up rows stay until after it.
    featured = add_regression_target(build_all_features(clean))
    positions = np.arange(len(featured))
    end = np.minimum(positions + HORIZON, len(featured) - 1)
    featured["label_end_date"] = featured["Date"].to_numpy()[end]
    featured.loc[positions + HORIZON >= len(featured), "label_end_date"] = pd.NaT
    featured = featured.loc[featured[list(FEATURE_COLUMNS)].notna().all(axis=1)].reset_index(drop=True)
    big = featured["return_1d"].abs() > SPLIT_SUSPECT_RETURN
    report = {"rows_removed_by_cleaning": report["rows_removed"], "split_suspect_days": int(big.sum())}
    return featured, report


def load_source(ticker: str, cutoff: np.datetime64 = SOURCE_CUTOFF, directory: Path = SOURCE_DIR) -> pd.DataFrame:
    """Features, target and label_end_date for one source ticker, every label inside the cutoff."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["sources"] if e["ticker"] == ticker)
    path = directory / f"{ticker}_daily.csv"
    if text_sha256(path) != entry["sha256"]:
        raise ValueError(f"Source changed on disk: {path}")
    frame, report = featurize(pd.read_csv(path, dtype=str), cutoff)
    if report["split_suspect_days"]:
        raise ValueError(f"{ticker}: {report['split_suspect_days']} day(s) look like an unadjusted split")
    frame = frame.loc[frame[REGRESSION_TARGET].notna()].reset_index(drop=True)
    if (frame["label_end_date"] > pd.Timestamp(cutoff)).any():
        raise AssertionError("A source label reaches past the cutoff")
    return frame


def qqq_extended(directory: Path = SOURCE_DIR) -> pd.DataFrame:
    """Accepted QQQ snapshot plus only the refreshed rows after it, featurized as one series.

    The refresh alone starts two months later than the snapshot, which shifts rsi_14
    (Wilder smoothing never forgets its start), so the new rows are appended to the
    snapshot instead; the overlap is checked to agree price for price.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["sources"] if e["ticker"] == "QQQ")
    if text_sha256(directory / "QQQ_daily.csv") != entry["sha256"]:
        raise ValueError("QQQ refresh changed on disk since the manifest was written")
    snapshot = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv", dtype=str)
    snapshot["Date"] = pd.to_datetime(snapshot["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    refresh = pd.read_csv(directory / "QQQ_daily.csv", dtype=str)
    overlap = refresh.merge(snapshot, on="Date", suffixes=("_new", ""))
    for column in COLUMNS[1:]:
        if not np.allclose(overlap[f"{column}_new"].astype(float), overlap[column].astype(float), rtol=0, atol=0):
            raise ValueError(f"QQQ refresh disagrees with the accepted snapshot on {column}")
    newer = refresh.loc[refresh["Date"] > snapshot["Date"].max()]
    frame, _ = featurize(pd.concat([snapshot[list(COLUMNS)], newer[list(COLUMNS)]], ignore_index=True))
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch the pretraining sources and the QQQ refresh")
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)
    entries = []
    for ticker in (*TICKERS, "QQQ"):
        entry = download(ticker, as_of)
        frame, report = featurize(pd.read_csv(SOURCE_DIR / f"{ticker}_daily.csv", dtype=str),
                                  None if ticker == "QQQ" else SOURCE_CUTOFF)
        entries.append({**entry, **report})
        print(entry["ticker"], entry["rows"], entry["date_min"], entry["date_max"], report)
    MANIFEST_PATH.write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vendor": "Nasdaq historical quote API (same endpoint family as data/raw/qqq_daily.csv)",
        "adjustment_policy": "Unadjusted OHLC as served by Nasdaq, identical to the accepted QQQ snapshot; "
                             "no Adj Close. Splits are screened with |return_1d| > 25%.",
        "source_cutoff": str(SOURCE_CUTOFF),
        "sources": entries,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
