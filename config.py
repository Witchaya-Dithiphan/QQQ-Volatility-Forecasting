"""Filesystem paths used by the QQQ volatility project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"
RAW_DATA_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "qqq_daily_snapshot.json"
)
LATEST_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily_latest.csv"
RAW_DATA_DOWNLOAD_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "qqq_download_report.json"
)
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "qqq_clean.csv"
CLEANING_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "cleaning_report.json"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "qqq_features.csv"
FEATURE_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "feature_report.json"
REGRESSION_TARGET_DATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "qqq_regression_target.csv"
)
REGRESSION_TARGET_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "regression_target_report.json"
)
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
DATA_SPLIT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_split_report.json"
TRAIN_LABELED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_labeled.csv"
VALIDATION_LABELED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation_labeled.csv"
)
TEST_LABELED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_labeled.csv"
CLASSIFICATION_THRESHOLD_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "classification_threshold.json"
)

# Phase 2 spike-analysis contract. Threshold values remain data-derived from the
# Original Train split; only the frozen rule parameters belong in configuration.
PRIMARY_SPIKE_IQR_MULTIPLIER = 3.0
SPIKE_AFFECTED_ROWS_BEFORE = 5
SPIKE_AFFECTED_ROWS_AFTER = 19
