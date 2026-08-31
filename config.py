"""Filesystem paths used by the QQQ volatility project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "qqq_clean.csv"
CLEANING_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "cleaning_report.json"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "qqq_features.csv"
FEATURE_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "feature_report.json"
REGRESSION_TARGET_DATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "qqq_regression_target.csv"
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
