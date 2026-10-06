"""Filesystem paths used by the QQQ volatility project."""

from pathlib import Path
from typing import Final

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
SPIKE_EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "processed" / "experiments"
SPIKE_INPUT_CONTRACT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "spike_input_contract.json"
)
SPIKE_ANALYSIS_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "spike_analysis.json"
)
SPIKE_EVENT_AUDIT_PATH = PROJECT_ROOT / "outputs" / "reports" / "spike_event_audit.csv"
SPIKE_DAILY_RETURN_FIGURE_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "spike_analysis" / "daily_return_spikes.png"
)
PRIMARY_SPIKE_IQR_MULTIPLIER = 3.0
SPIKE_RETURN_COLUMN: Final = "return_1d"
SPIKE_QUANTILE_METHOD: Final = "linear"
SPIKE_TARGET_BACKWARD_REACH = 5
SPIKE_FEATURE_FORWARD_REACH = 19
SPIKE_COMPARISON_RULE = "strict_greater_than"
SPIKE_COMPARISON_FORMULA: Final = "abs(return_1d) > threshold"
SPIKE_RSI_POLICY = "operational_window_s_minus_5_to_s_plus_19"
