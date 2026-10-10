"""Accepted feature/target identities and dataset paths. These are frozen."""
from config import (
    NON_SPIKE_TRAIN_PATH,
    TEST_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
    WITH_SPIKES_TRAIN_PATH,
)

FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "historical_volatility_5d",
    "historical_volatility_20d",
    "intraday_range",
    "sma_ratio_5_20",
    "rsi_14",
    "volume_zscore_20",
)
REGRESSION_TARGET = "target_volatility_5d"
CLASSIFICATION_TARGET = "target_high_volatility"
DATE_COLUMN = "Date"
VARIANTS = ("with_spike", "non_spike")
VARIANT_TRAIN_PATHS = {
    "with_spike": WITH_SPIKES_TRAIN_PATH,
    "non_spike": NON_SPIKE_TRAIN_PATH,
}
VALIDATION_PATH = VALIDATION_LABELED_DATA_PATH
TEST_PATH = TEST_LABELED_DATA_PATH
