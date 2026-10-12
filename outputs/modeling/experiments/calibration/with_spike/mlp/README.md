# Calibration diagnostic — mlp / with_spike

Calibration and threshold selection use Train walk-forward predictions only. Test was not read.

| policy | threshold | precision | recall | f1 | AP | Brier | log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_default | 0.500000 | 0.5000 | 0.2353 | 0.3200 | 0.2668 | 0.0945 | 0.3359 |
| raw_precision_threshold | 0.567130 | 0.4167 | 0.1471 | 0.2174 | 0.2668 | 0.0945 | 0.3359 |
| platt_default | 0.500000 | 0.5000 | 0.2059 | 0.2917 | 0.2668 | 0.1014 | 0.3588 |
| platt_precision_threshold | 0.517990 | 0.4167 | 0.1471 | 0.2174 | 0.2668 | 0.1014 | 0.3588 |

Minimum recall constraint selected on Train OOF: `0.40`.

## Threshold sensitivity

Every threshold below was selected on Train OOF; the right-hand metrics are diagnostics on Validation.

| method | minimum OOF recall | threshold | Validation precision | Validation recall | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.20 | 0.743527 | 0.3333 | 0.0294 | 0.0541 |
| raw | 0.30 | 0.653736 | 0.6000 | 0.0882 | 0.1538 |
| raw | 0.40 | 0.567130 | 0.4167 | 0.1471 | 0.2174 |
| raw | 0.50 | 0.489694 | 0.4444 | 0.2353 | 0.3077 |
| raw | 0.60 | 0.414218 | 0.3056 | 0.3235 | 0.3143 |
| platt | 0.20 | 0.651979 | 0.3333 | 0.0294 | 0.0541 |
| platt | 0.30 | 0.581186 | 0.6000 | 0.0882 | 0.1538 |
| platt | 0.40 | 0.517990 | 0.4167 | 0.1471 | 0.2174 |
| platt | 0.50 | 0.463591 | 0.4444 | 0.2353 | 0.3077 |
| platt | 0.60 | 0.411070 | 0.3056 | 0.3235 | 0.3143 |

The base model and its hyperparameters were selected previously; this is a diagnostic calibration experiment, not a new untouched Test result.
