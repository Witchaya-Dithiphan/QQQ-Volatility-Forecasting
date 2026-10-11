### ผลจริงของ `multiple_linear`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"fit_intercept": true} | 0.0689 | 0.0542 | -0.0446 |
| with_spike | test (one-shot) | {"fit_intercept": true} | 0.1158 | 0.0722 | 0.2959 |
| non_spike | validation (selection score) | {"fit_intercept": true} | 0.0668 | 0.0530 | 0.0165 |
| non_spike | test (one-shot) | {"fit_intercept": true} | 0.1180 | 0.0721 | 0.2684 |
