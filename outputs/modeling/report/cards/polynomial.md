### ผลจริงของ `polynomial`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"degree": 3, "feature": "historical_volatility_20d", "standardize_before_powers": true} | 0.0691 | 0.0552 | -0.0507 |
| with_spike | test (one-shot) | {"degree": 3, "feature": "historical_volatility_20d", "standardize_before_powers": true} | 0.1401 | 0.0861 | -0.0314 |
| non_spike | validation (selection score) | {"degree": 2, "feature": "return_5d", "standardize_before_powers": true} | 0.0661 | 0.0509 | 0.0389 |
| non_spike | test (one-shot) | {"degree": 2, "feature": "return_5d", "standardize_before_powers": true} | 0.1142 | 0.0743 | 0.3155 |
