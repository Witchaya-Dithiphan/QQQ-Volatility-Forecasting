### ผลจริงของ `simple_linear`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"feature": "historical_volatility_20d"} | 0.0690 | 0.0557 | -0.0498 |
| with_spike | test (one-shot) | {"feature": "historical_volatility_20d"} | 0.1356 | 0.0834 | 0.0347 |
| non_spike | validation (selection score) | {"feature": "return_1d"} | 0.0665 | 0.0527 | 0.0265 |
| non_spike | test (one-shot) | {"feature": "return_1d"} | 0.1372 | 0.0827 | 0.0114 |
