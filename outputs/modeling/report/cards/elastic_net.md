### ผลจริงของ `elastic_net`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.0668 | 0.0539 | 0.0169 |
| with_spike | test (one-shot) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.1174 | 0.0731 | 0.2758 |
| non_spike | validation (selection score) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.0643 | 0.0510 | 0.0906 |
| non_spike | test (one-shot) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.1230 | 0.0736 | 0.2059 |
