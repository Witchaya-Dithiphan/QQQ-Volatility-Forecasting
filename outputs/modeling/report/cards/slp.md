### ผลจริงของ `slp`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"batch": 32, "l2": 0, "learning_rate": 0.001, "max_epochs": 1000, "min_delta": 1e-06, "patience": 50, "random_state": 42} | 0.2427 | 0.6803 | 0.2609 | 0.1765 | 0.5000 | 0.9822 | 0.9084 |
| with_spike | test (one-shot) | {"batch": 32, "l2": 0, "learning_rate": 0.001, "max_epochs": 1000, "min_delta": 1e-06, "patience": 50, "random_state": 42} | 0.5536 | 0.7798 | 0.5041 | 0.4247 | 0.6200 | 0.9367 | 0.8365 |
| non_spike | validation (selection score) | {"batch": "full", "l2": 0, "learning_rate": 0.01, "max_epochs": 1000, "min_delta": 1e-06, "patience": 50, "random_state": 42} | 0.2474 | 0.6778 | 0.2258 | 0.2059 | 0.2500 | 0.9377 | 0.8706 |
| non_spike | test (one-shot) | {"batch": "full", "l2": 0, "learning_rate": 0.01, "max_epochs": 1000, "min_delta": 1e-06, "patience": 50, "random_state": 42} | 0.4963 | 0.7420 | 0.5000 | 0.4795 | 0.5224 | 0.8933 | 0.8123 |
