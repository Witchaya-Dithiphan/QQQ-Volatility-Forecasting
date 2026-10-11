### ผลจริงของ `mlp`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"activation": "relu", "batch": 32, "hidden_layers": [16], "l2": 0, "learning_rate": 0.001, "max_epochs": 1000, "min_delta": 1e-06, "momentum": 0.9, "patience": 50, "random_state": 42} | 0.2668 | 0.6757 | 0.3200 | 0.2353 | 0.5000 | 0.9763 | 0.9084 |
| with_spike | test (one-shot) | {"activation": "relu", "batch": 32, "hidden_layers": [16], "l2": 0, "learning_rate": 0.001, "max_epochs": 1000, "min_delta": 1e-06, "momentum": 0.9, "patience": 50, "random_state": 42} | 0.5991 | 0.7930 | 0.5714 | 0.4932 | 0.6792 | 0.9433 | 0.8552 |
| non_spike | validation (selection score) | {"activation": "relu", "batch": 32, "hidden_layers": [16, 8], "l2": 0, "learning_rate": 0.01, "max_epochs": 1000, "min_delta": 1e-06, "momentum": 0.9, "patience": 50, "random_state": 42} | 0.2808 | 0.6992 | 0.3077 | 0.2353 | 0.4444 | 0.9703 | 0.9030 |
| non_spike | test (one-shot) | {"activation": "relu", "batch": 32, "hidden_layers": [16, 8], "l2": 0, "learning_rate": 0.01, "max_epochs": 1000, "min_delta": 1e-06, "momentum": 0.9, "patience": 50, "random_state": 42} | 0.5639 | 0.7894 | 0.5039 | 0.4384 | 0.5926 | 0.9267 | 0.8311 |
