### ผลจริงของ `random_forest`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"bootstrap": true, "max_depth": 3, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 100, "random_state": 42} | 0.2548 | 0.6862 | 0.2917 | 0.2059 | 0.5000 | 0.9792 | 0.9084 |
| with_spike | test (one-shot) | {"bootstrap": true, "max_depth": 3, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 100, "random_state": 42} | 0.5775 | 0.7866 | 0.5000 | 0.4247 | 0.6078 | 0.9333 | 0.8338 |
| non_spike | validation (selection score) | {"bootstrap": true, "max_depth": 5, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.3010 | 0.6766 | 0.2917 | 0.2059 | 0.5000 | 0.9792 | 0.9084 |
| non_spike | test (one-shot) | {"bootstrap": true, "max_depth": 5, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.5240 | 0.7814 | 0.4715 | 0.3973 | 0.5800 | 0.9300 | 0.8257 |
