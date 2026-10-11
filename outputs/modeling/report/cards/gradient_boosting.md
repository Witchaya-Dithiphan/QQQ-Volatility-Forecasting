### ผลจริงของ `gradient_boosting`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"learning_rate": 0.1, "max_depth": 1, "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.2531 | 0.6754 | 0.2449 | 0.1765 | 0.4000 | 0.9733 | 0.9003 |
| with_spike | test (one-shot) | {"learning_rate": 0.1, "max_depth": 1, "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.5332 | 0.7665 | 0.4390 | 0.3699 | 0.5400 | 0.9233 | 0.8150 |
| non_spike | validation (selection score) | {"learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.2525 | 0.6732 | 0.2667 | 0.1765 | 0.5455 | 0.9852 | 0.9111 |
| non_spike | test (one-shot) | {"learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 50, "random_state": 42} | 0.5653 | 0.7861 | 0.3762 | 0.2603 | 0.6786 | 0.9700 | 0.8311 |
