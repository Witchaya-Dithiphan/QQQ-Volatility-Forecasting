### ผลจริงของ `decision_tree`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"max_depth": 2, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.1732 | 0.6418 | 0.3099 | 0.3235 | 0.2973 | 0.9228 | 0.8679 |
| with_spike | test (one-shot) | {"max_depth": 2, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.3792 | 0.7450 | 0.4800 | 0.4932 | 0.4675 | 0.8633 | 0.7909 |
| non_spike | validation (selection score) | {"max_depth": 5, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.1887 | 0.6924 | 0.1887 | 0.1471 | 0.2632 | 0.9585 | 0.8841 |
| non_spike | test (one-shot) | {"max_depth": 5, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.3844 | 0.6916 | 0.4310 | 0.3425 | 0.5814 | 0.9400 | 0.8231 |
