### ผลจริงของ `stacking`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.2262 | 0.6641 | 0.2353 | 0.1765 | 0.3529 | 0.9674 | 0.8949 |
| with_spike | test (one-shot) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.5131 | 0.7902 | 0.5714 | 0.5753 | 0.5676 | 0.8933 | 0.8311 |
| non_spike | validation (selection score) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.2311 | 0.6650 | 0.1961 | 0.1471 | 0.2941 | 0.9644 | 0.8895 |
| non_spike | test (one-shot) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.5499 | 0.7890 | 0.4853 | 0.4521 | 0.5238 | 0.9000 | 0.8123 |
