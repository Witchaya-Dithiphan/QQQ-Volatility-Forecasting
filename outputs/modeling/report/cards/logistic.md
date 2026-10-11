### ผลจริงของ `logistic`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"class_weight": null, "l2": 0.01, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.2330 | 0.6732 | 0.2609 | 0.1765 | 0.5000 | 0.9822 | 0.9084 |
| with_spike | test (one-shot) | {"class_weight": null, "l2": 0.01, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.5518 | 0.7822 | 0.5203 | 0.4384 | 0.6400 | 0.9400 | 0.8418 |
| non_spike | validation (selection score) | {"class_weight": null, "l2": 0, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.2513 | 0.6824 | 0.2553 | 0.1765 | 0.4615 | 0.9792 | 0.9057 |
| non_spike | test (one-shot) | {"class_weight": null, "l2": 0, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.5510 | 0.7782 | 0.5041 | 0.4247 | 0.6200 | 0.9367 | 0.8365 |
