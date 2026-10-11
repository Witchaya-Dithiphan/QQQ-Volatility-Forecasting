### ผลจริงของ `xgboost`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"gamma": 0, "learning_rate": 0.03, "max_depth": 1, "n_estimators": 100, "random_state": 42, "reg_lambda": 1} | 0.2508 | 0.6693 | 0.2449 | 0.1765 | 0.4000 | 0.9733 | 0.9003 |
| with_spike | test (one-shot) | {"gamma": 0, "learning_rate": 0.03, "max_depth": 1, "n_estimators": 100, "random_state": 42, "reg_lambda": 1} | 0.5305 | 0.7716 | 0.4553 | 0.3836 | 0.5600 | 0.9267 | 0.8204 |
| non_spike | validation (selection score) | {"gamma": 0.1, "learning_rate": 0.03, "max_depth": 2, "n_estimators": 100, "random_state": 42, "reg_lambda": 10} | 0.2652 | 0.6853 | 0.2800 | 0.2059 | 0.4375 | 0.9733 | 0.9030 |
| non_spike | test (one-shot) | {"gamma": 0.1, "learning_rate": 0.03, "max_depth": 2, "n_estimators": 100, "random_state": 42, "reg_lambda": 10} | 0.5823 | 0.7887 | 0.5469 | 0.4795 | 0.6364 | 0.9333 | 0.8445 |
