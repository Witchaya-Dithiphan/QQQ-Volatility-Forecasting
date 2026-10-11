### ผลจริงของ `adaboost`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.2363 | 0.6768 | 0.2222 | 0.1765 | 0.3000 | 0.9585 | 0.8868 |
| with_spike | test (one-shot) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.4858 | 0.7614 | 0.4853 | 0.4521 | 0.5238 | 0.9000 | 0.8123 |
| non_spike | validation (selection score) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.2407 | 0.6409 | 0.2353 | 0.1765 | 0.3529 | 0.9674 | 0.8949 |
| non_spike | test (one-shot) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.5443 | 0.7735 | 0.5000 | 0.4110 | 0.6383 | 0.9433 | 0.8391 |
