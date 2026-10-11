### ผลจริงของ `svm`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"C": 0.1, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.2514 | 0.6786 | 0.2553 | 0.1765 | 0.4615 | 0.9792 | 0.9057 |
| with_spike | test (one-shot) | {"C": 0.1, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.5716 | 0.7835 | 0.4874 | 0.3973 | 0.6304 | 0.9433 | 0.8365 |
| non_spike | validation (selection score) | {"C": 10, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.2501 | 0.6804 | 0.2727 | 0.1765 | 0.6000 | 0.9881 | 0.9137 |
| non_spike | test (one-shot) | {"C": 10, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.5805 | 0.7853 | 0.4587 | 0.3425 | 0.6944 | 0.9633 | 0.8418 |
