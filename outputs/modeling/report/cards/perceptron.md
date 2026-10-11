### ผลจริงของ `perceptron`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.0648 | 0.2993 | 0.0000 | 0.0000 | 0.0000 | 0.9644 | 0.8760 |
| with_spike | test (one-shot) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.1374 | 0.2749 | 0.0577 | 0.0411 | 0.0968 | 0.9067 | 0.7373 |
| non_spike | validation (selection score) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.0742 | 0.3383 | 0.1159 | 0.1176 | 0.1143 | 0.9080 | 0.8356 |
| non_spike | test (one-shot) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.1328 | 0.2626 | 0.0813 | 0.0685 | 0.1000 | 0.8500 | 0.6971 |
