### ผลจริงของ `knn`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"k": 15, "weights": "distance"} | 0.2037 | 0.6987 | 0.2000 | 0.1471 | 0.3125 | 0.9674 | 0.8922 |
| with_spike | test (one-shot) | {"k": 15, "weights": "distance"} | 0.5472 | 0.7847 | 0.5512 | 0.4795 | 0.6481 | 0.9367 | 0.8472 |
| non_spike | validation (selection score) | {"k": 9, "weights": "distance"} | 0.2082 | 0.6455 | 0.2222 | 0.1765 | 0.3000 | 0.9585 | 0.8868 |
| non_spike | test (one-shot) | {"k": 9, "weights": "distance"} | 0.5389 | 0.8008 | 0.4237 | 0.3425 | 0.5556 | 0.9333 | 0.8177 |
