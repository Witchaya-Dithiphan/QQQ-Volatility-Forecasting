### ผลจริงของ `gaussian_nb`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"var_smoothing": 1e-07} | 0.2218 | 0.6665 | 0.2812 | 0.2647 | 0.3000 | 0.9377 | 0.8760 |
| with_spike | test (one-shot) | {"var_smoothing": 1e-07} | 0.5538 | 0.7703 | 0.5806 | 0.6164 | 0.5488 | 0.8767 | 0.8257 |
| non_spike | validation (selection score) | {"var_smoothing": 1e-08} | 0.2292 | 0.6789 | 0.2381 | 0.2941 | 0.2000 | 0.8813 | 0.8275 |
| non_spike | test (one-shot) | {"var_smoothing": 1e-08} | 0.5884 | 0.7779 | 0.5824 | 0.7260 | 0.4862 | 0.8133 | 0.7962 |
