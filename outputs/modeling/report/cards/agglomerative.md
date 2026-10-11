### ผลจริงของ `agglomerative`

| variant | split | params | silhouette | inertia |
| --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"k": 2, "linkage": "ward"} | 0.3602 | 1070.4207 |
| with_spike | test (one-shot) | {"k": 2, "linkage": "ward"} | 0.3221 | 2096.4419 |
| non_spike | validation (selection score) | {"k": 2, "linkage": "average"} | 0.4642 | 1844.8884 |
| non_spike | test (one-shot) | {"k": 2, "linkage": "average"} | 0.4822 | 3532.7917 |
