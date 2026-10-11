### ผลจริงของ `kmeans`

| variant | split | params | silhouette | inertia |
| --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"k": 2, "max_iter": 300, "n_init": 10, "random_state": 42, "tol": 1e-06} | 0.3666 | 1075.1372 |
| with_spike | test (one-shot) | {"k": 2, "max_iter": 300, "n_init": 10, "random_state": 42, "tol": 1e-06} | 0.3237 | 2090.4357 |
| non_spike | validation (selection score) | {"k": 2, "max_iter": 300, "n_init": 10, "random_state": 42, "tol": 1e-06} | 0.3298 | 1448.9259 |
| non_spike | test (one-shot) | {"k": 2, "max_iter": 300, "n_init": 10, "random_state": 42, "tol": 1e-06} | 0.2994 | 3276.0491 |
