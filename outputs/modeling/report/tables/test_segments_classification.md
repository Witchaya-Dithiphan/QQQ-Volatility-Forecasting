### Test metrics by diagnostic segment — classification

| model | variant | segment | rows | positives | average_precision | roc_auc | recall | precision | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaboost | non_spike | non_spike | 319 | 35 | 0.3372 | 0.7370 | 0.3714 | 0.5200 | 0.4333 |
| adaboost | non_spike | spike_affected | 54 | 38 | 0.8281 | 0.6102 | 0.4474 | 0.7727 | 0.5667 |
| adaboost | with_spike | non_spike | 319 | 35 | 0.3212 | 0.7251 | 0.3429 | 0.4138 | 0.3750 |
| adaboost | with_spike | spike_affected | 54 | 38 | 0.6804 | 0.3413 | 0.5526 | 0.6176 | 0.5833 |
| logistic | non_spike | non_spike | 319 | 35 | 0.3639 | 0.7444 | 0.2000 | 0.5385 | 0.2917 |
| logistic | non_spike | spike_affected | 54 | 38 | 0.7200 | 0.3668 | 0.6316 | 0.6486 | 0.6400 |
| logistic | with_spike | non_spike | 319 | 35 | 0.3882 | 0.7545 | 0.2286 | 0.6154 | 0.3333 |
| logistic | with_spike | spike_affected | 54 | 38 | 0.7168 | 0.3734 | 0.6316 | 0.6486 | 0.6400 |
| mlp | non_spike | non_spike | 319 | 35 | 0.4048 | 0.7454 | 0.2857 | 0.5263 | 0.3704 |
| mlp | non_spike | spike_affected | 54 | 38 | 0.7310 | 0.4227 | 0.5789 | 0.6286 | 0.6027 |
| mlp | with_spike | non_spike | 319 | 35 | 0.4451 | 0.7709 | 0.3429 | 0.5714 | 0.4286 |
| mlp | with_spike | spike_affected | 54 | 38 | 0.7837 | 0.5296 | 0.6316 | 0.7500 | 0.6857 |
| random_forest | non_spike | non_spike | 319 | 35 | 0.3898 | 0.7344 | 0.3714 | 0.4815 | 0.4194 |
| random_forest | non_spike | spike_affected | 54 | 38 | 0.7664 | 0.5510 | 0.4211 | 0.6957 | 0.5246 |
| random_forest | with_spike | non_spike | 319 | 35 | 0.3782 | 0.7455 | 0.3429 | 0.4800 | 0.4000 |
| random_forest | with_spike | spike_affected | 54 | 38 | 0.8011 | 0.5757 | 0.5000 | 0.7308 | 0.5938 |
