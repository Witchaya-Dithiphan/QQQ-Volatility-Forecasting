### Test metrics by diagnostic segment

| model | variant | segment | task | rows | positives | average_precision | roc_auc | recall | f1 | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaboost | non_spike | non_spike | classification | 319 | 35.0000 | 0.3372 | 0.7370 | 0.3714 | 0.4333 | n/a | n/a | n/a |
| adaboost | non_spike | spike_affected | classification | 54 | 38.0000 | 0.8281 | 0.6102 | 0.4474 | 0.5667 | n/a | n/a | n/a |
| adaboost | with_spike | non_spike | classification | 319 | 35.0000 | 0.3212 | 0.7251 | 0.3429 | 0.3750 | n/a | n/a | n/a |
| adaboost | with_spike | spike_affected | classification | 54 | 38.0000 | 0.6804 | 0.3413 | 0.5526 | 0.5833 | n/a | n/a | n/a |
| logistic | non_spike | non_spike | classification | 319 | 35.0000 | 0.3639 | 0.7444 | 0.2000 | 0.2917 | n/a | n/a | n/a |
| logistic | non_spike | spike_affected | classification | 54 | 38.0000 | 0.7200 | 0.3668 | 0.6316 | 0.6400 | n/a | n/a | n/a |
| logistic | with_spike | non_spike | classification | 319 | 35.0000 | 0.3882 | 0.7545 | 0.2286 | 0.3333 | n/a | n/a | n/a |
| logistic | with_spike | spike_affected | classification | 54 | 38.0000 | 0.7168 | 0.3734 | 0.6316 | 0.6400 | n/a | n/a | n/a |
| mlp | non_spike | non_spike | classification | 319 | 35.0000 | 0.4048 | 0.7454 | 0.2857 | 0.3704 | n/a | n/a | n/a |
| mlp | non_spike | spike_affected | classification | 54 | 38.0000 | 0.7310 | 0.4227 | 0.5789 | 0.6027 | n/a | n/a | n/a |
| mlp | with_spike | non_spike | classification | 319 | 35.0000 | 0.4451 | 0.7709 | 0.3429 | 0.4286 | n/a | n/a | n/a |
| mlp | with_spike | spike_affected | classification | 54 | 38.0000 | 0.7837 | 0.5296 | 0.6316 | 0.6857 | n/a | n/a | n/a |
| random_forest | non_spike | non_spike | classification | 319 | 35.0000 | 0.3898 | 0.7344 | 0.3714 | 0.4194 | n/a | n/a | n/a |
| random_forest | non_spike | spike_affected | classification | 54 | 38.0000 | 0.7664 | 0.5510 | 0.4211 | 0.5246 | n/a | n/a | n/a |
| random_forest | with_spike | non_spike | classification | 319 | 35.0000 | 0.3782 | 0.7455 | 0.3429 | 0.4000 | n/a | n/a | n/a |
| random_forest | with_spike | spike_affected | classification | 54 | 38.0000 | 0.8011 | 0.5757 | 0.5000 | 0.5938 | n/a | n/a | n/a |
| elastic_net | non_spike | non_spike | regression | 319 | n/a | n/a | n/a | n/a | n/a | 0.0692 | 0.0550 | 0.1752 |
| elastic_net | non_spike | spike_affected | regression | 54 | n/a | n/a | n/a | n/a | n/a | 0.2759 | 0.1838 | -0.3351 |
| elastic_net | with_spike | non_spike | regression | 319 | n/a | n/a | n/a | n/a | n/a | 0.0707 | 0.0564 | 0.1395 |
| elastic_net | with_spike | spike_affected | regression | 54 | n/a | n/a | n/a | n/a | n/a | 0.2564 | 0.1722 | -0.1526 |
| multiple_linear | non_spike | non_spike | regression | 319 | n/a | n/a | n/a | n/a | n/a | 0.0683 | 0.0525 | 0.1972 |
| multiple_linear | non_spike | spike_affected | regression | 54 | n/a | n/a | n/a | n/a | n/a | 0.2620 | 0.1880 | -0.2042 |
| multiple_linear | with_spike | non_spike | regression | 319 | n/a | n/a | n/a | n/a | n/a | 0.0694 | 0.0536 | 0.1714 |
| multiple_linear | with_spike | spike_affected | regression | 54 | n/a | n/a | n/a | n/a | n/a | 0.2533 | 0.1819 | -0.1253 |
