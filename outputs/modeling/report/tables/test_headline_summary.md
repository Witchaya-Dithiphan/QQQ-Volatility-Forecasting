### Locked headline models: Validation score, one-shot Test score and the best baseline

| task | variant | headline | metric | validation | test | test_low | test_high | rank_on_test | best_baseline | baseline_test | beats_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | with_spike | mlp | average_precision | 0.2668 | 0.5991 | 0.2605 | 0.7603 | 1 of 13 | baseline_persistence | 0.5365 | yes |
| classification | non_spike | random_forest | average_precision | 0.3010 | 0.5240 | 0.2161 | 0.6796 | 10 of 13 | baseline_persistence | 0.5365 | no |
| regression | with_spike | elastic_net | rmse | 0.0668 | 0.1174 | 0.0709 | 0.1746 | 2 of 4 | baseline_train_mean | 0.1380 | yes |
| regression | non_spike | elastic_net | rmse | 0.0643 | 0.1230 | 0.0688 | 0.1859 | 3 of 4 | baseline_train_mean | 0.1400 | yes |
