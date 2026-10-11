### Headline vs classroom model on Test (paired block bootstrap)

| task | variant | metric | classroom_model | classroom_score | headline | headline_score | delta | delta_low | delta_high | improvement_percent | interval_excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | with_spike | average_precision | logistic | 0.5518 | mlp | 0.5991 | 0.0473 | -0.0137 | 0.0766 | 8.5643 | no |
| classification | non_spike | average_precision | logistic | 0.5510 | random_forest | 0.5240 | -0.0270 | -0.1678 | 0.0793 | -4.8974 | no |
| regression | with_spike | rmse | multiple_linear | 0.1158 | elastic_net | 0.1174 | 0.0016 | -0.0092 | 0.0089 | -1.4215 | no |
| regression | non_spike | rmse | multiple_linear | 0.1180 | elastic_net | 0.1230 | 0.0049 | -0.0119 | 0.0158 | -4.1832 | no |
