### Headline vs classroom model on Validation (paired block bootstrap)

| task | variant | metric | classroom_model | classroom_score | headline | headline_score | delta | delta_low | delta_high | improvement_percent | interval_excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | with_spike | average_precision | logistic | 0.2330 | mlp | 0.2668 | 0.0338 | -0.0038 | 0.0699 | 14.5214 | no |
| classification | non_spike | average_precision | logistic | 0.2513 | random_forest | 0.3010 | 0.0497 | -0.0174 | 0.1004 | 19.7965 | no |
| regression | with_spike | rmse | multiple_linear | 0.0689 | elastic_net | 0.0668 | -0.0021 | -0.0059 | 0.0014 | 2.9883 | no |
| regression | non_spike | rmse | multiple_linear | 0.0668 | elastic_net | 0.0643 | -0.0026 | -0.0068 | 0.0020 | 3.8380 | no |
