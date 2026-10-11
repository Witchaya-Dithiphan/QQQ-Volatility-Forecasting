### Same model with and without spike rows, on Test (paired block bootstrap)

| task | model | metric | with_spike | non_spike | delta | delta_low | delta_high | direction | interval_excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | adaboost | average_precision | 0.4858 | 0.5443 | 0.0585 | -0.0324 | 0.1168 | non_spike better | no |
| classification | decision_tree | average_precision | 0.3792 | 0.3844 | 0.0052 | -0.1502 | 0.1190 | non_spike better | no |
| classification | gaussian_nb | average_precision | 0.5538 | 0.5884 | 0.0346 | -0.0064 | 0.0632 | non_spike better | no |
| classification | gradient_boosting | average_precision | 0.5332 | 0.5653 | 0.0322 | -0.0254 | 0.0891 | non_spike better | no |
| classification | knn | average_precision | 0.5472 | 0.5389 | -0.0083 | -0.1005 | 0.0946 | with_spike better | no |
| classification | logistic | average_precision | 0.5518 | 0.5510 | -0.0008 | -0.0184 | 0.0156 | with_spike better | no |
| classification | mlp | average_precision | 0.5991 | 0.5639 | -0.0352 | -0.0792 | 0.0276 | with_spike better | no |
| classification | perceptron | average_precision | 0.1374 | 0.1328 | -0.0046 | -0.0313 | -0.0001 | with_spike better | yes |
| classification | random_forest | average_precision | 0.5775 | 0.5240 | -0.0534 | -0.1284 | 0.0106 | with_spike better | no |
| classification | slp | average_precision | 0.5536 | 0.4963 | -0.0573 | -0.1184 | -0.0038 | with_spike better | yes |
| classification | stacking | average_precision | 0.5131 | 0.5499 | 0.0368 | -0.0721 | 0.1233 | non_spike better | no |
| classification | svm | average_precision | 0.5716 | 0.5805 | 0.0088 | -0.0015 | 0.0184 | non_spike better | no |
| classification | xgboost | average_precision | 0.5305 | 0.5823 | 0.0518 | 0.0163 | 0.0931 | non_spike better | yes |
| regression | elastic_net | rmse | 0.1174 | 0.1230 | 0.0055 | -0.0045 | 0.0122 | with_spike better | no |
| regression | multiple_linear | rmse | 0.1158 | 0.1180 | 0.0022 | -0.0017 | 0.0051 | with_spike better | no |
| regression | polynomial | rmse | 0.1401 | 0.1142 | -0.0260 | -0.0575 | 0.0041 | non_spike better | no |
| regression | simple_linear | rmse | 0.1356 | 0.1372 | 0.0016 | -0.0288 | 0.0198 | with_spike better | no |
