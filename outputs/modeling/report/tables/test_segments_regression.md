### Test metrics by diagnostic segment — regression

| model | variant | segment | rows | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net | non_spike | non_spike | 319 | 0.0692 | 0.0550 | 0.1752 |
| elastic_net | non_spike | spike_affected | 54 | 0.2759 | 0.1838 | -0.3351 |
| elastic_net | with_spike | non_spike | 319 | 0.0707 | 0.0564 | 0.1395 |
| elastic_net | with_spike | spike_affected | 54 | 0.2564 | 0.1722 | -0.1526 |
| multiple_linear | non_spike | non_spike | 319 | 0.0683 | 0.0525 | 0.1972 |
| multiple_linear | non_spike | spike_affected | 54 | 0.2620 | 0.1880 | -0.2042 |
| multiple_linear | with_spike | non_spike | 319 | 0.0694 | 0.0536 | 0.1714 |
| multiple_linear | with_spike | spike_affected | 54 | 0.2533 | 0.1819 | -0.1253 |
