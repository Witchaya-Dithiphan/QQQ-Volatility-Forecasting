### Paired comparison: non_spike − with_spike (Validation)

| model | task | metric | with_spike | non_spike | delta | delta_95 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net | regression | rmse | 0.0668 | 0.0643 | -0.0026 | [-0.0053, +0.0019] | inconclusive |
| multiple_linear | regression | rmse | 0.0689 | 0.0668 | -0.0020 | [-0.0036, -0.0000] | non_spike better |
| polynomial | regression | rmse | 0.0691 | 0.0661 | -0.0030 | [-0.0073, +0.0016] | inconclusive |
| simple_linear | regression | rmse | 0.0690 | 0.0665 | -0.0026 | [-0.0066, +0.0026] | inconclusive |
| adaboost | classification | average_precision | 0.2363 | 0.2407 | 0.0044 | [-0.0307, +0.0627] | inconclusive |
| decision_tree | classification | average_precision | 0.1732 | 0.1887 | 0.0155 | [-0.0380, +0.0970] | inconclusive |
| gaussian_nb | classification | average_precision | 0.2218 | 0.2292 | 0.0074 | [-0.0007, +0.0187] | inconclusive |
| gradient_boosting | classification | average_precision | 0.2531 | 0.2525 | -0.0006 | [-0.0445, +0.0223] | inconclusive |
| knn | classification | average_precision | 0.2037 | 0.2082 | 0.0045 | [-0.0441, +0.0601] | inconclusive |
| logistic | classification | average_precision | 0.2330 | 0.2513 | 0.0183 | [-0.0110, +0.0401] | inconclusive |
| mlp | classification | average_precision | 0.2668 | 0.2808 | 0.0139 | [-0.0429, +0.0772] | inconclusive |
| perceptron | classification | average_precision | 0.0648 | 0.0742 | 0.0094 | [-0.0012, +0.0575] | inconclusive |
| random_forest | classification | average_precision | 0.2548 | 0.3010 | 0.0462 | [-0.0168, +0.1155] | inconclusive |
| slp | classification | average_precision | 0.2427 | 0.2474 | 0.0047 | [-0.0477, +0.0441] | inconclusive |
| stacking | classification | average_precision | 0.2262 | 0.2311 | 0.0049 | [-0.0386, +0.0477] | inconclusive |
| svm | classification | average_precision | 0.2514 | 0.2501 | -0.0013 | [-0.0108, +0.0082] | inconclusive |
| xgboost | classification | average_precision | 0.2508 | 0.2652 | 0.0144 | [-0.0148, +0.0386] | inconclusive |
| agglomerative | clustering | silhouette | 0.3602 | 0.4642 | 0.1040 | n/a | no interval |
| kmeans | clustering | silhouette | 0.3666 | 0.3298 | -0.0367 | n/a | no interval |

> ผลต่างในตารางนี้คือผลของ **นโยบายตัดแถวทั้งชุด** ไม่ใช่ผลของ spike เพียงอย่างเดียว: การตัด 213 แถวทำให้ class prior ของ Train เปลี่ยนจาก 25.0% เป็น 18.4%, จำนวนแถวลดจาก 1,733 เป็น 1,520 และ regime ที่ผันผวนสูงหายไปจาก Train พร้อมกัน ทั้งสาม confound นี้แยกจากกันไม่ได้ด้วยการทดลองนี้ delta = non_spike − with_spike บน Validation 371 แถวชุดเดียวกัน ช่วง 95% จาก paired moving-block bootstrap (block 20 วัน) และเมื่อเทียบ 17 โมเดลพร้อมกัน ควรคาดว่าจะมีราว 1 แถวที่ช่วงไม่คร่อมศูนย์โดยบังเอิญ
