### Test leaderboard — regression — with_spike

| model | kind | headline | validation_rmse | rmse_ci | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiple_linear | model | no | 0.0689 | 0.1158 [0.0711, 0.1680] | 0.2959 [-0.1971, 0.3897] | yes | yes |
| elastic_net | model | yes | 0.0668 | 0.1174 [0.0709, 0.1746] | 0.2758 [-0.0402, 0.3258] | yes | yes |
| simple_linear | model | no | 0.0690 | 0.1356 [0.0754, 0.1985] | 0.0347 [-0.7702, 0.2310] | no | yes |
| baseline_train_mean | baseline | no | n/a | 0.1380 [0.0806, 0.2060] | -0.0006 [-0.2998, -0.0000] | yes | n/a |
| polynomial | model | no | 0.0691 | 0.1401 [0.0754, 0.2037] | -0.0314 [-1.1539, 0.2401] | no | no |
| baseline_persistence | baseline | no | n/a | 0.1502 [0.0919, 0.2222] | -0.1855 [-1.2071, 0.1280] | no | n/a |

> ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้
