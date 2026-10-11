### Test leaderboard — regression — non_spike

| model | kind | headline | validation_rmse | rmse_ci | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| polynomial | model | no | 0.0661 | 0.1142 [0.0720, 0.1641] | 0.3155 [-0.0360, 0.4224] | yes | yes |
| multiple_linear | model | no | 0.0668 | 0.1180 [0.0705, 0.1732] | 0.2684 [-0.1829, 0.3623] | yes | yes |
| elastic_net | model | yes | 0.0643 | 0.1230 [0.0688, 0.1859] | 0.2059 [0.0554, 0.2619] | yes | yes |
| simple_linear | model | no | 0.0665 | 0.1372 [0.0733, 0.2087] | 0.0114 [-0.1001, 0.0488] | yes | yes |
| baseline_train_mean | baseline | no | n/a | 0.1400 [0.0745, 0.2132] | -0.0298 [-0.1430, -0.0001] | yes | n/a |
| baseline_persistence | baseline | no | n/a | 0.1502 [0.0919, 0.2222] | -0.1855 [-1.2071, 0.1280] | no | n/a |

> ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้
