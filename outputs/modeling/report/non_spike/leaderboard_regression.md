### Validation leaderboard — regression — non_spike

| model | kind | rmse_ci | mae | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net | model | 0.0643 [0.0572, 0.0741] | 0.0510 | 0.0906 [-0.0841, 0.2010] | yes | yes |
| polynomial | model | 0.0661 [0.0596, 0.0755] | 0.0509 | 0.0389 [-0.1903, 0.1745] | yes | yes |
| simple_linear | model | 0.0665 [0.0582, 0.0783] | 0.0527 | 0.0265 [-0.1099, 0.0616] | yes | yes |
| multiple_linear | model | 0.0668 [0.0597, 0.0761] | 0.0530 | 0.0165 [-0.2942, 0.2361] | yes | yes |
| baseline_train_mean | baseline | 0.0676 [0.0585, 0.0805] | 0.0535 | -0.0076 [-0.1436, -0.0000] | yes | n/a |
| baseline_persistence | baseline | 0.0826 [0.0750, 0.0901] | 0.0673 | -0.5032 [-1.0576, -0.0677] | no | n/a |

> ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% (moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น
> 4 จาก 4 โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้
