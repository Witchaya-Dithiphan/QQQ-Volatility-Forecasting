### Validation leaderboard — regression — with_spike

| model | kind | rmse_ci | mae | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net | model | 0.0668 [0.0610, 0.0739] | 0.0539 | 0.0169 [-0.2649, 0.2159] | yes | yes |
| multiple_linear | model | 0.0689 [0.0617, 0.0779] | 0.0542 | -0.0446 [-0.3994, 0.2085] | yes | yes |
| simple_linear | model | 0.0690 [0.0609, 0.0789] | 0.0557 | -0.0498 [-0.2829, 0.0997] | yes | yes |
| polynomial | model | 0.0691 [0.0606, 0.0797] | 0.0552 | -0.0507 [-0.2579, 0.0879] | yes | yes |
| baseline_train_mean | baseline | 0.0751 [0.0673, 0.0837] | 0.0623 | -0.2430 [-0.6975, -0.0225] | no | n/a |
| baseline_persistence | baseline | 0.0826 [0.0750, 0.0901] | 0.0673 | -0.5032 [-1.0576, -0.0677] | no | n/a |

> ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% (moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น
> 4 จาก 4 โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้
