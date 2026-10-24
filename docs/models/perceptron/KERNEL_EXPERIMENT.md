# RBF Kernel Perceptron — การทดลองเสริมหลังส่งงาน

การทดลองนี้แยกจาก Perceptron แบบเส้นตรงที่เป็นหนึ่งใน 19 โมเดลของงานเดิม โมเดลเดิม, ข้อมูลหลัก, `selection_lock.json`, `finalize_receipt.json` และผล Test เดิมไม่ถูกแก้ ค่า hyperparameter ของโมเดลเสริมเลือกจาก Validation ก่อนประเมิน Test ครั้งนี้ ผล Test นี้เป็น **การติดตามผลหลังส่งงาน** ไม่ใช่ผล Test ที่ล็อกไว้ล่วงหน้าของ 19 โมเดล จึงไม่ใช้จัดอันดับใหม่ในรายงานหลัก

## วิธีทดลอง

- ใช้ 8 ฟีเจอร์และ label เดิม โดย fit `Standardizer` จาก Train เท่านั้น เช่นเดียวกับ trainer เดิม
- โมเดลจาก scratch ใช้ RBF kernel และอัปเดตน้ำหนักเมื่อทายผิด ไม่ใช่ SLP/MLP และไม่ได้แทน Perceptron เดิม
- ลอง `gamma` = 0.5, 0.75, 1.0, 1.5, 2.0 ด้วยเพดาน 1,000 epochs ทุก candidate หยุดเมื่อครบหนึ่ง epoch โดยไม่มีการอัปเดต เลือกค่า AP สูงสุดบน Validation แยกตาม variant
- โมเดลที่เลือกทาย Train ถูกทุกแถว; `with_spike` มี 1,733 แถว (positive 433), `non_spike` มี 1,520 แถว (positive 280) และทั้งสองแบบโหลดโมเดลที่บันทึกไว้กลับมาแล้วได้คำทำนาย Validation ตรงเดิม
- หลังล็อกตัวเลือกข้างต้น โหลดโมเดลจาก artifact เพื่อประเมิน Test 373 แถว (positive 73) **โดยไม่ refit และไม่ปรับ gamma จาก Test**

| variant | gamma | epoch ที่หยุด | updates รอบสุดท้าย | Train ถูก | support vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `with_spike` | 1.0 | 139 | 0 | 1,733/1,733 | 929 |
| `non_spike` | 0.75 | 142 | 0 | 1,520/1,520 | 722 |

## ผลของโมเดลที่เลือก

Confusion matrix เขียนเป็น `[[TN, FP], [FN, TP]]`; AP คือ average precision ค่าคะแนนเป็น decision score ไม่ใช่ความน่าจะเป็น

| variant | split | AP | ROC-AUC | accuracy | precision | recall | specificity | F1 | confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | Validation | 0.1732 | 0.6835 | 0.8140 | 0.1930 | 0.3235 | 0.8635 | 0.2418 | `[[291,46],[23,11]]` |
| `with_spike` | Test | 0.4416 | 0.7826 | 0.7962 | 0.4831 | 0.5890 | 0.8467 | 0.5309 | `[[254,46],[30,43]]` |
| `non_spike` | Validation | 0.1460 | 0.6533 | 0.8571 | 0.0870 | 0.0588 | 0.9377 | 0.0702 | `[[316,21],[32,2]]` |
| `non_spike` | Test | 0.3728 | 0.7469 | 0.7936 | 0.4000 | 0.1096 | 0.9600 | 0.1720 | `[[288,12],[65,8]]` |

Perceptron แบบเดิมได้ ROC-AUC 0.2993/0.3383 บน Validation และ 0.2749/0.2626 บน Test (`with_spike`/`non_spike`) Kernel Perceptron จัดอันดับได้ดีกว่าทั้งสอง variant แต่ `non_spike` ยังจับ positive บน Validation ได้เพียง 2 จาก 34 วัน และบน Test 8 จาก 73 วัน การลู่เข้าบน Train ไม่ใช่หลักฐานว่าพยากรณ์วันใหม่ได้ดีทุกกรณี นอกจากนี้ Validation มี positive 34/371 วัน ส่วน Test มี 73/373 วัน จึงเทียบค่า AP ข้าม split โดยตรงไม่ได้

## หลักฐานและการทำซ้ำ

- ฝึกและเลือกบน Validation: `python -m src.ml.kernel_perceptron_experiment` → [`summary.json`](../../../outputs/experiments/kernel_perceptron/summary.json)
- ประเมิน Test จากโมเดลที่บันทึกไว้: `python -m src.ml.kernel_perceptron_test --allow-test` → [`test_receipt.json`](../../../outputs/experiments/kernel_perceptron/test_receipt.json) คำสั่งนี้ปฏิเสธการรันซ้ำเมื่อมี receipt แล้ว
- ผลราย variant อยู่ใน `outputs/experiments/kernel_perceptron/<variant>/classification/kernel_perceptron/` ได้แก่ `config.json`, `history.json`, `validation_metrics.json`, `validation_predictions.csv`, `test_metrics.json`, `test_predictions.csv` และไฟล์ model/preprocessor สำหรับโหลดซ้ำ
- ตรวจ AP, ROC-AUC และ confusion matrix จาก `test_predictions.csv` อีกครั้งด้วย scikit-learn แล้วตรงกับ `test_metrics.json` ทั้งสอง variant; hash ของไฟล์ Test, config และโมเดลบันทึกไว้ใน receipt

ผลนี้เป็นการทดลองเสริมที่เปิด Test **หลัง** การส่งงานและหลังเห็นผล Test ของโมเดลเดิม จึงไม่อ้างว่าเป็นการเปรียบเทียบแบบ blind หรือมีนัยสำคัญทางสถิติ

## กราฟและอันดับใน notebook

Notebook [การจำแนก](../../../notebooks/04_classification.ipynb) แสดงอันดับ AP บน Validation รวม Kernel กับ 13 classifiers และ 2 baselines เดิม พร้อมกราฟการอัปเดตบน Train, ROC, precision–recall และ confusion matrix; [การเปรียบเทียบผล](../../../notebooks/05_results_comparison.ipynb) แสดงอันดับ AP บน Test และกราฟ ROC, precision–recall, confusion matrix และ metric เทียบ Validation/Test โดยอ่านจาก artifact ที่บันทึกไว้

เมื่อนับทั้ง baseline อันดับ AP ของ Kernel คือ **13/16** (`with_spike`) และ **14/16** (`non_spike`) ทั้งบน Validation และ Test ตารางใน notebook ระบุ `study=post-submission` เพื่อแยกจาก leaderboard ที่ล็อกไว้ของ 19 โมเดลเดิม จึงไม่ใช้ตารางนี้เปลี่ยน headline เดิม

## การตรวจเพิ่มเติมให้เทียบกับโมเดลอื่นได้

รัน `python -m src.ml.kernel_perceptron_analysis` เพื่อสร้าง [รายงานเสริม](../../../outputs/experiments/kernel_perceptron/report/) จากโมเดลและคำทำนายที่บันทึกไว้ โดยไม่แก้ `outputs/modeling/` ไม่ refit บน Test และไม่เปิดไฟล์ข้อมูล Test ต้นฉบับอีกครั้ง ผลที่สร้างมี leaderboard เสริมพร้อมช่วงความเชื่อมั่น, `walk_forward.csv`, `feature_importance.csv`, `segmented_metrics.csv`, ผล paired bootstrap แยก split/variant และ `reference_parity.json` Notebook ทั้งสองแสดงตารางและกราฟจากไฟล์เหล่านี้แล้ว

- **Walk-forward:** expanding-window 4 fold ภายใน Train; เว้น 5 วันก่อนแต่ละ block, fit scaler จากประวัติของ fold เท่านั้น และใช้ `gamma` ที่เลือกไว้ ทุก fold ลู่เข้า ผล pooled AP = 0.5127 [0.4049, 0.6249] (`with_spike`) และ 0.3987 [0.2899, 0.5231] (`non_spike`) ช่วง 95% มาจาก moving-block bootstrap 2,000 รอบ ตัวเลข pooled นี้วัดบนคนละช่วงกับ Validation/Test และมีสัดส่วน positive ต่างกัน จึงไม่ใช้จัดอันดับข้าม split
- **Feature importance:** สลับฟีเจอร์บน Validation เป็นก้อน 20 วัน 30 ครั้ง แล้ววัด AP ที่ลดลง เช่น `with_spike` มี `rsi_14` +0.0317 และ `historical_volatility_20d` +0.0282; `non_spike` มี `historical_volatility_20d` +0.0168 ค่าเป็นการพึ่งพาฟีเจอร์เชิงพยากรณ์ ไม่ใช่เหตุและผล
- **ช่วงความเชื่อมั่นและผลต่างแบบ paired:** ใช้ moving-block bootstrap ก้อน 20 วัน 2,000 รอบบนคำทำนายที่จับคู่วันตรงกัน ผล Test AP ของ Kernel คือ 0.4416 [0.2265, 0.5974] และ 0.3728 [0.1833, 0.5398] ผลต่าง AP ระหว่าง Kernel กับ Perceptron เส้นตรงบน Test = +0.3042 [0.1030, 0.4360] / +0.2400 [0.0784, 0.3730] แต่เมื่อเทียบกับโมเดล headline ที่ล็อกไว้ ช่วงผลต่างยังคร่อมศูนย์ทั้งสอง variant จึงยังสรุปว่า Kernel ดีกว่า headline ไม่ได้
- **Test แยกช่วง:** ช่วง `spike_affected` มี 54 แถว (positive 38) และ `non_spike` มี 319 แถว (positive 35) Kernel `with_spike` มี ROC-AUC 0.7138 / 0.7241 และ recall 0.6053 / 0.5714 ตามลำดับ; Kernel `non_spike` มี ROC-AUC 0.7862 / 0.6910 แต่ recall เพียง 0.1053 / 0.1143 AP ของสองช่วงเทียบตรง ๆ ไม่ได้ เพราะอัตรา positive ต่างกันมาก
- **การตรวจเทียบไลบรารี:** คำนวณ RBF kernel บน Validation ด้วย `sklearn.metrics.pairwise.rbf_kernel` แล้วประกอบคะแนนจาก support vectors ที่บันทึกไว้ ได้ label ตรงกันทุกแถว; ความต่างของคะแนนสูงสุด < 6×10⁻¹⁴ การฝึกอิสระที่คำนวณคะแนนใหม่จาก dual weights ทุกครั้งก็ลู่เข้าและทาย Train ถูกทุกแถว `non_spike` มีประวัติการอัปเดตและ epoch 142 ตรงกัน ส่วน `with_spike` หยุดที่ epoch 144 แทน 139, Validation AP 0.1744 แทน 0.1732 เพราะค่าคะแนนใกล้ศูนย์ทำให้ลำดับการบวกทศนิยมเปลี่ยนการตัดสินใจอัปเดต (พบตัวอย่างความต่างระดับ 4.8×10⁻¹⁷) จึง **ไม่อ้าง parity ของขั้นตอนฝึกแบบตรงทุกอัปเดต** รายละเอียดอยู่ใน `training_reference.json`; scikit-learn ไม่มีคลาส online Kernel Perceptron ที่ตรงกัน

ผล Test ของ Kernel เป็นการวิเคราะห์ย้อนหลังหลังเห็นผลชุดเดิม การเพิ่มช่วงความเชื่อมั่นและตารางเปรียบเทียบช่วยบอกความไม่แน่นอน แต่ไม่ทำให้ผลนี้กลายเป็นการประเมินแบบ blind ได้
