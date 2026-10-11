# QQQ Volatility Forecasting

โปรเจกต์วิชา Machine Learning: พยากรณ์ **ความผันผวนของกองทุน QQQ ใน 5 วันทำการข้างหน้า** จากข้อมูลราคารายวัน 10 ปี (2016–2026) ในสองรูปแบบ

- **Regression** ทำนายค่าความผันผวน (`target_volatility_5d`)
- **Classification** ทำนายว่าจะเป็นช่วง "ผันผวนสูง" หรือไม่ (`target_high_volatility`)

คำถามวิจัยของโปรเจกต์คือ **การตัดช่วงที่ราคากระชากแรง (spike) ออกจากข้อมูลเทรน ช่วยหรือทำร้ายโมเดล** เราจึงเทรนโมเดลทุกตัวสองแบบ คือแบบเก็บ spike ไว้ (`with_spike`) และแบบตัดออก (`non_spike`) แล้ววัดบนข้อมูลชุดเดียวกัน

โมเดลทั้ง 19 ตัว **เขียนเองด้วย NumPy** ส่วน scikit-learn และ XGBoost ใช้เพียงเพื่อยืนยันว่าเขียนถูก

> ไม่เคยเรียน ML มาก่อน เริ่มที่ [`docs/guides/PROJECT_GUIDE_TH.md`](docs/guides/PROJECT_GUIDE_TH.md) ซึ่งอธิบายทุกอย่างจากศูนย์พร้อมอภิธานศัพท์

## ผลลัพธ์โดยย่อ

ผลบน Test ถูกอ่าน **ครั้งเดียว** หลังล็อกโมเดลและการตั้งค่าทั้งหมดแล้ว (ไม่มีการเทรนใหม่หรือเลือกใหม่หลังเห็นผล) ตารางนี้สร้างจากไฟล์ผลโดยตรง

<!-- BEGIN GENERATED: outputs/modeling/report/tables/test_headline_summary.md -->
### Locked headline models: Validation score, one-shot Test score and the best baseline

| task | variant | headline | metric | validation | test | test_low | test_high | rank_on_test | best_baseline | baseline_test | beats_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | with_spike | mlp | average_precision | 0.2668 | 0.5991 | 0.2605 | 0.7603 | 1 of 13 | baseline_persistence | 0.5365 | yes |
| classification | non_spike | random_forest | average_precision | 0.3010 | 0.5240 | 0.2161 | 0.6796 | 10 of 13 | baseline_persistence | 0.5365 | no |
| regression | with_spike | elastic_net | rmse | 0.0668 | 0.1174 | 0.0709 | 0.1746 | 2 of 4 | baseline_train_mean | 0.1380 | yes |
| regression | non_spike | elastic_net | rmse | 0.0643 | 0.1230 | 0.0688 | 0.1859 | 3 of 4 | baseline_train_mean | 0.1400 | yes |
<!-- END GENERATED -->

วิธีอ่าน: `validation` คือคะแนนตอนเลือกโมเดล, `test` คือคะแนนบนข้อมูลที่โมเดลไม่เคยเห็น พร้อมช่วงความเชื่อมั่น 95% (`test_low` ถึง `test_high`), `best_baseline` คือกฎง่าย ๆ ที่ไม่ต้องเทรน สำหรับ `average_precision` ค่ามากดีกว่า ส่วน `rmse` ค่าน้อยดีกว่า

**ข้อค้นพบหลัก**

1. **โมเดลทำนายได้ดีกว่าการเดาสุ่มชัดเจน แต่เหนือกว่ากฎง่าย ๆ เพียงเล็กน้อย** กฎ "5 วันข้างหน้าจะผันผวนเท่ากับ 5 วันที่ผ่านมา" ทำได้ใกล้เคียงโมเดลส่วนใหญ่
2. **ยังสรุปไม่ได้ว่าการตัด spike ช่วยหรือทำร้ายโมเดล** ผลต่างเล็กกว่าความไม่แน่นอนของการวัด ทั้งบน Validation และ Test และทิศทางของสองชุดไม่ตรงกัน
3. **โมเดลเก่งในช่วงตลาดปกติ แต่ไม่เก่งในช่วงปั่นป่วน** ภายในช่วงรอบวัน spike โมเดลแยกไม่ได้ว่าวันไหนจะผันผวนสูง (ROC-AUC ใกล้ 0.5)
4. **ผู้นำบน Validation ไม่จำเป็นต้องเป็นผู้นำบน Test** Validation มีวันผันผวนสูงเพียง 34 วัน จึงตัดสินผู้ชนะไม่ได้

อ่านผลเต็มได้ที่ [`notebooks/05_results_comparison.ipynb`](notebooks/05_results_comparison.ipynb) และรายงานใน [`docs/report/`](docs/report/)

## เริ่มใช้งาน

ต้องมี Windows, Python 3.14 และ Git รายละเอียดและการแก้ปัญหาอยู่ใน [`docs/guides/RUNBOOK.md`](docs/guides/RUNBOOK.md)

```bat
rem 1. สร้าง environment และติดตั้ง library
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

rem 2. สร้างข้อมูลจาก snapshot ที่มากับ repo (ไม่ต้องดาวน์โหลดอะไร)
.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp\phase1-reproduction" --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_spike_analysis --reproduced-root "tmp\phase1-reproduction" --overwrite-generated

rem 3. ตรวจว่าทุกอย่างทำงาน (ราว 10 นาที)
.\.venv\Scripts\python.exe -m pytest -q
```

ผลของโมเดลที่เทรนแล้วทั้ง 38 ชุดมากับ repo ใต้ `outputs/modeling/` จึงเปิด notebook 03–05 ดูผลได้ทันทีโดยไม่ต้องเทรนใหม่

## ดูผลและทำซ้ำ

| ต้องการ | คำสั่ง | ใช้เวลา |
| --- | --- | --- |
| ดูผลพร้อมคำอธิบาย | เปิด `notebooks/03_regression.ipynb`, `04_classification.ipynb`, `05_results_comparison.ipynb` | — |
| สร้างตารางจัดอันดับและรูปใหม่ | `.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike` (และ `non_spike`) | ~15 วินาที |
| เทียบ `with_spike` กับ `non_spike` | `.\.venv\Scripts\python.exe -m src.ml.run compare` | ~15 วินาที |
| feature importance และ walk-forward | `.\.venv\Scripts\python.exe -m src.ml.run analyze --variant with_spike` (และ `non_spike`) | ~4 นาที |
| เทรนโมเดลทั้งหมดใหม่ | `.\.venv\Scripts\python.exe -m src.ml.run train --all --variant with_spike` (และ `non_spike`) | หลายนาที |
| อัปเดตตารางในเอกสารให้ตรงกับไฟล์ผล | `.\.venv\Scripts\python.exe -m src.ml.docs_sync` | ~30 วินาที |

> **ห้ามรัน `finalize --allow-test` ซ้ำ** Test ถูกอ่านไปแล้วหนึ่งครั้งเมื่อ 2026-10-11 หลักฐานอยู่ที่ `outputs/modeling/finalize_receipt.json` คำสั่งจะปฏิเสธการรันซ้ำเอง และการเทรนโมเดลใหม่จะทำให้ไฟล์ไม่ตรงกับที่ล็อกไว้ใน `outputs/modeling/selection_lock.json`

## โครงสร้าง repo

| ที่อยู่ | เนื้อหา |
| --- | --- |
| `data/raw/qqq_daily.csv` | ข้อมูลดิบ 2,512 วัน (snapshot ที่ตรึงไว้ พร้อม SHA-256 ใน `data/manifests/`) |
| `src/*.py` | data pipeline: ทำความสะอาด, สร้าง feature และ target, แบ่งข้อมูล, ตรวจหา spike |
| `src/ml/` | โมเดล 19 ตัว (`regression/`, `classification/`, `clustering/`) และแกนกลาง (`core/`) |
| `src/ml/run.py` | คำสั่ง `train`, `report`, `compare`, `analyze`, `finalize` |
| `configs/modeling.json` | รายการโมเดลและช่วงค่าที่ลอง (grid) |
| `outputs/modeling/<variant>/<task>/<model>/` | ผลของแต่ละโมเดล: น้ำหนัก, การตั้งค่า, คะแนน, คำทำนาย |
| `outputs/modeling/report/`, `outputs/modeling/final/` | ตารางสรุปบน Validation และบน Test |
| `outputs/figures/` | รูปทั้งหมด |
| `notebooks/` | 01–02 ข้อมูลและ feature · 03–05 ผลของโมเดล |
| `tests/` | test กว่า 1,200 ข้อ รวม test ที่กันข้อมูลรั่วและกันการอ่าน Test ก่อนเวลา |
| `docs/` | เอกสารทั้งหมด ดูสารบัญที่ [`docs/README.md`](docs/README.md) |

## โมเดล 19 ตัว

| งาน | โมเดล |
| --- | --- |
| Regression (4) | `simple_linear`, `multiple_linear`, `polynomial`, `elastic_net` ★ |
| Classification (13) | `logistic`, `gaussian_nb`, `knn`, `perceptron`, `slp`, `mlp`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `adaboost` ★, `stacking`, `svm` |
| Clustering (2) | `kmeans`, `agglomerative` |

★ = โมเดลนอกเนื้อหาที่สอน ซึ่งโจทย์กำหนดให้อธิบายกลไกภายในอย่างละเอียด สเปกของทุกโมเดลและผลจริงอยู่ใน [`docs/plan/MODEL_CARDS.md`](docs/plan/MODEL_CARDS.md) การตั้งค่าที่ถูกเลือกอยู่ใน `outputs/modeling/report/tables/model_inventory.md`

## ข้อจำกัดที่ต้องรู้ก่อนอ่านตัวเลข

- **ตัวอย่างไม่อิสระต่อกัน** target ของวันติดกันใช้ข้อมูลช่วงเดียวกัน 4 ใน 5 วัน ช่วงความเชื่อมั่นจึงคำนวณด้วย block bootstrap และกว้างกว่าที่จำนวนแถวบอก
- **ตัวเลขบน Validation เป็น selection score** เพราะการตั้งค่าของทุกโมเดลถูกเลือกบนข้อมูลชุดนั้น ตัวเลขที่ไม่ลำเอียงคือผลบน Test
- **การเทียบ `with_spike` กับ `non_spike` ถูก confound** การตัดแถวเปลี่ยนจำนวนแถว สัดส่วนคลาส และช่วงตลาดในข้อมูลเทรนพร้อมกัน
- **AP เทียบข้ามชุดข้อมูลไม่ได้** เพราะขึ้นกับสัดส่วนวันผันผวนสูง (Validation 9.2%, Test 19.6%)
- **สินทรัพย์เดียว ช่วงเวลาเดียว** ผลอาจไม่ใช้กับสินทรัพย์หรือช่วงเวลาอื่น

## เอกสาร

- สารบัญเอกสารทั้งหมด: [`docs/README.md`](docs/README.md)
- กฎการเขียนโค้ดของโปรเจกต์: [`AGENTS.md`](AGENTS.md)
- ผลตรวจความถูกต้องของข้อมูล: [`docs/audit/DATA_PIPELINE_AUDIT.md`](docs/audit/DATA_PIPELINE_AUDIT.md)
- โจทย์ของวิชา: [`_69-1-01076641_Project-description.pdf`](_69-1-01076641_Project-description.pdf)
