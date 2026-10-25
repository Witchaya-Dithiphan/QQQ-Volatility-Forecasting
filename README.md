# QQQ Volatility Forecasting

โครงงานรายวิชา Machine Learning เรื่องการพยากรณ์ความผันผวนของกองทุน QQQ ในช่วง 5 วันทำการข้างหน้า จากข้อมูลราคารายวันย้อนหลัง 10 ปี (พ.ศ. 2559–2569 หรือ ค.ศ. 2016–2026) โจทย์ถูกตั้งไว้สองแบบ แบบแรกเป็น regression ซึ่งทำนายค่าความผันผวน (`target_volatility_5d`) แบบที่สองเป็น classification ซึ่งทำนายว่าช่วงข้างหน้าจะเป็นช่วงผันผวนสูงหรือไม่ (`target_high_volatility`)

คำถามหลักของงานคือ การตัดช่วงที่ราคากระชากแรง (spike) ออกจากข้อมูลฝึกช่วยให้โมเดลดีขึ้นหรือแย่ลง เราจึงฝึกโมเดลทุกตัวสองครั้ง ครั้งหนึ่งใช้ข้อมูลที่เก็บช่วง spike ไว้ (`with_spike`) อีกครั้งใช้ข้อมูลที่ตัดช่วงนั้นออก (`non_spike`) แล้ววัดผลบนข้อมูลทดสอบชุดเดียวกัน

โมเดลทั้ง 19 ตัวเขียนขึ้นเองด้วย NumPy ส่วน scikit-learn และ XGBoost ใช้เพียงเป็นตัวอ้างอิงเพื่อยืนยันว่าโมเดลที่เขียนเองให้ผลตรงกัน

ผู้จัดทำ: Chern, Phoom

ผู้ที่ยังไม่คุ้นกับ machine learning ควรเริ่มอ่านจาก [`docs/guides/PROJECT_GUIDE_TH.md`](docs/guides/PROJECT_GUIDE_TH.md) ซึ่งอธิบายงานทั้งหมดตั้งแต่พื้นฐานและมีอภิธานศัพท์ท้ายเล่ม ส่วนรายงานฉบับสมบูรณ์อยู่ที่ [`deliverables/report.pdf`](deliverables/report.pdf)

## ผลโดยสรุป

ข้อมูลชุด Test ถูกเปิดอ่านเพียงครั้งเดียว หลังจากล็อกโมเดลและการตั้งค่าทั้งหมดแล้ว ไม่มีการฝึกใหม่หรือเลือกโมเดลใหม่หลังเห็นผล ตารางด้านล่างสร้างจากไฟล์ผลโดยตรง

<!-- BEGIN GENERATED: outputs/modeling/report/tables/test_headline_summary.md -->
### Locked headline models: Validation score, one-shot Test score and the best baseline

| task | variant | headline | metric | validation | test | test_low | test_high | rank_on_test | best_baseline | baseline_test | beats_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | with_spike | mlp | average_precision | 0.2668 | 0.5991 | 0.2605 | 0.7603 | 1 of 13 | baseline_persistence | 0.5365 | yes |
| classification | non_spike | random_forest | average_precision | 0.3010 | 0.5240 | 0.2161 | 0.6796 | 10 of 13 | baseline_persistence | 0.5365 | no |
| regression | with_spike | elastic_net | rmse | 0.0668 | 0.1174 | 0.0709 | 0.1746 | 2 of 4 | baseline_train_mean | 0.1380 | yes |
| regression | non_spike | elastic_net | rmse | 0.0643 | 0.1230 | 0.0688 | 0.1859 | 3 of 4 | baseline_train_mean | 0.1400 | yes |
<!-- END GENERATED -->

คอลัมน์ `validation` คือคะแนนที่ใช้เลือกโมเดล คอลัมน์ `test` คือคะแนนบนข้อมูลที่โมเดลไม่เคยเห็น โดยมีช่วงความเชื่อมั่น 95% อยู่ใน `test_low` ถึง `test_high` และ `best_baseline` คือกฎง่าย ๆ ที่ไม่ต้องฝึก สำหรับ `average_precision` ค่ายิ่งมากยิ่งดี ส่วน `rmse` ค่ายิ่งน้อยยิ่งดี

ข้อค้นพบหลักมีสี่ข้อ

1. โมเดลทำนายได้ดีกว่าการเดาสุ่มอย่างชัดเจน แต่ดีกว่ากฎง่าย ๆ เพียงเล็กน้อย กฎที่ถือว่า 5 วันข้างหน้าจะผันผวนเท่ากับ 5 วันที่ผ่านมาให้ผลใกล้เคียงกับโมเดลส่วนใหญ่
2. ยังสรุปไม่ได้ว่าการตัด spike ช่วยหรือทำร้ายโมเดล เพราะผลต่างเล็กกว่าความไม่แน่นอนของการวัดทั้งบน Validation และ Test และทิศทางของผลในสองชุดข้อมูลไม่ตรงกัน
3. โมเดลทำงานได้ดีในช่วงตลาดปกติ แต่ในช่วงรอบวัน spike โมเดลแยกไม่ออกว่าวันใดจะผันผวนสูง (ROC-AUC ใกล้ 0.5)
4. โมเดลที่ชนะบน Validation ไม่จำเป็นต้องชนะบน Test เพราะ Validation มีวันผันผวนสูงเพียง 34 วัน ซึ่งน้อยเกินกว่าจะตัดสินผู้ชนะได้

ผลฉบับเต็มอยู่ใน [`notebooks/05_results_comparison.ipynb`](notebooks/05_results_comparison.ipynb) และในรายงาน ส่วนเกณฑ์ที่ใช้นิยาม spike (Q3 + 3 × IQR ของข้อมูลฝึก) และเหตุผลที่เลือกเกณฑ์นี้อยู่ใน [`notebooks/06_spike_analysis.ipynb`](notebooks/06_spike_analysis.ipynb)

## การทดลองเพิ่มเติมหลังส่งงาน

หลังส่งงานและนำเสนอแล้ว เราทดลองเพิ่มอีกสามเรื่องตามคำถามและข้อเสนอแนะที่ได้รับ ทุกการทดลองแยกจากผลหลักของ 19 โมเดล ไม่แก้ข้อมูลหรือผลที่ล็อกไว้ และรันทั้งแบบ `with_spike` และ `non_spike` เนื่องจากข้อมูลชุด Test เคยถูกเปิดไปแล้ว ผล Test ของการทดลองเหล่านี้จึงเป็นผลประเมินภายหลัง (post-hoc) ไม่ใช่ผลที่ล็อกไว้ล่วงหน้า

| การทดลอง | สิ่งที่ทำ | ผลโดยย่อ | บันทึก |
| --- | --- | --- | --- |
| RBF Kernel Perceptron | ใช้ kernel แก้ปัญหาที่ Perceptron แบบเส้นตรงไม่ลู่เข้า เพราะข้อมูลสองคลาสแบ่งด้วยเส้นตรงไม่ได้ | ลู่เข้าและทาย Train ถูกทุกแถว แต่อยู่อันดับ 13 และ 14 จาก 16 ทั้งบน Validation และ Test (AP บน Test 0.4416 และ 0.3728) | [`KERNEL_EXPERIMENT.md`](docs/models/perceptron/KERNEL_EXPERIMENT.md) |
| Transfer learning ด้วย Transformer และ GRU | pretrain แบบ self-supervised บน ETF 10 ตัว ต่อด้วย domain-adaptive pretraining บน QQQ แล้ว fine-tune กับ label ของ QQQ ทั้งหมดเขียนด้วย NumPy รวมถึง autograd | GRU ทำได้ใกล้เคียง MLP (AP บน Test 0.580 เทียบกับ 0.599) ส่วน Transformer แย่กว่า baseline และ pretraining ช่วยได้เฉพาะ Transformer | [`EXPERIMENT.md`](docs/models/transfer_learning/EXPERIMENT.md) |
| Log-HAR และ pooled-prior transfer | โมเดล HAR ในสเกล log สำหรับงาน regression โดยดึงค่าสัมประสิทธิ์เข้าหาค่าที่ประมาณจาก ETF 10 ตัว | บน Validation ยังไม่ดีกว่า Elastic Net บน Test แบบที่ใช้ prior ได้ RMSE 0.1090 และ 0.1091 เทียบกับ 0.1174 และ 0.1230 ของ Elastic Net แต่ช่วงความเชื่อมั่นของผลต่างยังคร่อมศูนย์ และ prior จากตลาดอื่นแทบไม่ช่วยเมื่อเทียบกับแบบที่ใช้ข้อมูล QQQ อย่างเดียว | [`EXPERIMENT.md`](docs/models/log_har/EXPERIMENT.md) |

รายละเอียดของทั้งสามการทดลองอยู่ในภาคผนวก ง จ และ ฉ ของรายงาน และใน notebook 03, 04 และ 05

## การติดตั้งและรัน

เครื่องที่ใช้ต้องเป็น Windows และมี Python 3.14 กับ Git คำอธิบายอย่างละเอียดและวิธีแก้ปัญหาที่พบบ่อยอยู่ใน [`docs/guides/RUNBOOK.md`](docs/guides/RUNBOOK.md)

```bat
rem 1. สร้าง environment และติดตั้ง library
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

rem 2. สร้างข้อมูลจาก snapshot ที่มากับ repo (ไม่ต้องดาวน์โหลดเพิ่ม)
.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp\phase1-reproduction" --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_spike_analysis --reproduced-root "tmp\phase1-reproduction" --overwrite-generated

rem 3. ตรวจว่าทุกส่วนทำงานถูกต้อง (ใช้เวลาราว 10 นาที)
.\.venv\Scripts\python.exe -m pytest -q
```

ผลของโมเดลที่ฝึกแล้วทั้ง 38 ชุดอยู่ใน repo ใต้ `outputs/modeling/` จึงเปิด notebook 03 ถึง 06 ดูผลได้ทันทีโดยไม่ต้องฝึกใหม่

## การดูผลและการทำซ้ำ

| ต้องการ | คำสั่ง | เวลาโดยประมาณ |
| --- | --- | --- |
| ดูผลพร้อมคำอธิบาย | เปิด `notebooks/03_regression.ipynb`, `04_classification.ipynb` และ `05_results_comparison.ipynb` | – |
| ดูเกณฑ์ spike และเหตุผลที่เลือก | เปิด `notebooks/06_spike_analysis.ipynb` | – |
| สร้างตารางจัดอันดับและรูปใหม่ | `.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike` (และ `non_spike`) | 15 วินาที |
| เทียบ `with_spike` กับ `non_spike` | `.\.venv\Scripts\python.exe -m src.ml.run compare` | 15 วินาที |
| feature importance และ walk-forward | `.\.venv\Scripts\python.exe -m src.ml.run analyze --variant with_spike` (และ `non_spike`) | 4 นาที |
| ฝึกโมเดลทั้งหมดใหม่ | `.\.venv\Scripts\python.exe -m src.ml.run train --all --variant with_spike` (และ `non_spike`) | หลายนาที |
| ปรับตารางในเอกสารให้ตรงกับไฟล์ผล | `.\.venv\Scripts\python.exe -m src.ml.docs_sync` | 30 วินาที |

ห้ามรัน `finalize --allow-test` ซ้ำ ข้อมูลชุด Test ถูกอ่านไปแล้วหนึ่งครั้งเมื่อวันที่ 11 ตุลาคม 2026 หลักฐานอยู่ที่ `outputs/modeling/finalize_receipt.json` คำสั่งนี้จะปฏิเสธการรันซ้ำเอง และถ้าฝึกโมเดลใหม่ ไฟล์ผลจะไม่ตรงกับค่าที่ล็อกไว้ใน `outputs/modeling/selection_lock.json` การทดลองเพิ่มเติมทั้งสามก็มีใบรับรองการเปิด Test ของตัวเอง (`test_receipt.json`) และปฏิเสธการรันซ้ำเช่นกัน

## โครงสร้างของ repo

| ตำแหน่ง | เนื้อหา |
| --- | --- |
| `data/raw/qqq_daily.csv` | ข้อมูลดิบ 2,512 วัน เป็น snapshot ที่ตรึงไว้พร้อมค่า SHA-256 ใน `data/manifests/` ([ดาวน์โหลดจาก Google Drive](https://drive.google.com/file/d/1x4PBm9WrHYX8uxSn_pVoLY7_QOHrTWCR/view?usp=sharing)) |
| `src/*.py` | data pipeline: ทำความสะอาดข้อมูล สร้าง feature และ target แบ่งข้อมูล และตรวจหา spike |
| `src/ml/` | โมเดล 19 ตัว (`regression/`, `classification/`, `clustering/`) แกนกลางที่ใช้ร่วมกัน (`core/`) และการทดลองเพิ่มเติม (`transfer_learning/`, `log_har/`) |
| `src/ml/run.py` | คำสั่ง `train`, `report`, `compare`, `analyze` และ `finalize` |
| `configs/modeling.json` | รายชื่อโมเดลและช่วงค่าของ hyperparameter ที่ลอง |
| `outputs/modeling/<variant>/<task>/<model>/` | ผลของแต่ละโมเดล ได้แก่ น้ำหนัก การตั้งค่า คะแนน และค่าทำนาย |
| `outputs/modeling/report/`, `outputs/modeling/final/` | ตารางสรุปบน Validation และบน Test |
| `outputs/experiments/` | ผลของการทดลองเพิ่มเติมทั้งสาม รวมถึงข้อมูลราคาของ ETF 10 ตัวที่ใช้ pretrain (`transfer_learning/sources/`) |
| `outputs/figures/` | รูปทั้งหมด |
| `notebooks/` | 01–02 ข้อมูลและ feature, 03–05 ผลของโมเดล, 06 เกณฑ์ spike และเหตุผลที่เลือก |
| `tests/` | ชุดทดสอบอัตโนมัติ รวมถึงการทดสอบที่กันข้อมูลรั่วและกันการอ่าน Test ก่อนเวลา |
| `deliverables/` | รายงาน (`report.docx`, `report.pdf`) และสไลด์นำเสนอ (`slides.pptx`) |
| `docs/` | เอกสารประกอบทั้งหมด ดูสารบัญได้ที่ [`docs/README.md`](docs/README.md) |

## โมเดลทั้ง 19 ตัว

| งาน | โมเดล |
| --- | --- |
| Regression (4) | `simple_linear`, `multiple_linear`, `polynomial`, `elastic_net` ★ |
| Classification (13) | `logistic`, `gaussian_nb`, `knn`, `perceptron`, `slp`, `mlp`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `adaboost` ★, `stacking`, `svm` |
| Clustering (2) | `kmeans`, `agglomerative` |

★ คือโมเดลนอกเนื้อหาที่สอน ซึ่งโจทย์กำหนดให้อธิบายกลไกภายในอย่างละเอียด สูตร ช่วงค่าที่ลอง และผลจริงของทุกโมเดลอยู่ใน [`docs/design/MODEL_CARDS.md`](docs/design/MODEL_CARDS.md) ส่วนการตั้งค่าที่ถูกเลือกอยู่ใน `outputs/modeling/report/tables/model_inventory.md`

## ข้อจำกัดที่ควรทราบก่อนอ่านตัวเลข

- ตัวอย่างแต่ละแถวไม่เป็นอิสระต่อกัน target ของวันที่ติดกันใช้ข้อมูลช่วงเดียวกันถึง 4 ใน 5 วัน ช่วงความเชื่อมั่นจึงคำนวณด้วย block bootstrap และกว้างกว่าที่จำนวนแถวบ่งบอก
- ตัวเลขบน Validation เป็นคะแนนที่ใช้เลือกการตั้งค่า จึงลำเอียงไปทางดี ตัวเลขที่ไม่ลำเอียงคือผลบน Test
- การเทียบ `with_spike` กับ `non_spike` มีปัจจัยปนกัน เพราะการตัดแถวเปลี่ยนจำนวนแถว สัดส่วนคลาส และช่วงตลาดในข้อมูลฝึกไปพร้อมกัน
- ค่า AP เทียบข้ามชุดข้อมูลไม่ได้ เพราะขึ้นกับสัดส่วนวันผันผวนสูง ซึ่งเป็น 9.2% ใน Validation และ 19.6% ใน Test
- งานนี้ศึกษาสินทรัพย์เดียวในช่วงเวลาเดียว ผลอาจใช้กับสินทรัพย์หรือช่วงเวลาอื่นไม่ได้

## เอกสารประกอบ

- สารบัญเอกสารทั้งหมด: [`docs/README.md`](docs/README.md)
- การออกแบบระบบและเหตุผลของการตัดสินใจ: [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md)
- รายละเอียดของแต่ละโมเดล: [`docs/design/MODEL_CARDS.md`](docs/design/MODEL_CARDS.md)
- ผลตรวจความถูกต้องของข้อมูล: [`docs/audit/DATA_PIPELINE_AUDIT.md`](docs/audit/DATA_PIPELINE_AUDIT.md)
- วิธีสร้างรายงานจากต้นฉบับ: [`docs/report/README.md`](docs/report/README.md)
- โจทย์ของรายวิชา: [`_69-1-01076641_Project-description.pdf`](_69-1-01076641_Project-description.pdf)
