# stacking — classification · 🟡 รอ leaderboard

**ผู้รับผิดชอบ:** วิชญา (ตั้งแต่ขั้น T8 พิทวัสทำต่อ) · **สเปก:** [card #16](../../design/MODEL_CARDS.md) · **วันที่:** 2026-10-11

**Dependency:** `logistic`, `decision_tree`, `knn` — ใช้ class เดิมทั้งสาม ไม่ได้แก้ไฟล์ใด

## ผล (validation)

| variant | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 0.226248 | 0.664078 | 0.235294 | 0.894879 | 0.176471 | 0.967359 | `[[326,11],[28,6]]` |
| `non_spike` | 0.231130 | 0.665038 | 0.196078 | 0.889488 | 0.147059 | 0.964392 | `[[325,12],[29,5]]` |

grid ว่าง → 1 candidate ต่อ variant. ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/stacking/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่.

| variant | น้ำหนัก meta: logistic | tree | k-NN | intercept | แถวที่ meta ใช้ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `with_spike` | 3.2754 | 0.8258 | 0.7285 | −2.0932 | 1,040 |
| `non_spike` | 1.8884 | 1.2072 | 1.7592 | −2.2155 | 852 |

fold จริง (ตำแหน่งบน timeline เดิม 1,733 วัน; block เริ่มที่ 693 = 40%):

| block | `with_spike` train / ทำนาย | `non_spike` train / ทำนาย | วันสุดท้ายที่ใช้ train |
| --- | ---: | ---: | --- |
| [693, 953) | 688 / 260 | 663 / 183 | 687 (ทั้งคู่) |
| [953, 1213) | 948 / 260 | 848 / 233 | 947 / 924 |
| [1213, 1473) | 1,208 / 260 | 1,079 / 226 | 1,207 (ทั้งคู่) |
| [1473, 1733) | 1,468 / 260 | 1,305 / 210 | 1,467 (ทั้งคู่) |

ข้อสังเกตจากตัวเลข: **stacking ไม่ได้ดีกว่าโมเดลเดี่ยว** AP 0.226–0.231 ต่ำกว่าหลายตัว (เช่น `gradient_boosting` 0.253). meta-learner เรียนจากแค่ 852–1,040 แถวที่มี positive น้อย และ base ทั้งสามใช้ค่า default ไม่ใช่ค่าที่ tune แล้ว. Validation มี positive 34 แถว ยังไม่ได้ทดสอบนัยสำคัญ.

## Parity

`StackingClassifier` ของ sklearn **ใช้เทียบตัวเลขไม่ได้** (ดูหัวข้อปัญหา). การตรวจที่เข้มจริงคือ **rebuild fold ต่อ fold**: ประกอบ protocol เดียวกันจาก sklearn reference ของ base แต่ละตัว (`LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`) บน fold ชุดเดียวกัน แล้วเทียบทั้ง meta features และ probability สุดท้าย.

วัดบนข้อมูลจริงทั้งสอง variant:

| | `with_spike` | `non_spike` |
| --- | ---: | ---: |
| meta feature ของ tree ต่างสูงสุด | 0 | 0 |
| meta feature ของ k-NN ต่างสูงสุด | 0 | 0 |
| meta feature ของ logistic ต่างสูงสุด | 2.1e-07 | 3.9e-07 |
| probability สุดท้ายต่างสูงสุด (Train / Validation) | 5.3e-07 / 4.3e-07 | 6.2e-07 / 3.6e-07 |
| label ต่าง | 0 | 0 |

`test_stacking.py` 25 tests ผ่าน.

**การตรวจ leakage (ไม่พึ่ง reference):** สำหรับแต่ละ block ทั้ง 4 — กลับ label ทุกแถวตั้งแต่ต้น purge window เป็นต้นไป และแยกอีกรอบเขียนทับ feature ของแถวใน purge window กับแถวหลัง block ด้วยค่าขยะ — meta features ของ block นั้นต้องเท่าเดิม**ทุก bit**. ผ่านทั้ง 8 กรณี: base ของแต่ละ fold (รวม scaler ของมัน) ไม่เห็นอะไรจากปัจจุบันหรืออนาคต.

## การตัดสินใจ

- **purge นับวันทำการบน timeline เดิม ไม่ใช่นับแถว.** ต้องให้ trainer ส่ง `positions` (index ของแต่ละแถวบน timeline ของ `with_spike`). บน `non_spike` การนับสองแบบให้ผลต่างจริง: **166 แถวอยู่คนละ fold** และ meta ได้ 852 แถวแทน 912. ทิศทางของความเสี่ยงต่างจากที่ card เขียน: นับ 5 แถวบน `non_spike` ครอบเวลา ≥ 5 วันทำการเสมอ จึงไม่รั่วมากขึ้น — ที่ผิดคือ **ขอบ block เลื่อน** ทำให้ fold ของสอง variant ไม่ตรงช่วงเวลาเดียวกันและเทียบกันไม่ได้.
- **base logistic รันจนลู่เข้า** (`learning_rate=0.5, max_iter=60000, tol=1e-9`) ไม่ใช่ค่า default. ดูหัวข้อปัญหา.
- **base ใช้ hyperparameter default** (`DecisionTree(max_depth=3, min_samples_leaf=5)`, `KNN(k=5, uniform)`, logistic ไม่มี L2) ตาม "fixed base/meta set" ของแผน ไม่ดึงค่าที่ชนะจาก run เดี่ยวของแต่ละตัว — เพื่อไม่ให้ stacking ขึ้นกับไฟล์ผลลัพธ์ของโมเดลอื่น.
- **preprocessing เป็นของ base แต่ละตัว** (`base_specific`): logistic กับ k-NN standardize, tree ใช้ค่าดิบ. ใน fold scaler fit บนแถว train ของ fold นั้นเท่านั้น. trainer จึงต้องไม่ standardize ซ้ำ.
- **meta เป็น `Logistic` scratch** บน 3 คอลัมน์ probability ดิบ ด้วย step ใหญ่ขึ้น (`learning_rate=1.0, max_iter=200000, tol=1e-8`) เพราะ feature สามตัว correlate กันและอยู่ในช่วง [0, 1] ทำให้ gradient descent ช้า. เทียบ `LogisticRegression` บน feature ชุดเดียวกัน: coef ตรงที่ rtol 1e-4, probability ที่ 1e-6.
- fold ที่ประวัติมีคลาสเดียวหรือไม่มีแถวให้ทำนาย → `ValueError` ตาม `invalid_fold: failed`.

## ไฟล์

- `src/ml/classification/stacking.py`, `src/ml/registry.py`
- `src/ml/core/trainer.py` — ส่ง `positions`, ไม่ scale เมื่อ `base_specific`
- `tests/ml/classification/test_stacking.py`, `tests/ml/core/test_trainer.py`

## ปัญหาที่เจอและแก้แล้ว

**1. reference ใน card รันไม่ได้.** `StackingClassifier(cv=TimeSeriesSplit(4))` โยน `ValueError: cross_val_predict only works for partitions` — sklearn บังคับว่าทุกแถวต้องถูกทำนายโดย fold ใด fold หนึ่ง ซึ่ง time-series split ไม่ทำ (ช่วงแรกไม่มีใครทำนาย). sklearn จึง**แสดง protocol นี้ไม่ได้เลย**. `reference()` ใช้ `KFold(4, shuffle=False)` แทนเป็น behavioural reference ซึ่ง block แรก ๆ ถูกทำนายจากโมเดลที่เห็นอนาคต ตัวเลขจึงต่างโดยเจตนา. การตรวจความถูกต้องย้ายไปที่ rebuild fold ต่อ fold.

**2. meta features ของ logistic ห่างจาก reference 0.03–0.07 — สาเหตุอยู่ที่ base ไม่ใช่ stacking.** อาการ: rebuild ตรงพอดีสำหรับ tree และ k-NN แต่คอลัมน์ logistic ต่างถึง 0.073. วัดทีละ fold: `Logistic()` ด้วยค่า default ตาม config (`learning_rate=0.05, max_iter=5000`) ชน max_iter ทุก fold โดย gradient ยังอยู่ที่ ~1e-3. ให้งบพอแล้วลู่เข้าใน 22,000–38,000 รอบและตรง reference ที่ 5e-5. แก้ใน stacking โดยให้ base logistic ใช้ค่าที่ลู่เข้า.

> **ผลต่อโมเดล `logistic` เดี่ยว — เล็ก และไม่ได้แก้:** ค่าตาม config ก็ไม่ลู่เข้าบน train เต็มเช่นกัน. ทั้ง 12 การรัน (6 candidates × 2 variants) หยุดเพราะครบ 5,000 รอบ ไม่ใช่เพราะถึง `tol`. เทียบกับโมเดลที่รันจนลู่เข้า บน Validation: probability ต่างสูงสุด 0.0079, AP ต่างสูงสุด 0.0007, AUC ต่างสูงสุด 0.0012, label ต่างสูงสุด 1 จาก 371 แถว; ผู้ชนะทั้งสอง variant ได้ AP/AUC เท่าเดิมถึงทศนิยม 4 ตำแหน่ง. test parity ของ `logistic` ตั้งใจรันจนลู่เข้าและเขียนเหตุผลไว้ใน docstring แล้ว (ถามว่าอัลกอริทึมถูกไหม ไม่ใช่ถามว่า production ใช้งบเท่าไร) จุดที่ตกหล่นคือ docstring วัดบน fixture ที่หยุดที่รอบ 550 จึงไม่เห็นว่าข้อมูลจริงชน max_iter. ผลกระทบจริงอยู่ที่ stacking ซึ่งเทียบ probability ทีละแถวบน fold ที่เล็กกว่า (ห่าง 0.03–0.07) ไม่ใช่ที่ผลของ `logistic` เอง.

**3. เขียนโมเดลก่อน test.** สำหรับตัวนี้เขียน `stacking.py` ก่อน `test_stacking.py` ผิดลำดับ TDD ของโปรเจกต์. test ที่เขียนตามมาจับปัญหาข้อ 1 และ 2 ได้ทั้งคู่ แต่ไม่ได้เห็น test fail ก่อนมีโค้ด.

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant
- (ทางเลือก) เพิ่มหมายเหตุในเอกสารของ `logistic` ว่าค่า production หยุดที่ 5,000 รอบบนข้อมูลจริง — ไม่จำเป็นต้องแก้โมเดล
- ยังไม่แตะ Test set

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/stacking.md -->
### ผลจริงของ `stacking`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.2262 | 0.6641 | 0.2353 | 0.1765 | 0.3529 | 0.9674 | 0.8949 |
| with_spike | test (one-shot) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.5131 | 0.7902 | 0.5714 | 0.5753 | 0.5676 | 0.8933 | 0.8311 |
| non_spike | validation (selection score) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.2311 | 0.6650 | 0.1961 | 0.1471 | 0.2941 | 0.9644 | 0.8895 |
| non_spike | test (one-shot) | {"folds": 4, "initial_fraction": 0.4, "purge": 5, "random_state": 42} | 0.5499 | 0.7890 | 0.4853 | 0.4521 | 0.5238 | 0.9000 | 0.8123 |
<!-- END GENERATED -->
