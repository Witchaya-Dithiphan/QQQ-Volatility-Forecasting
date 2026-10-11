# decision_tree — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex · **สเปก:** [card #11](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | depth | min leaf | AUC | AP | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 2 | 5 | 0.641779 | 0.173177 | 0.309859 | 0.867925 | 0.323529 | 0.922849 | `[[311,26],[23,11]]` |
| `non_spike` | 5 | 5 | 0.692355 | 0.188709 | 0.188679 | 0.884097 | 0.147059 | 0.958457 | `[[323,14],[29,5]]` |

ทั้งสอง variant เลือกจาก 6 candidates บน Validation, preprocess `none`, `reload=ok` และ `load_verification.json` มี `"passed": true` ผลเต็มอยู่ใน `outputs/modeling/<variant>/classification/decision_tree/`

## Parity

`assert_parity` ผ่านบน synthetic 3 ชุด และ Train/Validation จริงสำหรับ **ทุก 6 candidates × 2 variants**: label ตรง 100%, probability ตรงทุกค่าที่ตรวจ (`max abs diff = 0`) โครงสร้างตรง sklearn ทุก node รวม **142 nodes** (`with_spike`) และ **138 nodes** (`non_spike`); threshold ต่างสูงสุด `0`

Weighted stump ผ่าน parity กับ `DecisionTreeClassifier.fit(..., sample_weight=...)` บนตัวอย่างควบคุมได้

## การตัดสินใจ

- `_best_split()` คำนวณ weighted Gini และเลือก feature index/threshold ต่ำสุดเมื่อ gain เสมอ ตาม task
- `fit()` เรียง feature ที่ส่งให้ `_best_split()` ตามลำดับ seeded ของ sklearn เพื่อให้ต้นไม้ตรงเมื่อ gain เสมอ; ไม่มีการเรียก sklearn ในเส้นทาง scratch
- แปลง X เป็น `float32` ก่อนหา midpoint เพราะ sklearn tree ทำเช่นเดียวกัน และเก็บ threshold ใน `float64`
- เก็บ tree เป็น NumPy arrays และ verify save → load ทั้ง label และ probability

## ไฟล์

- `src/ml/classification/decision_tree.py`
- `tests/ml/classification/test_decision_tree.py` (21 tests)
- `src/ml/registry.py`

## ปัญหาที่เจอและแก้แล้ว

เมื่อขยาย parity ไปถึง depth 5, โครงสร้างเริ่มต่างที่ node 11/32 (`with_spike`) และ node 4/47 (`non_spike`) ทำให้ probability ต่างสูงสุด 0.2879 และบาง label ต่าง สาเหตุคือหลาย feature มี Gini gain **เท่ากันเป๊ะ** แต่ sklearn สลับลำดับ feature ด้วย RNG ภายในแม้ `splitter="best"`; การเลือก feature index ต่ำสุดจึงไม่ตรง reference วิธีแก้คือจำลองลำดับเลือก feature ของ sklearn ด้วย seed 42 แล้วใช้ `_best_split()` เดิมบนลำดับนั้น ผลหลังแก้ต่าง `0` ทุก candidate

`non_spike` มี 3 แถวที่ probability เท่ากับ 0.5: `DecisionTree.predict()` ให้ label 0 ตาม sklearn แต่ trainer เดิมคำนวณ metric ด้วย `score >= 0.5` จึงเคยรายงาน accuracy 0.876011 ทั้งที่ `validation_predictions.csv` ถูกจริง 328/371 = 0.884097 แก้ core ให้ confusion metric ใช้ label จาก `model.predict()` ส่วน AUC/AP ยังคำนวณจาก score แล้วเทรนทั้งสอง variant ใหม่; ค่าในตารางนี้ตรงกับ prediction CSV แล้ว

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/decision_tree.md -->
### ผลจริงของ `decision_tree`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"max_depth": 2, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.1732 | 0.6418 | 0.3099 | 0.3235 | 0.2973 | 0.9228 | 0.8679 |
| with_spike | test (one-shot) | {"max_depth": 2, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.3792 | 0.7450 | 0.4800 | 0.4932 | 0.4675 | 0.8633 | 0.7909 |
| non_spike | validation (selection score) | {"max_depth": 5, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.1887 | 0.6924 | 0.1887 | 0.1471 | 0.2632 | 0.9585 | 0.8841 |
| non_spike | test (one-shot) | {"max_depth": 5, "max_features": null, "min_samples_leaf": 5, "random_state": 42} | 0.3844 | 0.6916 | 0.4310 | 0.3425 | 0.5814 | 0.9400 | 0.8231 |
<!-- END GENERATED -->
