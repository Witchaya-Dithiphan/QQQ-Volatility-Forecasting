# random_forest — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex · **สเปก:** [Random Forest card](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | depth | trees | PCA | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 3 | 100 | none | 0.254788 | 0.686202 | 0.291667 | 0.908356 | 0.205882 | 0.979228 | `[[330,7],[27,7]]` |
| `non_spike` | 5 | 50 | none | 0.301006 | 0.676645 | 0.291667 | 0.908356 | 0.205882 | 0.979228 | `[[330,7],[27,7]]` |

ทั้งสอง variant เลือกจาก 7 candidates บน Validation (raw 4 + PCA 2/4/6); ผู้ชนะเป็น raw ทั้งคู่. `config.json`, `search_results.json` และ `load_verification.json` อยู่ใน `outputs/modeling/<variant>/classification/random_forest/`; reload ผ่านทั้งคู่และ validation predictions มี 371 แถวต่อ variant.

## Parity

`assert_parity` ผ่าน strict บน synthetic 2 seeds และ Train/Validation จริงสำหรับ `(n_estimators, max_depth) = (50,3), (100,5)` ในทั้งสอง variant: label ตรง 100%, probability ผ่าน `allclose(rtol=1e-5)`, โครงสร้าง feature/threshold ตรง sklearn ทุก tree ที่ทดสอบ. ทดสอบเพิ่มหลัง PCA(2,4,6) บน Train/Validation ทั้งสอง variant แล้วผ่าน strict เช่นกัน; test ดัก `sample_weight` ที่ส่งเข้าต้นไม้และยืนยัน bootstrap count ตรง sklearn โดยตรง. Unit tests ครอบคลุม probability averaging และ save→load. Focused suite ล่าสุดของ trainer, preprocess, tree, forest ผ่าน 108 tests.

## การตัดสินใจ

- สุ่ม seed ต่อ tree ด้วย `RandomState(42)` แล้วใช้ seed เดียวกันกับ bootstrap และการเลือก feature ที่แต่ละ node ตาม sklearn.
- trainer ใช้ AP ก่อน AUC เพื่อเลือก best raw hyperparameters, ทดลอง PCA(2,4,6) ด้วยค่านั้น แล้วเลือก global winner จาก 7 candidates. แต่ละ search row บันทึก `pca_components` และ status; artifacts และ reload ใช้ preprocessing ของผู้ชนะ.

## ไฟล์

- `src/ml/classification/random_forest.py`, `src/ml/classification/decision_tree.py`, `src/ml/registry.py`
- `src/ml/core/trainer.py`
- `tests/ml/classification/test_random_forest.py`, `tests/ml/core/test_trainer.py`

## ปัญหาที่เจอและแก้แล้ว

เมื่อเปิด `max_features="sqrt"` ใน Decision Tree การสุ่มบาง node ของ sklearn ยอมรับ split ที่ Gini gain เป็นศูนย์ และความต่างจากการปัดเศษระดับ `1e-17` เคยทำให้เลือกฟีเจอร์คนละตัวจน label ต่าง 3 แถวบนข้อมูลสังเคราะห์. เพิ่ม regression tests สำหรับทั้งสองกรณีและปรับเฉพาะทางเดินสุ่มฟีเจอร์ โดยคงพฤติกรรม T1 เมื่อ `max_features=None`; รีวิวตรวจ bootstrap tree เพิ่ม 400 ชุดแล้ว probability ตรง reference ทั้งหมด.

trainer เดิมใช้ preprocessing ชุดเดียวกับทุก grid candidate จึงรันได้เพียง 4 raw rows และถ้า PCA ชนะ จะ reload ด้วย validation matrix ผิดมิติ. เพิ่ม PCA อีกสาม candidate หลังหา best raw และเก็บ matrix/standardizer/components ประจำ candidate; test บังคับให้ PCA(4) ชนะตรวจทั้ง 7 rows, params ที่ reuse, metadata, และ reload. การทดสอบก่อนแก้ fail ที่ 4 rows; หลังแก้ผ่าน.

เมื่อ `PCA(6)` เกิน numerical rank เดิม trainer หยุดทั้ง grid. ตอนนี้บันทึกแถวที่ 7 เป็น `invalid` พร้อม AP/AUC เป็น `null` และเหตุผล; error ประเภทอื่นยัง raise. เดิม reload ตรวจเฉพาะ model บน validation matrix ในหน่วยความจำ ทำให้ไม่ตรวจ preprocessor ที่ save. ตอนนี้อ่าน `preprocessor.npz` กลับมาแล้ว transform Validation ดิบด้วย mean/scale, constant mask และ PCA components; test ปรับ components ที่อ่านกลับให้ผิดแล้วตรวจว่า verification fail. อีก test ใช้ feature คงที่บน Train แต่ต่างบน Validation เพื่อยืนยันว่าการ reload ยัง zero feature นั้นตาม Standardizer. Focused tests ของ trainer, preprocess, tree, forest ผ่าน 108 tests.

## ค้างอยู่

- `src.ml.run report` / leaderboard ยังไม่ implement (Phase C1 ของอีก branch). เมื่อพร้อม ให้ตรวจแถว `random_forest` ทั้งสอง variants แล้วเปลี่ยนสถานะเป็น ✅.
- ยังไม่ถึงขั้น finalize จึงยังไม่ใช้ Test set.
