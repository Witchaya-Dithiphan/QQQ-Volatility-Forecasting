# random_forest — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex · **สเปก:** [card #12](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | depth | trees | PCA | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 3 | 100 | none | 0.254788 | 0.686202 | 0.291667 | 0.908356 | 0.205882 | 0.979228 | `[[330,7],[27,7]]` |
| `non_spike` | 5 | 50 | none | 0.301006 | 0.676645 | 0.291667 | 0.908356 | 0.205882 | 0.979228 | `[[330,7],[27,7]]` |

ทั้งสอง variant เลือกจาก 7 candidates บน Validation (raw 4 + PCA 2/4/6); ผู้ชนะเป็น raw ทั้งคู่. `config.json`, `search_results.json` และ `load_verification.json` อยู่ใน `outputs/modeling/<variant>/classification/random_forest/`; reload ผ่านทั้งคู่และ validation predictions มี 371 แถวต่อ variant.

## Parity

`assert_parity` ผ่าน strict บน synthetic 2 seeds และ Train/Validation จริงสำหรับ `(n_estimators, max_depth) = (50,3), (100,5)` ในทั้งสอง variant: label ตรง 100%, probability ผ่าน `allclose(rtol=1e-5)`, โครงสร้าง feature/threshold ตรง sklearn ทุก tree ที่ทดสอบ. Unit tests ยืนยัน bootstrap indices, probability averaging, และ save→load. Focused suite ของ trainer, tree, forest ผ่าน 81 tests.

## การตัดสินใจ

- สุ่ม seed ต่อ tree ด้วย `RandomState(42)` แล้วใช้ seed เดียวกันกับ bootstrap และการเลือก feature ที่แต่ละ node ตาม sklearn.
- trainer ใช้ AP ก่อน AUC เพื่อเลือก best raw hyperparameters, ทดลอง PCA(2,4,6) ด้วยค่านั้น แล้วเลือก global winner จาก 7 candidates. แต่ละ search row บันทึก `pca_components`; artifacts และ reload ใช้ preprocessing ของผู้ชนะ.

## ไฟล์

- `src/ml/classification/random_forest.py`, `src/ml/classification/decision_tree.py`, `src/ml/registry.py`
- `src/ml/core/trainer.py`
- `tests/ml/classification/test_random_forest.py`, `tests/ml/core/test_trainer.py`

## ปัญหาที่เจอและแก้แล้ว

trainer เดิมใช้ preprocessing ชุดเดียวกับทุก grid candidate จึงรันได้เพียง 4 raw rows และถ้า PCA ชนะ จะ reload ด้วย validation matrix ผิดมิติ. เพิ่ม PCA อีกสาม candidate หลังหา best raw และเก็บ matrix/standardizer/components ประจำ candidate; test บังคับให้ PCA(4) ชนะตรวจทั้ง 7 rows, params ที่ reuse, metadata, และ reload. การทดสอบก่อนแก้ fail ที่ 4 rows; หลังแก้ผ่าน.

## ค้างอยู่

- `src.ml.run report` / leaderboard ยังไม่ implement (Phase C1 ของอีก branch). เมื่อพร้อม ให้ตรวจแถว `random_forest` ทั้งสอง variants แล้วเปลี่ยนสถานะเป็น ✅.
- ยังไม่ถึงขั้น finalize จึงยังไม่ใช้ Test set.
