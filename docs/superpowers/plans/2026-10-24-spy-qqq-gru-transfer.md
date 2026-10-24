# SPY to QQQ GRU Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** วัดว่า GRU ที่ pretrain บน SPY แล้ว fine-tune บน QQQ ทำนาย `target_high_volatility` ได้ดีกว่า GRU ที่ฝึกบน QQQ อย่างเดียวและ baseline เดิมหรือไม่

**Architecture:** แยกการทดลองออกจาก pipeline 19 โมเดลเดิม ใช้ฟีเจอร์และ label สูตรเดิมสร้าง window 20 วัน, train source ก่อน QQQ cutoff, fine-tune เฉพาะ QQQ Train และจัดอันดับบน Validation เท่านั้น บันทึกการตัดสินก่อนเปิด Test ที่เคยใช้แล้ว

**Tech Stack:** Python, NumPy, pandas, pytest, PyTorch ใน environment แยก (ตรวจ runtime และข้อกำหนดงานก่อนเริ่ม)

**Spec:** [แผนการทดลองเดิม](../../plan/TIME_SERIES_TRANSFER_LEARNING_PLAN.md) ซึ่งกำหนดคำถามวิจัย ตัวเลือก metric และข้อจำกัดของ Test

## Global Constraints

- ห้ามแก้ QQQ raw/processed snapshot, `outputs/modeling/`, selection lock หรือ registry ของ 19 โมเดลเดิม
- ใช้ฟีเจอร์ 8 ตัวจาก `src/ml/core/contracts.py`; source และ target ต้องใช้นิยาม forward realized volatility 5 วันเดียวกัน
- source pretrain ในแต่ละ fold ใช้ label ที่สิ้นสุดก่อน cutoff ของ fold; fit scaler, source Q75, class weight และ early stopping เฉพาะช่วงฝึกที่อนุญาต
- `with_spike` เป็นผลหลัก; `non_spike` เป็น sensitivity analysis และไม่เปลี่ยน Validation/Test
- AP เป็น metric หลัก; ทุก arm ต้องประเมินบนวันและ label ชุดเดียวกัน
- ผล Test ปัจจุบันเป็น post-hoc เพราะเคยเปิดแล้ว; ต้องมี future holdout ใหม่จึงอ้างการยืนยันอิสระได้
- `deliverables/slides.pptx` มีการแก้ค้างจากผู้ใช้ ห้าม stage หรือเปลี่ยนไฟล์นี้

## Review Focus

1. SPY/QQQ ใช้นโยบายปรับราคาไม่เท่ากัน: source loader ต้องหยุดและแจ้งเมื่อ manifest ไม่ระบุหรือไม่ผ่านการตรวจ compatibility (Task 1)
2. source row ใกล้ cutoff มี target ที่กินวันอนาคต: ทดสอบว่า label-end ไม่เกิน cutoff ในทุก fold (Task 1)
3. QQQ Validation window ต้องใช้วัน purge เป็น **feature context** ได้ แต่ห้ามใช้ label วัน purge ในการฝึก (Task 2)
4. Validation มี positive เพียง 34 วัน: CI ที่ resample แล้วไม่มีทั้งสอง class ต้องนับเป็น invalid ไม่แปลงเป็นศูนย์ (Task 4)
5. Test เคยเปิดแล้ว: finalize ต้องปฏิเสธการรันหากไม่มี receipt ของ configuration ที่ล็อกไว้ก่อน (Task 5)

---

## File map

- `src/ml/transfer_learning/data.py`: อ่านและตรวจ source snapshot, reuse feature/target builders, ทำ sequence window และ cutoff checks
- `src/ml/transfer_learning/model.py`: PyTorch GRU encoder/head และการบันทึก/โหลด state ที่ใช้ทุก arm ร่วมกัน
- `src/ml/transfer_learning/experiment.py`: train source, QQQ scratch/frozen/fine-tune, seed และ early stopping ภายใน Train
- `src/ml/transfer_learning/run.py`: CLI สำหรับ prepare, validate, finalize และ artifact manifest แยก
- `tests/ml/transfer_learning/test_data.py`, `test_model.py`, `test_experiment.py`, `test_run.py`: checks ของแต่ละส่วน
- `data/external/transfer_learning/spy_daily.csv` กับ manifest: source snapshot แยกจาก QQQ; บันทึก provenance/license/hash ตามสิทธิ์ของแหล่งข้อมูล
- `outputs/experiments/transfer_learning/`: checkpoints, predictions, metrics, receipt และรูป โดยไม่แตะ `outputs/modeling/`

## Task 1: Source snapshot และ label ที่ไม่ข้าม cutoff

**Files:** Create `src/ml/transfer_learning/data.py`, `tests/ml/transfer_learning/test_data.py`; add SPY snapshot + manifest under `data/external/transfer_learning/` เมื่อได้แหล่งข้อมูลที่ตรวจได้

**Interfaces:** `load_spy(path: Path, manifest_path: Path, cutoff: np.datetime64) -> pd.DataFrame` คืน frame ที่มี `Date`, 8 features, `target_volatility_5d`, `target_high_volatility`, และ `label_end_date`; เรียก `build_all_features()`/`add_regression_target()` ของเดิม ไม่เขียนทับ input

- [ ] **Step 1: เขียน failing tests** — fixture SPY ขนาดเล็กที่รู้ค่า OHLCV/อนาคต: assert source Q75 มาจากแถวก่อน cutoff เท่านั้น, แถวที่ `label_end_date > cutoff` ถูกตัด, source hash/schema ผิดถูกปฏิเสธ, manifest ที่ไม่ระบุ adjustment policy ถูกปฏิเสธ
- [ ] **Step 2: รันทดสอบให้ล้ม** — `python -m pytest tests/ml/transfer_learning/test_data.py -q`; ต้องล้มเพราะ `load_spy` ยังไม่มี
- [ ] **Step 3: implement ฟังก์ชันตาม interface** — reuse `src/download_qqq_data.py:profile_csv` สำหรับ schema/hash และ `src/build_features.py:build_all_features`, `src/build_targets.py:add_regression_target`; ไม่เรียก save functions ของ QQQ; manifest ระบุ source URL, license, retrieved_at, SHA-256, date range และ adjustment policy
- [ ] **Step 4: รันทดสอบให้ผ่าน** — คำสั่งเดิมต้อง PASS; ตรวจ profile source จริงและเปรียบเทียบ policy กับ `data/manifests/qqq_daily_snapshot.json`; หากพิสูจน์ compatibility ไม่ได้ให้บันทึก blocker ก่อน train
- [ ] **Step 5: commit เฉพาะงาน Task 1** — `git add src/ml/transfer_learning/data.py tests/ml/transfer_learning/test_data.py data/external/transfer_learning/` แล้ว `git commit -m "feat: validate SPY transfer source"` (หาก license ไม่อนุญาต commit CSV ให้เก็บไฟล์นอก Git และ commit เฉพาะ manifest/ขั้นตอนดึงข้อมูล)

## Task 2: QQQ และ SPY sequence windows

**Files:** Modify `src/ml/transfer_learning/data.py`; modify `tests/ml/transfer_learning/test_data.py`

**Interfaces:** `make_windows(features: pd.DataFrame, target_dates: np.ndarray, labels: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]` คืน `(dates, X, y)` โดย `X.shape == (n, 20, 8)`; `features` เป็น timeline เต็มที่มี gap context, `target_dates` เป็นวันที่ที่ได้รับอนุญาตให้ประเมินหรือฝึก

- [ ] **Step 1: เขียน failing tests** — สร้าง timeline 30+ วันใส่ค่าฟีเจอร์เป็นเลขลำดับ; assert window ของวัน `t` มี `t-19..t` และไม่มี `t+1`; assert target day ที่ไม่มี context ครบถูกตัด; assert Validation ใช้ feature ของช่วง purge เป็น context ได้แต่ train labels ไม่มีวัน purge; assert วันที่ output ตรงกับ accepted split ของ `src/ml/core/data.py`
- [ ] **Step 2: รันทดสอบให้ล้ม** — `python -m pytest tests/ml/transfer_learning/test_data.py -q`; test ใหม่ต้องล้มเพราะ `make_windows` ยังไม่มี
- [ ] **Step 3: implement `make_windows`** — ดึง QQQ context จาก `data/interim/qqq_features.csv` โดยตรวจวันและค่า 8 ฟีเจอร์ที่ทับกับ accepted split; ใช้ `load()`/`load_test(allow_test=True)` เฉพาะใน stage ที่เหมาะสมเพื่อรับ target dates/labels; ห้ามต่อ Train กับ Validation แล้วถือว่าวัน purge ไม่มีอยู่
- [ ] **Step 4: รันทดสอบให้ผ่าน** — คำสั่งเดิม PASS; assert วัน/label ของ baselines และ GRU ตรงกันหลัง align
- [ ] **Step 5: commit** — `git add src/ml/transfer_learning/data.py tests/ml/transfer_learning/test_data.py` แล้ว `git commit -m "feat: build causal transfer windows"`

## Task 3: GRU arm ที่เทียบกันได้

**Files:** Create `src/ml/transfer_learning/model.py`, `src/ml/transfer_learning/experiment.py`, `tests/ml/transfer_learning/test_model.py`, `tests/ml/transfer_learning/test_experiment.py`

**Interfaces:** `GRUClassifier(input_size: int = 8, hidden_size: int = 16)` คืน logit ต่อ window; `fit_arm(source: tuple | None, target: tuple, arm: str, seed: int, config: dict) -> tuple[GRUClassifier, dict]` รองรับ `scratch`, `source_only`, `frozen`, `fine_tuned`; `predict_scores(model, X) -> np.ndarray` คืน probability

- [ ] **Step 1: ตรวจ Python/PyTorch** — `python -c "import sys, torch; print(sys.version); print(torch.__version__)"`; ถ้า runtime ที่ใช้ไม่รองรับ ให้สร้าง environment แยกและบันทึกเวอร์ชันก่อนเขียน model; ไม่เปลี่ยน `requirements-lock.txt` ของงานหลัก
- [ ] **Step 2: เขียน failing tests** — tensor `(4,20,8)` ให้ output `(4,)` ที่ finite; seed เดียวกันให้คะแนนซ้ำตรงกันบน CPU; `frozen` ไม่แก้ encoder แต่ `fine_tuned` แก้ได้; source-only ไม่มีขั้น fit QQQ; class weight และ early-stopping split ใช้ QQQ Train เฉพาะส่วนที่อนุญาต
- [ ] **Step 3: รันทดสอบให้ล้ม** — `python -m pytest tests/ml/transfer_learning/test_model.py tests/ml/transfer_learning/test_experiment.py -q`; ต้องล้มเพราะ interfaces ยังไม่มี
- [ ] **Step 4: implement อย่างน้อยที่พอทดสอบผ่าน** — GRU 1 ชั้น hidden 16, binary head, BCEWithLogitsLoss; source pretrain + target fine-tune ใช้ architecture/seed/normalization เดียวกับ scratch; early stop บน QQQ Train ส่วนท้ายที่เว้น purge; บันทึก epoch/weights/scaler/seed
- [ ] **Step 5: รันทดสอบให้ผ่านและ commit** — คำสั่งเดิม PASS; `git add src/ml/transfer_learning/model.py src/ml/transfer_learning/experiment.py tests/ml/transfer_learning/test_model.py tests/ml/transfer_learning/test_experiment.py`; `git commit -m "feat: compare GRU transfer arms"`

## Task 4: Validation, baseline และเกณฑ์ตัดสิน

**Files:** Create `src/ml/transfer_learning/run.py`, `tests/ml/transfer_learning/test_run.py`; write only under `outputs/experiments/transfer_learning/`

**Interfaces:** `python -m src.ml.transfer_learning.run validate --spy <path> --manifest <path> --variant with_spike`; output `validation_predictions.csv`, `validation_metrics.json`, `decision.json`, checkpoints และ manifest

- [ ] **Step 1: เขียน failing tests** — fixture predictions ที่วันต่างกันต้องถูกปฏิเสธ; AP difference ใช้ paired dates/labels และ `src/ml/core/bootstrap.py` 20-day blocks, 2,000 reps; resample ที่ class เดียวต้องเป็น invalid; `decision.json` เก็บ configuration, source/data hashes, seeds, threshold และเปรียบเทียบ `fine_tuned` กับ scratch, MLP, persistence
- [ ] **Step 2: รันทดสอบให้ล้ม** — `python -m pytest tests/ml/transfer_learning/test_run.py -q`; test ใหม่ต้องล้ม
- [ ] **Step 3: implement validate CLI** — ใช้ `src/ml/core/report.py:classification_scores`, baseline prediction artifacts เดิมและ bootstrap helper เดิม; fix candidate set ล่วงหน้า: hidden 16, lookback 20, seeds 3 ค่า, `with_spike` หลัก; threshold เลือกโดยไม่ใช้ Test; walk-forward ภายใน Train ใช้ cutoff source ราย fold และไม่ใช้ fold score เดียวกันเป็นทั้งตัวเลือกและหลักฐานอิสระ
- [ ] **Step 4: รันทดสอบให้ผ่านและรัน Validation จริง** — `python -m pytest tests/ml/transfer_learning/test_run.py -q` PASS; validate CLI ต้องสร้าง artifacts พร้อมจำนวนวัน/positive ที่ตรงทุก arm และไม่เปิด Test
- [ ] **Step 5: สรุปผลตามเกณฑ์ที่ประกาศไว้** — transfer ช่วยต่อเมื่อ fine-tuned AP > scratch AP โดยไม่อาศัย seed เดียว; เหนือ baseline ชัดเจนต่อเมื่อ AP มากกว่า MLP และ persistence อย่างน้อย 0.05 และ paired 95% CI ของทั้งสองผลต่างไม่คร่อม 0; หากไม่ผ่านให้รายงานผลจริง ไม่จูนกับ Test
- [ ] **Step 6: commit** — `git add src/ml/transfer_learning/run.py tests/ml/transfer_learning/test_run.py outputs/experiments/transfer_learning/`; `git commit -m "feat: evaluate transfer on validation"`

## Task 5: ล็อกผล เลือก Test ครั้งเดียว และรายงาน

**Files:** Modify `src/ml/transfer_learning/run.py`, `tests/ml/transfer_learning/test_run.py`, `notebooks/04_classification.ipynb`, `notebooks/05_results_comparison.ipynb`, `docs/report/REPORT_DRAFT.md`; output test artifacts only under `outputs/experiments/transfer_learning/`

**Interfaces:** `python -m src.ml.transfer_learning.run finalize --decision <path> --allow-test`; ตรวจ code/config/source hashes ของ receipt ก่อน `load_test(allow_test=True)` และเขียน `test_predictions.csv`, `test_metrics.json`, figures, `test_receipt.json`

- [ ] **Step 1: เขียน failing tests** — ไม่ระบุ `--allow-test`, ไม่มี decision, hash mismatch หรือ configuration ต่างจาก Validation ต้องปฏิเสธก่อนอ่าน Test; finalize ซ้ำต้องไม่เลือก model/threshold ใหม่; output ระบุ `post_hoc: true`
- [ ] **Step 2: รันทดสอบให้ล้ม** — `python -m pytest tests/ml/transfer_learning/test_run.py -q`; test ใหม่ต้องล้ม
- [ ] **Step 3: implement finalize guard และรายงาน** — ล็อก choice จาก Validation, ประเมิน baseline บนวันเดียวกัน, สร้าง AP/ROC-AUC/precision/recall/confusion/PR/calibration/paired CI; เพิ่ม section ของการทดลองเพิ่มใน notebook 04/05 และรายงานโดยอ่าน artifacts ไม่ hardcode ตัวเลขหรือเปลี่ยน leaderboard 19 โมเดลหลัก
- [ ] **Step 4: ตรวจทั้งหมดก่อนรัน Test** — `python -m pytest tests/ml/transfer_learning tests/ml/core/test_data.py tests/test_notebooks.py tests/test_docs_sync.py -q` PASS; ตรวจ manifest, receipt, checkpoint reload และว่า `git status` ไม่มีการแก้ข้อมูล/โมเดลหลัก; freeze commit/config
- [ ] **Step 5: เปิด Test หนึ่งครั้งแล้วตรวจผล** — รัน finalize command บน configuration ที่ล็อกไว้; ตรวจ artifacts ว่าจำนวนวัน/label ตรงกัน, metric คำนวณซ้ำได้ และทุกที่ระบุว่า post-hoc; หาก source ไม่มีหรือ Validation ไม่ผ่านเกณฑ์ ก็ยังบันทึกผลลบอย่างตรงไปตรงมาได้โดยไม่ปรับ Test
- [ ] **Step 6: commit ผลและเอกสาร** — stage เฉพาะไฟล์การทดลอง/notebooks/report ที่ตรวจแล้ว, ไม่รวม `deliverables/slides.pptx`; commit พร้อมสรุป Test และข้อจำกัด

## Self-review ที่ต้องทำก่อนส่งมอบ implementation

- ตรวจว่าแต่ละ Task มี test ที่ล้มก่อนและผ่านหลัง implementation, แยกจาก main model registry
- ตรวจว่า source provenance/adjustment และวัน label-end ของ SPY พิสูจน์ได้จริง; หากไม่ได้ให้หยุดและรายงาน blocker
- ตรวจว่า Test ของงานใหม่นี้ไม่ถูกเรียกว่า blind test และตัวเลข baseline เทียบบน aligned dates
- ตรวจว่าไม่มีส่วนใดรับประกัน AP สูงขึ้น; ผลลบ/negative transfer เป็นผลการทดลองที่ยอมรับได้
