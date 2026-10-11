# Phase C — Report, Robustness and Finalize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** เปลี่ยน artifact 38 ชุดที่เทรนเสร็จแล้วให้เป็นผลงานที่ส่งได้ — leaderboard พร้อมช่วงความเชื่อมั่น, การเทียบ `with_spike` ↔ `non_spike`, feature importance, ตาราง walk-forward, ผล Test ครั้งเดียว, notebook 03/04/05, เอกสารที่คนนอกใช้ตามได้, รายงานภาษาไทยและสไลด์ 22 แผ่น

**Architecture:** ทุกอย่างใน Phase C **อ่าน artifact แล้วคำนวณต่อ ไม่เทรนผู้ชนะใหม่** โมดูลใหม่ใน `src/ml/core/` แยกตามหน้าที่เดียว (`artifacts` อ่าน run กลับมา · `bootstrap` สุ่มซ้ำแบบ block · `report` ทำ leaderboard · `comparison` เทียบ A↔B · `importance` · `walk_forward`) ส่วน `src/ml/finalize.py` รับ array ของ Test จาก `run.py` เท่านั้น จึงไม่ต้องเพิ่มข้อยกเว้นใน static guard ตัวเลขทุกตัวในรายงานถูก generate ลงไฟล์ใต้ `outputs/` แล้วเอกสารดึงจากไฟล์นั้น ไม่พิมพ์มือ

**Tech Stack:** Python 3.14.8 · NumPy 2.5.3 · pandas 3.0.6 · matplotlib 3.11.2 · nbconvert 7.17.1 · pytest 9.1.1 · scikit-learn 1.9.1 / xgboost 3.4.1 (reference เท่านั้น)

**Spec:** [`AGENTS.md`](../../../AGENTS.md) (contract) · [`docs/superpowers/specs/2026-10-10-modeling-layer-design.md`](../specs/2026-10-10-modeling-layer-design.md) · [`docs/audit/DATA_PIPELINE_AUDIT.md`](../../audit/DATA_PIPELINE_AUDIT.md) (findings E9, E10, D12, D13, F13, F15) · [`docs/plan/REPORT_AND_SLIDES_OUTLINE.md`](../../plan/REPORT_AND_SLIDES_OUTLINE.md) · โจทย์ `_69-1-01076641_Project-description.pdf`

**คำสั่งรัน test ตลอดแผนนี้:** `.\.venv\Scripts\python.exe -m pytest <path> -v` (PowerShell จาก project root)

## Global Constraints

- ใช้ `.\.venv\Scripts\python.exe` เสมอ (Python 3.14.8) · ห้ามเพิ่ม dependency ลง `requirements-lock.txt`
- **ห้ามแก้ชั้นข้อมูล:** `data/**`, `src/clean_data.py`, `build_features.py`, `build_targets.py`, `split_data.py`, `detect_spikes.py`, `build_experiment_datasets.py`, `run_data_pipeline.py`, `run_spike_analysis.py`
- **ห้ามแตะ Test ก่อน Task C4.3** — ทุก task ก่อนหน้านั้นใช้ Train/Validation เท่านั้น และ test ใน pytest ห้ามอ่าน `test_labeled.csv` / `test_flagged.csv` เพิ่มจากที่มีอยู่เดิม
- **ไม่ refit บน Train+Validation · ไม่เลือก hyperparameter ใหม่** — ผู้ชนะคือค่าใน `config.json` ของแต่ละ run
- โมเดลทุกตัวเป็น NumPy from scratch · sklearn/xgboost ใช้ได้เฉพาะผ่าน `model.reference()`
- สุ่มด้วย `np.random.RandomState(seed)` เท่านั้น · `tests/test_audit_guards.py::test_src_never_shuffles_or_splits_at_random` ห้ามสตริง `train_test_split`, `shuffle=True`, `permutation(`, `default_rng` ในทุกไฟล์ใต้ `src/` (รวม docstring และ comment)
- เฉพาะ `src/ml/core/data.py`, `src/ml/core/contracts.py`, `src/ml/run.py` เท่านั้นที่มีสตริง `load_test(`, `TEST_DATA_PATH`, `TEST_LABELED_DATA_PATH`, `TEST_FLAGGED_PATH`, `test_labeled`, `test_flagged` ได้ — โมดูลใหม่ทุกตัวต้องไม่มีสตริงเหล่านี้
- Bootstrap: moving block, block = **20** วันทำการ, B = **2000**, seed = **20261011**, ช่วง 95% แบบ percentile
- ตัวเลข Validation ทุกตัวต้องเรียกว่า **selection score** · ห้ามอ้างอันดับระหว่างโมเดลที่ช่วงของผลต่างคร่อมศูนย์
- ห้ามรายงาน accuracy เดี่ยว ๆ — ต้องมี AP, ROC-AUC, F1, recall คู่กันเสมอ (majority baseline accuracy = 337/371)
- ผลต่าง A↔B ต้องเขียนว่าเป็น **ผลของนโยบายตัดแถวทั้งชุด** (class prior 25.0% → 18.4%, 1,733 → 1,520 แถว, regime mix)
- เอกสาร: ภาษาไทย ศัพท์เทคนิคอังกฤษ · โค้ดและคำสั่ง: อังกฤษ
- **ห้าม commit/push เองโดยไม่ถามเจ้าของ** — ทุก step "Commit" ในแผนนี้ให้แสดง `git status --short` + สรุปการเปลี่ยนแปลงแล้วรออนุมัติก่อน
- ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและแสดง output จริง (AGENTS.md กฎข้อ 6)

## Review Focus

1. **Bootstrap replicate ที่ไม่มี positive เลย** — Validation มี positive 34 แถวและ 20 แถวกระจุกใน 2024Q3 การสุ่ม block อาจได้ชุดที่ไม่มี positive → AP/AUC เป็น undefined ต้องถูกนับและรายงาน (`n_valid`) ไม่ใช่ถูกตัดเงียบ ๆ และถ้าใช้ได้ไม่ถึง 90% ต้องไม่พิมพ์ช่วง → test `test_undefined_replicates_are_counted_not_hidden` (Task C1.2)
2. **Artifact ล้าสมัย** — มีคนเทรนโมเดลซ้ำหลังสร้าง leaderboard หรือหลังล็อก → `report` ต้องปฏิเสธ run ที่ `validation_sha256` ไม่ตรงข้อมูลปัจจุบัน และ `finalize` ต้องปฏิเสธเมื่อ hash ต่างจาก lock → test `test_stale_artifact_is_rejected` (C1.4), `test_changed_artifact_blocks_finalize_before_scoring` (C4.2)
3. **รัน `finalize` ซ้ำโดยไม่ตั้งใจ** — ต้องปฏิเสธ **ก่อน** เปิดไฟล์ Test ไม่ใช่หลังอ่านแล้ว → test `test_finalize_refuses_a_second_run_before_reading_the_holdout` (C4.3)
4. **Metric ที่ undefined ในตาราง** — โมเดลที่ไม่ทำนาย positive เลย (`perceptron` F1 = 0, precision หารศูนย์) ต้องขึ้น `n/a` และถูกเรียงไว้ท้ายตารางทั้งแบบ ascending และ descending → test `test_leaderboard_puts_undefined_metrics_last_when_descending` (C1.3)
5. **Segment ที่มีคลาสเดียวหรือแถวน้อย** — Test ช่วง spike-affected มีแค่ 54 แถว บาง metric อาจคำนวณไม่ได้ ต้องเขียนเหตุผลลงไฟล์ ไม่ crash และไม่ใส่ช่วงความเชื่อมั่นที่ไม่มีความหมาย → test `test_one_class_segment_is_reported_with_a_reason` (C4.2)

---

## ภาพรวม

### Phase และชั่วโมง

| Phase | เนื้อหา | ชม. | ตัดก่อนถ้าเวลาไม่พอ |
| --- | --- | ---: | --- |
| C1 | `run.py report` — leaderboard + CI + baseline + รูป | 8 | residual plot → รูป PR → Markdown ของ clustering |
| C2 | paired comparison A↔B | 3 | CI ของ metric รอง (เหลือเฉพาะ metric หลัก) |
| C3 | feature importance + walk-forward | 6 | walk-forward ของ `mlp`/`slp`/`stacking` → ลด repeats 30 → 10 |
| C4 | `run.py finalize --allow-test` (**STOP gate**) | 5 | CI บน Test ของ metric รอง (ห้ามตัด segment และ receipt) |
| C5 | notebooks 03 / 04 / 05 | 8 | cell error analysis ของ 05 → cell PCA search ของ 04 |
| C6 | documentation pass | 5 | `docs_sync` อัตโนมัติ → test ตรวจลิงก์ |
| C7 | รายงานไทย + 22 สไลด์ | 12 | สไลด์ S18, S21, S3 ตามลำดับใน outline |
| | **รวม** | **47** | |

### Dependency graph

```text
C1.1 artifacts ──► C1.2 bootstrap ──► C1.3 figures ──► C1.4 report core ──► C1.5 CLI report
                                                            │
                         ┌──────────────────────────────────┼─────────────────────┐
                         ▼                                  ▼                     ▼
                  C2 comparison                     C3 importance          C4.1 data segments
                         │                          C3 walk-forward        C4.2 finalize module
                         │                                  │              C4.3 lock ► ⛔ STOP ► run
                         └──────────────┬───────────────────┘                     │
                                        ▼                                         │
                              C5 notebooks 03, 04 ◄── (05 ส่วน Test รอ C4.3) ◄────┘
                                        ▼
                              C6 docs ──► C7 report ──► slides ──► ส่ง
```

**Critical path:** C1.1 → C1.2 → C1.4 → C1.5 → C4.2 → C4.3 (STOP gate) → C5 (05) → C7
C2 และ C3 ขนานกับ C4 ได้หลัง C1.5 แต่ **ต้องเสร็จก่อนกด finalize** เพราะหลังเห็น Test แล้วห้ามเพิ่มการวิเคราะห์ที่เปลี่ยนการตัดสินใจ

### แผนรายวัน (ทำงานร่วมกันบนเครื่องเดียว ไม่แบ่ง owner)

| วัน | งาน | Gate ก่อนจบวัน |
| --- | --- | --- |
| **12 อา.** | C1 ทั้งหมด | `run.py report` ทั้งสอง variant exit 0 · leaderboard classification มี 15 แถว (13 + baseline 2) |
| **13 จ.** | C2 + C3 | `comparison_a_vs_b.csv`, `feature_importance.csv`, `walk_forward.csv` มีครบทั้งสอง variant |
| **14 อ.** | C4.1–C4.2 เช้า · `finalize --dry-run` + commit lock · **ขอ go-ahead จากเจ้าของ** · C4.3 · เริ่ม notebook 03 | `finalize_receipt.json` มีอยู่ · `pytest -q` เขียว |
| **15 พ.** | C5 (04, 05) + C6 | notebook 3 เล่มรันด้วย nbconvert ผ่าน · ลิงก์เอกสารไม่ขาด |
| **16 พฤ.** | C7 รายงานร่างเต็ม + สไลด์ | ทุกช่องในตาราง traceability ชี้ไฟล์ที่มีจริง |
| **17 ศ.** | **Buffer** — เก็บตกจากวันก่อน · ขัดเกลา · ซ้อมพูด | ไม่มีงานใหม่เข้าวันนี้ |
| **18 ส.** | final verification + ส่ง | `pytest -q` เขียว · deliverable ครบ 4 อย่าง |

### Skill ที่ใช้ในแต่ละ Phase

| Phase | Skill |
| --- | --- |
| ทุก task ที่เขียนโค้ด (C1–C4) | `superpowers:test-driven-development` |
| C1.3, C2, C3 (ทุกครั้งที่เขียนโค้ดวาดรูป) | `dataviz` — อ่านก่อนเลือกสี/ชนิดกราฟ โดยเฉพาะ ROC 13 เส้นซ้อนกัน |
| C3 walk-forward, C5 cell parity | `qqq-model` — ทบทวน interface, `positions`, กับดัก parity |
| ก่อนติ๊กเสร็จทุก Phase และก่อนขอ commit | `superpowers:verification-before-completion` |
| จบ C1, จบ C4.2 (ก่อน STOP gate), จบ C5 | `code-review` (diff ของ Phase นั้น, effort high ที่ C4) |
| C7 รายงาน | `anthropic-skills:docx` |
| C7 สไลด์ | `anthropic-skills:pptx` |

### ความเสี่ยงและการรับมือ

| # | ความเสี่ยง | การรับมือ |
| --- | --- | --- |
| P1 | อ่าน Test ก่อนเวลา หรืออ่านมากกว่าหนึ่งครั้ง | STOP gate ที่ C4.3 · lock ถูก commit ก่อน · `finalize` ปฏิเสธเมื่อมี receipt · guard test ใน `tests/test_audit_guards.py` |
| P2 | ผล Test แย่กว่า Validation มาก หรืออันดับสลับ | คาดไว้แล้ว (E9/E10) — รายงานตามจริงพร้อมคอลัมน์ Validation คู่ Test และ headline ที่ล็อกไว้ก่อน ห้ามเปลี่ยนโมเดลเด่นหลังเห็นผล |
| P3 | ช่วงความเชื่อมั่นกว้างจนแทบทุกโมเดล "เสมอ" | เป็นข้อค้นพบ ไม่ใช่ความล้มเหลว — รายงานว่าแยกกลุ่มได้กี่กลุ่ม และใช้ walk-forward (positive 353 แถว) เป็นหลักฐานเสริม |
| P4 | walk-forward 13 โมเดล × 4 fold × 2 variant ช้าเกิน | มี `--models` เลือกเฉพาะตัว · ลำดับตัดใน C3 · รันค้างไว้ระหว่างทำ C2 |
| P5 | งานเขียนรายงานถูกเบียดโดยงานโค้ด | วันที่ 17 เป็น buffer · C6/C7 ดึงตารางจากไฟล์ generate จึงไม่ต้องพิมพ์ตัวเลขใหม่ |
| P6 | ตัวเลขในเอกสารไม่ตรงไฟล์ (เกิดมาแล้วรอบก่อน) | block `<!-- BEGIN GENERATED -->` + `tests/test_docs_sync.py` (C6.2) |
| P7 | notebook รันไม่จบบนเครื่องผู้ตรวจ | ไม่เทรนใน notebook · path หาจาก `config.py` · ตรวจด้วย `nbconvert --execute` จาก project root |
| P8 | permutation importance ถูกอ่านเกินจริง (ฟีเจอร์สัมพันธ์กัน, positive 34 แถว) | พิมพ์ std คู่ทุกค่า · เขียนข้อจำกัดในตารางและในรายงาน §2.2 |
| P9 | `python-docx` / `python-pptx` ไม่อยู่ใน venv | ใช้เครื่องมือของ skill `docx`/`pptx` นอก venv ของโปรเจกต์ · ไม่แตะ lock file |
| P10 | ยังไม่ได้ยืนยันการลงทะเบียนกลุ่มใน Google Sheet (R10) | งานธุรการ — เจ้าของเช็คภายในวันที่ 12 |

---

## File Structure

| ไฟล์ | สถานะ | ความรับผิดชอบ |
| --- | --- | --- |
| `src/ml/core/artifacts.py` | (ใหม่) | อ่าน run กลับจากดิสก์: ตัวเลข, prediction, โมเดล + preprocessing, SHA-256 |
| `src/ml/core/bootstrap.py` | (ใหม่) | moving-block bootstrap: index, replicate, ช่วง, ผลต่างแบบ paired |
| `src/ml/core/report.py` | (ใหม่) | score, baseline, leaderboard, `build_report()` |
| `src/ml/core/comparison.py` | (ใหม่) | paired A↔B บน Validation ชุดเดียวกัน |
| `src/ml/core/importance.py` | (ใหม่) | block-permutation importance + สัมประสิทธิ์ของโมเดลเชิงเส้น |
| `src/ml/core/walk_forward.py` | (ใหม่) | expanding-window ภายใน Train ด้วย hyperparameter ที่ชนะ |
| `src/ml/finalize.py` | (ใหม่) | selection lock, receipt, ประเมินบน holdout ที่ `run.py` ส่งมา |
| `src/ml/docs_sync.py` | (ใหม่) | คัดลอกตารางที่ generate เข้า block ในเอกสาร |
| `src/ml/core/figures.py` | แก้ | เพิ่ม PR, confusion grid, predicted-vs-actual, residual, performance grid, interval bars, heatmap, Markdown table · แก้การเรียง `None` |
| `src/ml/core/contracts.py` | แก้ | เพิ่ม `SEGMENT_PATHS` |
| `src/ml/core/data.py` | แก้ | เพิ่ม `load_segments()` |
| `src/ml/run.py` | แก้ | `report` จริง, `compare` (ใหม่), `analyze` (ใหม่), `finalize` จริง |
| `tests/test_audit_guards.py` | แก้ที่ C4.2 | ยอมให้ `test_predictions.csv` มีวันที่ช่วง Test **เฉพาะเมื่อมี receipt** |
| `tests/ml/test_run.py` | แก้ | แทน test "report not implemented" |
| `tests/ml/core/test_{artifacts,bootstrap,report,comparison,importance,walk_forward}.py` | (ใหม่) | test ของโมดูลชื่อเดียวกัน |
| `tests/ml/test_finalize.py` | (ใหม่) | lock / receipt / segment โดยใช้ Validation เป็น holdout ปลอม |
| `tests/test_notebooks.py`, `tests/test_docs_sync.py`, `tests/test_docs_links.py` | (ใหม่) | guard ของ notebook และเอกสาร |
| `notebooks/03_regression.ipynb`, `04_classification.ipynb` | เติม (ตอนนี้ 0 cell) | |
| `notebooks/05_results_comparison.ipynb` | (ใหม่) | |
| `docs/report/REPORT_DRAFT.md`, `docs/report/TRACEABILITY.md` | (ใหม่) | ร่างรายงาน + ตารางเกณฑ์ → หลักฐาน |
| `deliverables/report.docx`, `deliverables/slides.pptx` | (ใหม่) | ไฟล์ส่ง |

**ที่อยู่ของผลลัพธ์**

```text
outputs/modeling/report/<variant>/leaderboard_{regression,classification,clustering}.{csv,md}
outputs/modeling/report/<variant>/{feature_importance,linear_coefficients,walk_forward}.csv
outputs/modeling/report/comparison_a_vs_b.{csv,md}
outputs/modeling/report/cards/<model>.md
outputs/figures/modeling/<variant>/*.png        outputs/figures/modeling/comparison_delta.png
outputs/modeling/selection_lock.json            (เขียนโดย finalize --dry-run)
outputs/modeling/finalize_receipt.json          (เขียนโดย finalize --allow-test)
outputs/modeling/<variant>/<task>/<model>/test_metrics.json, test_predictions.csv
outputs/modeling/final/<variant>/leaderboard_{regression,classification}.{csv,md}
outputs/modeling/final/segmented_metrics.csv
```

CSV ใต้ `outputs/modeling/` ที่ไม่ใช่ `test_predictions.csv` **ห้ามมีคอลัมน์แรกเป็นวันที่** เพราะ guard test สแกนด้วย regex `^(\d{4}-\d{2}-\d{2}),`

---

# PHASE C1 — `run.py report` (8 ชม.)

**เกณฑ์รับงานของ Phase**

- `.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike` และ `--variant non_spike` คืน exit 0
- `leaderboard_classification.csv` มี 15 แถว (13 โมเดล + `baseline_majority` + `baseline_persistence`) · `leaderboard_regression.csv` มี 6 แถว (4 + 2) · `leaderboard_clustering.csv` มี 2 แถว
- ตาราง classification มีคอลัมน์ครบตามโจทย์: loss, tn/fp/fn/tp, accuracy, precision, sensitivity, specificity, tnr, f1, roc_auc + `average_precision` พร้อม `_low`/`_high` และ `tied_with_leader`
- แถว `baseline_majority` ได้ accuracy = 337/371, AP = 34/371, ROC-AUC = 0.5
- มีรูป `roc_overlay.png`, `pr_overlay.png`, `confusion_matrices.png`, `performance_curves.png`, `regression_predicted_vs_actual.png`, `regression_residuals.png` ต่อ variant
- ไม่มีไฟล์ใดใน `src/ml` อ้าง Test เพิ่ม · `pytest -q` เขียว

**นิยาม baseline (ไม่มี leakage)**

| ชื่อ | task | กฎ | ข้อมูลที่ใช้ |
| --- | --- | --- | --- |
| `baseline_train_mean` | regression | ทำนายค่าเฉลี่ยของ `target_volatility_5d` ทุกแถว | ค่าเฉลี่ยจาก Train ของ variant นั้นเท่านั้น |
| `baseline_persistence` | regression | ทำนาย `historical_volatility_5d[t]` | เป็น feature ของวัน t (ใช้ return t−4…t รู้ตอนปิดตลาดวัน t) สูตรและหน่วยเดียวกับ target (audit D6) ไม่ fit อะไรเลย |
| `baseline_majority` | classification | ทำนายคลาสข้างมากของ Train ทุกแถว (score คงที่) | สัดส่วนคลาสจาก Train เท่านั้น |
| `baseline_persistence` | classification | score = `historical_volatility_5d[t]`, label = score > Q75 | Q75 = threshold ที่ fit จาก Original Train (`Dataset.threshold`) |

### Task C1.1: `artifacts.py` — อ่าน run กลับมาและพิสูจน์ว่าทำนายซ้ำได้

**Files:**
- Create: `src/ml/core/artifacts.py`
- Test: `tests/ml/core/test_artifacts.py`

**Interfaces:**
- Consumes: `src.ml.core.persist.load_npz(path) -> tuple[dict, dict]` · `src.ml.registry.MODELS`, `model_names()` · `src.ml.core.trainer.evaluate(model, X, y_regression, y_classification, task) -> (metrics, prediction, scores)`
- Produces:
  - `Run` (frozen dataclass): `name, task, variant, directory, config, metrics, dates, target, score, prediction, history`
  - `run_directory(root: Path, variant: str, name: str) -> Path`
  - `read_run(root, variant, name) -> Run`
  - `read_runs(root, variant, *, task: str | None = None, require_all: bool = True) -> list[Run]`
  - `preprocess(X, arrays: dict, metadata: dict) -> np.ndarray`
  - `load_model(directory: Path) -> tuple[BaseModel, Callable[[np.ndarray], np.ndarray]]`
  - `sha256_file(path: Path) -> str` · `HASHED_FILES: tuple[str, ...]`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_artifacts.py
import numpy as np
import pytest

from src.ml.core.artifacts import load_model, read_run, read_runs, run_directory, sha256_file
from src.ml.core.contracts import VARIANTS
from src.ml.core.data import load
from src.ml.core.trainer import OUTPUT_ROOT, evaluate
from src.ml.registry import model_names


def test_run_directory_follows_the_artifact_layout(tmp_path):
    assert run_directory(tmp_path, "with_spike", "knn") == tmp_path / "with_spike" / "classification" / "knn"


def test_read_run_exposes_predictions_as_arrays():
    run = read_run(OUTPUT_ROOT, "with_spike", "knn")
    assert run.task == "classification" and run.config["selected_candidate"] == "knn-07"
    assert run.dates.shape == run.target.shape == run.score.shape == run.prediction.shape == (371,)
    assert str(run.dates[0]) == "2023-08-28"
    assert int(run.target.sum()) == 34


def test_history_is_none_when_a_model_has_no_training_curve():
    assert read_run(OUTPUT_ROOT, "with_spike", "knn").history is None
    assert len(read_run(OUTPUT_ROOT, "with_spike", "adaboost").history["loss"]) == 100


def test_read_runs_filters_by_task():
    assert [run.name for run in read_runs(OUTPUT_ROOT, "with_spike", task="regression")] == [
        "elastic_net", "multiple_linear", "polynomial", "simple_linear"]
    assert len(read_runs(OUTPUT_ROOT, "non_spike", task="classification")) == 13


def test_missing_runs_fail_loudly_unless_explicitly_tolerated(tmp_path):
    with pytest.raises(FileNotFoundError, match="no finished run"):
        read_runs(tmp_path, "with_spike")
    assert read_runs(tmp_path, "with_spike", require_all=False) == []


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", model_names())
def test_every_saved_model_reproduces_its_stored_validation_predictions(variant, name):
    run = read_run(OUTPUT_ROOT, variant, name)
    model, transform = load_model(run.directory)
    validation = load(variant).validation
    _, prediction, score = evaluate(
        model, transform(validation.X), validation.y_regression, validation.y_classification, run.task
    )
    if run.task == "regression":
        np.testing.assert_allclose(prediction, run.prediction, rtol=1e-10, atol=1e-12)
    else:
        np.testing.assert_array_equal(prediction, run.prediction)
    np.testing.assert_allclose(score, run.score, rtol=1e-10, atol=1e-12)


def test_sha256_is_the_hex_digest_of_the_file_bytes(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_artifacts.py -v`
Expected: FAIL ตอน collect — `ModuleNotFoundError: No module named 'src.ml.core.artifacts'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/artifacts.py
"""Read a finished training run back from disk: its numbers, and the model itself."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..registry import MODELS, model_names
from .persist import load_npz

# The files that define what a run would predict; hashing them freezes the selection.
HASHED_FILES = ("config.json", "model.npz", "model.json", "preprocessor.npz", "preprocessor.json")


@dataclass(frozen=True)
class Run:
    name: str
    task: str
    variant: str
    directory: Path
    config: dict
    metrics: dict
    dates: np.ndarray
    target: np.ndarray
    score: np.ndarray
    prediction: np.ndarray
    history: dict | None


def run_directory(root: Path, variant: str, name: str) -> Path:
    return Path(root) / variant / MODELS[name].task / name


def read_run(root: Path, variant: str, name: str) -> Run:
    directory = run_directory(root, variant, name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
    with (directory / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    history_path = directory / "history.json"
    return Run(
        name=name,
        task=config["task"],
        variant=variant,
        directory=directory,
        config=config,
        metrics=metrics,
        dates=np.array([row["Date"] for row in rows], dtype="datetime64[D]"),
        target=np.array([float(row["target"]) for row in rows]),
        score=np.array([float(row["raw_score"]) for row in rows]),
        prediction=np.array([float(row["final_prediction"]) for row in rows]),
        history=json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else None,
    )


def read_runs(root: Path, variant: str, *, task: str | None = None, require_all: bool = True) -> list[Run]:
    names = [name for name in model_names() if task is None or MODELS[name].task == task]
    missing = [
        name for name in names
        if not (run_directory(root, variant, name) / "validation_metrics.json").is_file()
    ]
    if missing and require_all:
        raise FileNotFoundError(f"{variant}: no finished run for {missing}")
    return [read_run(root, variant, name) for name in names if name not in missing]


def preprocess(X, arrays: dict, metadata: dict) -> np.ndarray:
    """The transform the trainer applied before this model saw the data."""
    X = np.array(X, dtype=np.float64, copy=True)
    if metadata["standardized"]:
        X = (X - arrays["mean"]) / arrays["scale"]
        X[:, arrays["constant"].astype(bool)] = 0.0
    if metadata["pca_components"]:
        X = X @ arrays["components"].T
    return X


def load_model(directory: Path):
    """Return (model rebuilt from model.npz, function mapping raw features to its inputs)."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    arrays, metadata = load_npz(directory / "model.npz")
    model = MODELS[config["model"]]().load_state(arrays, metadata)
    scaler_arrays, scaler_metadata = load_npz(directory / "preprocessor.npz")
    return model, lambda X: preprocess(X, scaler_arrays, scaler_metadata)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

หมายเหตุ: `history.json` มีอยู่ใน 14 จาก 19 โมเดลต่อ variant (ตรวจแล้ว 2026-10-11: `knn`, `decision_tree`, `gaussian_nb`, `random_forest`, `stacking` ไม่มี) จึงต้องรองรับ `history=None`

- [ ] **Step 4: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_artifacts.py -v`
Expected: PASS — 44 passed (6 + 38 จาก parametrize 19 โมเดล × 2 variant)

- [ ] **Step 5: Commit (ขออนุมัติเจ้าของก่อน)**

```powershell
git add src/ml/core/artifacts.py tests/ml/core/test_artifacts.py
git commit -m "feat: read finished runs back and prove all 38 reproduce their predictions" -m "Phase C only consumes artifacts, so the reload path has to be proven once for every model before any table is built on it."
```

### Task C1.2: `bootstrap.py` — moving-block bootstrap

**Files:**
- Create: `src/ml/core/bootstrap.py`
- Test: `tests/ml/core/test_bootstrap.py`

**Interfaces:**
- Consumes: NumPy เท่านั้น
- Produces:
  - `BLOCK = 20`, `REPLICATES = 2000`, `SEED = 20261011`, `MIN_VALID_FRACTION = 0.9`
  - `block_indices(n: int, *, block: int = BLOCK, replicates: int = REPLICATES, seed: int = SEED) -> np.ndarray` รูป `(replicates, n)`
  - `replicate(statistic, arrays, indices) -> dict[str, np.ndarray]` โดย `statistic(*arrays) -> dict[str, float | None]`
  - `interval(point, values, *, level: float = 0.95) -> dict` คีย์ `point, low, high, std, n_valid, n_replicates, level`
  - `difference(point_a, values_a, point_b, values_b, *, level: float = 0.95) -> dict` (= b − a) เพิ่มคีย์ `excludes_zero`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_bootstrap.py
import numpy as np
import pytest

from src.ml.core.bootstrap import block_indices, difference, interval, replicate


def test_indices_are_reproducible_chains_of_whole_blocks():
    first = block_indices(371, block=20, replicates=50, seed=7)
    assert first.shape == (50, 371)
    assert first.min() >= 0 and first.max() <= 370
    np.testing.assert_array_equal(first, block_indices(371, block=20, replicates=50, seed=7))
    steps = np.diff(first, axis=1)
    inside_a_block = np.arange(370) % 20 != 19
    assert (steps[:, inside_a_block] == 1).all()


def test_a_different_seed_draws_different_blocks():
    assert not np.array_equal(
        block_indices(100, replicates=5, seed=1), block_indices(100, replicates=5, seed=2)
    )


@pytest.mark.parametrize("n, block", [(0, 1), (10, 11), (10, 0)])
def test_impossible_block_lengths_are_rejected(n, block):
    with pytest.raises(ValueError):
        block_indices(n, block=block)


def test_blocks_widen_the_interval_of_an_autocorrelated_mean():
    rng = np.random.RandomState(0)
    series = np.zeros(400)
    for t in range(1, 400):
        series[t] = 0.9 * series[t - 1] + rng.normal()

    def mean(values):
        return {"mean": float(values.mean())}

    blocked = replicate(mean, (series,), block_indices(400, block=20, replicates=500))["mean"].std()
    rowwise = replicate(mean, (series,), block_indices(400, block=1, replicates=500))["mean"].std()
    assert blocked > 1.5 * rowwise


def test_undefined_replicates_are_counted_not_hidden():
    y = np.r_[np.ones(3), np.zeros(97)]

    def rate(values):
        return {"rate": float(values.mean()) if values.any() else None}

    values = replicate(rate, (y,), block_indices(100, block=10, replicates=200))["rate"]
    summary = interval(0.03, values)
    assert summary["n_replicates"] == 200
    assert summary["n_valid"] == int(np.isfinite(values).sum()) < 180
    assert summary["low"] is None and summary["high"] is None


def test_interval_is_the_percentile_range_of_the_replicates():
    summary = interval(0.5, np.linspace(0.0, 1.0, 1001))
    assert summary["low"] == pytest.approx(0.025) and summary["high"] == pytest.approx(0.975)
    assert summary["n_valid"] == summary["n_replicates"] == 1001


def test_paired_difference_of_identical_scores_is_exactly_zero():
    values = np.random.RandomState(1).rand(300)
    gap = difference(0.4, values, 0.4, values)
    assert gap["point"] == 0.0 and gap["low"] == 0.0 and gap["high"] == 0.0
    assert gap["excludes_zero"] is False


def test_paired_difference_detects_a_consistent_gain():
    base = np.random.RandomState(2).rand(300)
    gap = difference(0.4, base, 0.5, base + 0.1)
    assert gap["excludes_zero"] is True
    assert gap["low"] == pytest.approx(0.1)
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.core.bootstrap'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/bootstrap.py
"""Moving-block bootstrap for autocorrelated daily rows.

The 5-day target overlaps itself (lag-1 autocorrelation 0.92), so resampling single
rows would treat a few hundred dependent days as independent and report intervals that
are far too narrow. Whole blocks of consecutive days are resampled instead.
"""
from __future__ import annotations

import numpy as np

BLOCK = 20
REPLICATES = 2000
SEED = 20261011
# Below this share of usable replicates an interval describes the resampling, not the model.
MIN_VALID_FRACTION = 0.9


def block_indices(n: int, *, block: int = BLOCK, replicates: int = REPLICATES, seed: int = SEED) -> np.ndarray:
    """(replicates, n) row indices; each row of the result is a chain of whole blocks."""
    if n < 1 or not 1 <= block <= n or replicates < 1:
        raise ValueError("Need 1 <= block <= n and at least one replicate")
    rng = np.random.RandomState(seed)
    blocks_needed = -(-n // block)
    starts = rng.randint(0, n - block + 1, size=(replicates, blocks_needed))
    chains = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replicates, -1)
    return chains[:, :n]


def replicate(statistic, arrays, indices) -> dict[str, np.ndarray]:
    """Evaluate statistic(*resampled arrays) -> {name: float | None} on every replicate.

    An undefined value (a resample with no positive day, say) is stored as NaN so the
    caller can count it instead of silently averaging over fewer replicates.
    """
    arrays = [np.asarray(array) for array in arrays]
    names = list(statistic(*arrays))
    values = {name: np.full(len(indices), np.nan) for name in names}
    for row, index in enumerate(indices):
        result = statistic(*(array[index] for array in arrays))
        for name in names:
            if result[name] is not None:
                values[name][row] = result[name]
    return values


def interval(point, values, *, level: float = 0.95) -> dict:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    summary = {"point": point, "low": None, "high": None, "std": None,
               "n_valid": int(len(valid)), "n_replicates": int(len(values)), "level": level}
    if point is None or len(valid) < MIN_VALID_FRACTION * len(values):
        return summary
    tail = (1 - level) / 2
    low, high = np.quantile(valid, [tail, 1 - tail])
    summary.update(low=float(low), high=float(high), std=float(valid.std(ddof=1)))
    return summary


def difference(point_a, values_a, point_b, values_b, *, level: float = 0.95) -> dict:
    """Interval for b - a when both were scored on the same resampled rows."""
    point = None if point_a is None or point_b is None else point_b - point_a
    summary = interval(point, np.asarray(values_b) - np.asarray(values_a), level=level)
    summary["excludes_zero"] = (
        None if summary["low"] is None else bool(summary["low"] > 0 or summary["high"] < 0)
    )
    return summary
```

- [ ] **Step 4: รันให้ผ่าน + ตรวจ static guard**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_bootstrap.py tests/test_audit_guards.py::test_src_never_shuffles_or_splits_at_random -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/bootstrap.py tests/ml/core/test_bootstrap.py
git commit -m "feat: add a moving-block bootstrap for overlapping daily targets" -m "Audit D13: the target's lag-1 autocorrelation is 0.92, so any interval built from row-wise resampling would be several times too narrow."
```

### Task C1.3: `figures.py` — รูปและตาราง Markdown ที่ report ต้องใช้

**Files:**
- Modify: `src/ml/core/figures.py:69-81` (แก้ `write_leaderboard`) และต่อท้ายไฟล์
- Test: `tests/ml/core/test_figures.py` (ต่อท้าย)

**Interfaces:**
- Consumes: `_save(figure, path)` ที่มีอยู่แล้ว (dpi 150, `bbox_inches="tight"`, backend Agg)
- Produces (ทุกตัวเขียนไฟล์แล้วคืน `None`):
  - `plot_pr_curves(curves: dict, path, *, prevalence: float, title: str = "Precision-recall curve")` — `curves: {name: (recall, precision, average_precision)}`
  - `plot_confusion_grid(matrices: dict, path, *, title: str = "Confusion matrices", columns: int = 4)` — `{name: [[tn, fp], [fn, tp]]}`
  - `plot_predicted_vs_actual(series: dict, path, *, title: str = "Predicted vs actual")` — `{name: (actual, predicted)}`
  - `plot_residuals(series: dict, path, *, title: str = "Residuals over time")` — `{name: (dates, residual)}`
  - `plot_performance_grid(histories: dict, path, *, title: str = "Training loss", columns: int = 4)` — `{name: {"iteration": [...], "loss": [...]}}`
  - `plot_interval_bars(rows: list[tuple], path, *, title: str, xlabel: str)` — `[(label, value, low, high)]`, `low`/`high` เป็น `None` ได้
  - `plot_heatmap(matrix, row_labels, column_labels, path, *, title: str, colorbar_label: str)`
  - `write_markdown_table(rows: list[dict], path, *, columns: list[str], title: str, notes: tuple[str, ...] = ())`

ก่อนเขียนโค้ด ให้ invoke skill `dataviz` — โค้ดด้านล่างคือขั้นต่ำที่ผ่าน test และตาม convention เดิมของไฟล์ (ขนาดรูป, `grid(alpha=0.3)`, legend `fontsize="small"`) การปรับสีตาม skill ทำได้โดยไม่เปลี่ยน signature

- [ ] **Step 1: เขียน test ที่ fail ก่อน** (ต่อท้าย `tests/ml/core/test_figures.py`)

```python
import numpy as np

from src.ml.core.figures import (
    plot_confusion_grid,
    plot_heatmap,
    plot_interval_bars,
    plot_performance_grid,
    plot_pr_curves,
    plot_predicted_vs_actual,
    plot_residuals,
    write_markdown_table,
)


def test_leaderboard_puts_undefined_metrics_last_when_descending(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [{"model": "a", "ap": None}, {"model": "b", "ap": 0.2}, {"model": "c", "ap": None},
         {"model": "d", "ap": 0.3}],
        path, sort_by="ap", descending=True,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["d", "b", "a", "c"]


def test_pr_figure_is_written(tmp_path):
    path = tmp_path / "pr.png"
    plot_pr_curves({"m": ([0.0, 0.5, 1.0], [1.0, 0.4, 0.1], 0.3)}, path, prevalence=0.09)
    assert path.stat().st_size > 0


def test_confusion_grid_handles_a_partly_filled_last_row(tmp_path):
    path = tmp_path / "grid.png"
    plot_confusion_grid({f"m{i}": [[300, 20], [15, 9]] for i in range(13)}, path)
    assert path.stat().st_size > 0


def test_regression_figures_are_written(tmp_path):
    actual, predicted = np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.18, 0.25])
    dates = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    plot_predicted_vs_actual({"a": (actual, predicted)}, tmp_path / "pva.png")
    plot_residuals({"a": (dates, actual - predicted)}, tmp_path / "res.png")
    assert (tmp_path / "pva.png").stat().st_size > 0 and (tmp_path / "res.png").stat().st_size > 0


def test_performance_grid_is_written(tmp_path):
    path = tmp_path / "perf.png"
    plot_performance_grid({"a": {"iteration": [1, 2, 3], "loss": [1.0, 0.6, 0.5]}}, path)
    assert path.stat().st_size > 0


def test_interval_bars_accept_rows_without_an_interval(tmp_path):
    path = tmp_path / "delta.png"
    plot_interval_bars([("a", 0.02, -0.01, 0.05), ("b", -0.03, None, None)], path,
                       title="Delta", xlabel="non_spike - with_spike")
    assert path.stat().st_size > 0


def test_heatmap_is_written(tmp_path):
    path = tmp_path / "heat.png"
    plot_heatmap(np.array([[0.1, 0.0], [0.02, 0.3]]), ["m1", "m2"], ["f1", "f2"], path,
                 title="Importance", colorbar_label="drop in AP")
    assert path.stat().st_size > 0


def test_markdown_table_formats_missing_values_and_flags(tmp_path):
    path = tmp_path / "table.md"
    write_markdown_table(
        [{"model": "a", "ap": 0.25481, "tied": True}, {"model": "b", "ap": None, "tied": False}],
        path, columns=["model", "ap", "tied"], title="Leaderboard", notes=("selection score",),
    )
    text = path.read_text(encoding="utf-8")
    assert "### Leaderboard" in text
    assert "| a | 0.2548 | yes |" in text and "| b | n/a | no |" in text
    assert "> selection score" in text


def test_empty_markdown_table_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty table"):
        write_markdown_table([], tmp_path / "x.md", columns=["a"], title="t")
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_figures.py -v`
Expected: FAIL ตอน collect — `ImportError: cannot import name 'plot_confusion_grid'`

- [ ] **Step 3: แก้ `write_leaderboard` ให้ค่า undefined อยู่ท้ายทั้งสองทิศ**

แทนบรรทัด `ordered = sorted(...)` (3 บรรทัด) ใน `src/ml/core/figures.py` ด้วย

```python
    # Undefined metrics go last in both directions; reversing a (is None, value) key
    # would have put them first and compared None with None.
    defined = [row for row in rows if row.get(sort_by) is not None]
    undefined = [row for row in rows if row.get(sort_by) is None]
    ordered = sorted(defined, key=lambda row: row[sort_by], reverse=descending) + undefined
```

- [ ] **Step 4: ต่อท้าย `src/ml/core/figures.py`**

```python
def plot_pr_curves(curves: dict, path: Path, *, prevalence: float, title: str = "Precision-recall curve") -> None:
    """curves: {model_name: (recall, precision, average_precision)}"""
    figure, axes = plt.subplots(figsize=(7, 6))
    for name, (recall, precision, average_precision) in sorted(curves.items()):
        axes.step(recall, precision, where="post", linewidth=1.5,
                  label=f"{name} (AP={average_precision:.3f})")
    axes.axhline(prevalence, color="k", linestyle="--", linewidth=0.8,
                 label=f"positive rate ({prevalence:.3f})")
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_ylim(0, 1.02)
    axes.set_title(title)
    axes.legend(fontsize="small", loc="upper right")
    axes.grid(alpha=0.3)
    _save(figure, path)


def _grid(count: int, columns: int, width: float, height: float):
    rows = -(-count // columns)
    figure, grid = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    for axes in grid.ravel()[count:]:
        axes.axis("off")
    return figure, grid.ravel()[:count]


def plot_confusion_grid(matrices: dict, path: Path, *, title: str = "Confusion matrices", columns: int = 4) -> None:
    """matrices: {model_name: [[tn, fp], [fn, tp]]}"""
    names = sorted(matrices)
    figure, panels = _grid(len(names), columns, 3.2, 3.0)
    for axes, name in zip(panels, names):
        matrix = matrices[name]
        axes.imshow(matrix, cmap="Blues")
        largest = max(max(row) for row in matrix)
        for i in range(2):
            for j in range(2):
                axes.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center",
                          color="white" if matrix[i][j] > largest / 2 else "black")
        axes.set_xticks([0, 1], ["pred 0", "pred 1"])
        axes.set_yticks([0, 1], ["actual 0", "actual 1"])
        axes.set_title(name, fontsize="small")
    figure.suptitle(title)
    _save(figure, path)


def plot_predicted_vs_actual(series: dict, path: Path, *, title: str = "Predicted vs actual") -> None:
    """series: {model_name: (actual, predicted)}"""
    names = sorted(series)
    figure, panels = _grid(len(names), len(names), 4.0, 4.0)
    for axes, name in zip(panels, names):
        actual, predicted = series[name]
        limit = float(max(max(actual), max(predicted)))
        axes.scatter(actual, predicted, s=8, alpha=0.5)
        axes.plot([0, limit], [0, limit], "k--", linewidth=0.8)
        axes.set_xlabel("actual")
        axes.set_ylabel("predicted")
        axes.set_title(name, fontsize="small")
        axes.grid(alpha=0.3)
    figure.suptitle(title)
    _save(figure, path)


def plot_residuals(series: dict, path: Path, *, title: str = "Residuals over time") -> None:
    """series: {model_name: (dates, actual - predicted)}"""
    figure, axes = plt.subplots(figsize=(10, 4.5))
    for name, (dates, residual) in sorted(series.items()):
        axes.plot(dates, residual, linewidth=1.0, label=name)
    axes.axhline(0, color="k", linewidth=0.8)
    axes.set_xlabel("date")
    axes.set_ylabel("actual - predicted")
    axes.set_title(title)
    axes.legend(fontsize="small")
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_performance_grid(histories: dict, path: Path, *, title: str = "Training loss", columns: int = 4) -> None:
    """histories: {model_name: {"iteration": [...], "loss": [...]}}"""
    names = sorted(histories)
    figure, panels = _grid(len(names), min(columns, len(names)), 3.6, 2.8)
    for axes, name in zip(panels, names):
        history = histories[name]
        axes.plot(history["iteration"], history["loss"], linewidth=1.2,
                  marker="o" if len(history["iteration"]) < 20 else None)
        axes.set_title(name, fontsize="small")
        axes.set_xlabel("iteration")
        axes.set_ylabel("loss")
        axes.grid(alpha=0.3)
    figure.suptitle(title)
    _save(figure, path)


def plot_interval_bars(rows: list[tuple], path: Path, *, title: str, xlabel: str) -> None:
    """rows: [(label, value, low, high)]; low/high are None when no interval could be computed."""
    figure, axes = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    for position, (label, value, low, high) in enumerate(rows):
        axes.barh(position, value, color="tab:blue" if low is not None else "lightgray")
        if low is not None:
            axes.plot([low, high], [position, position], color="k", linewidth=1.2)
    axes.axvline(0, color="k", linewidth=0.8)
    axes.set_yticks(range(len(rows)), [row[0] for row in rows])
    axes.invert_yaxis()
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    axes.grid(alpha=0.3, axis="x")
    _save(figure, path)


def plot_heatmap(matrix, row_labels, column_labels, path: Path, *, title: str, colorbar_label: str) -> None:
    figure, axes = plt.subplots(figsize=(1.1 * len(column_labels) + 2.5, 0.4 * len(row_labels) + 2))
    image = axes.imshow(matrix, cmap="Blues", aspect="auto")
    axes.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    axes.set_yticks(range(len(row_labels)), row_labels)
    axes.set_title(title)
    figure.colorbar(image, ax=axes, label=colorbar_label)
    _save(figure, path)


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(
    rows: list[dict], path: Path, *, columns: list[str], title: str, notes: tuple[str, ...] = ()
) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty table")
    lines = [f"### {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |" for row in rows]
    lines += ["", *(f"> {note}" for note in notes)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
```

- [ ] **Step 5: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_figures.py -v`
Expected: PASS — 18 passed (9 เดิม + 9 ใหม่)

- [ ] **Step 6: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/figures.py tests/ml/core/test_figures.py
git commit -m "feat: add the figures and Markdown tables the report needs" -m "Also keeps undefined metrics at the bottom of a descending leaderboard; the old key would have ranked them first."
```

### Task C1.4: `report.py` — score, baseline, leaderboard

**Files:**
- Create: `src/ml/core/report.py`
- Test: `tests/ml/core/test_report.py`

**Interfaces:**
- Consumes: `Run`, `read_runs` (C1.1) · `block_indices`, `replicate`, `interval`, `difference`, `REPLICATES` (C1.2) · figure functions (C1.3) · `metrics.classification_metrics`, `regression_metrics`, `loss` · `data.load(variant) -> Dataset` · `trainer.OUTPUT_ROOT`
- Produces:
  - `CLASSIFICATION_FIELDS`, `REGRESSION_FIELDS`, `INTERVAL_FIELDS: dict[str, tuple]`, `PRIMARY: dict[str, tuple[str, bool]]` (ชื่อ metric หลัก, มากดีกว่าหรือไม่)
  - `classification_scores(y, score, label) -> dict[str, float | None]` · `regression_scores(y, prediction) -> dict[str, float | None]`
  - `baselines(task: str, train: Split, evaluation: Split, threshold: float) -> dict[str, tuple[np.ndarray, np.ndarray]]` (= `{name: (score, final prediction)}`)
  - `as_entry(run: Run) -> dict` คีย์ `model, kind, target, score, prediction`
  - `score_entries(task: str, entries: list[dict], indices: np.ndarray) -> list[dict]` (เพิ่มคีย์ `point`, `replicates`)
  - `leaderboard(task: str, scored: list[dict]) -> list[dict]`
  - `ranked(rows: list[dict], field: str, higher: bool) -> list[dict]` (ดีสุดก่อน, `None` ท้าย)
  - `with_interval(row: dict, name: str) -> str`
  - `build_report(variant: str, *, artifact_root=None, report_root=None, figure_root=None, replicates: int = REPLICATES, require_all: bool = True) -> dict` คีย์ `written: list[Path]`, `tables: dict[str, list[dict]]`
  - `SELECTION_NOTE: str`, `FIGURE_ROOT: Path`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_report.py
import csv
import json
import shutil

import numpy as np
import pytest

from src.ml.core.artifacts import run_directory
from src.ml.core.bootstrap import block_indices
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.data import load
from src.ml.core.report import (
    SELECTION_NOTE,
    baselines,
    build_report,
    classification_scores,
    leaderboard,
    score_entries,
)
from src.ml.core.trainer import OUTPUT_ROOT


def test_regression_baselines_use_only_train_statistics_and_todays_feature():
    data = load("with_spike")
    rules = baselines("regression", data.train, data.validation, data.threshold)
    assert set(rules) == {"baseline_train_mean", "baseline_persistence"}
    assert np.all(rules["baseline_train_mean"][1] == data.train.y_regression.mean())
    column = FEATURE_COLUMNS.index("historical_volatility_5d")
    np.testing.assert_array_equal(rules["baseline_persistence"][1], data.validation.X[:, column])


def test_persistence_baseline_is_identical_for_both_variants():
    a, b = load("with_spike"), load("non_spike")
    first = baselines("classification", a.train, a.validation, a.threshold)["baseline_persistence"]
    second = baselines("classification", b.train, b.validation, b.threshold)["baseline_persistence"]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_majority_baseline_reproduces_the_audited_numbers():
    data = load("with_spike")
    score, label = baselines("classification", data.train, data.validation, data.threshold)["baseline_majority"]
    scores = classification_scores(data.validation.y_classification, score, label)
    assert scores["accuracy"] == pytest.approx(337 / 371)
    assert scores["average_precision"] == pytest.approx(34 / 371)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert scores["recall"] == 0.0 and scores["f1"] == 0.0


def _entry(name, target, score):
    return {"model": name, "kind": "model", "target": target, "score": score,
            "prediction": (score > 0.5).astype(float)}


def test_leaderboard_separates_a_clearly_worse_model_but_not_an_identical_one():
    rng = np.random.RandomState(0)
    target = (rng.rand(400) < 0.3).astype(float)
    strong = target * 0.3 + rng.rand(400) * 0.7
    scored = score_entries(
        "classification",
        [_entry("strong", target, strong), _entry("copy", target, strong.copy()),
         _entry("inverted", target, 1 - strong)],
        block_indices(400, replicates=200),
    )
    rows = {row["model"]: row for row in leaderboard("classification", scored)}
    assert rows["strong"]["tied_with_leader"] is True
    assert rows["copy"]["tied_with_leader"] is True
    assert rows["inverted"]["tied_with_leader"] is False
    assert rows["inverted"]["gap_high"] < 0
    assert rows["strong"]["average_precision_low"] < rows["strong"]["average_precision"] < rows["strong"]["average_precision_high"]


def test_leaderboard_needs_at_least_one_model_row():
    with pytest.raises(ValueError, match="No model rows"):
        leaderboard("classification", [])


def test_build_report_writes_every_table_and_figure(tmp_path):
    result = build_report("with_spike", report_root=tmp_path / "report", figure_root=tmp_path / "fig", replicates=50)
    report, figures = tmp_path / "report" / "with_spike", tmp_path / "fig" / "with_spike"
    rows = list(csv.DictReader((report / "leaderboard_classification.csv").open(encoding="utf-8")))
    assert len(rows) == 15
    for column in ("loss", "tn", "fp", "fn", "tp", "accuracy", "precision", "sensitivity", "specificity",
                   "tnr", "f1", "roc_auc", "average_precision", "average_precision_low",
                   "average_precision_high", "tied_with_leader"):
        assert column in rows[0]
    assert len(list(csv.DictReader((report / "leaderboard_regression.csv").open(encoding="utf-8")))) == 6
    assert len(list(csv.DictReader((report / "leaderboard_clustering.csv").open(encoding="utf-8")))) == 2
    assert SELECTION_NOTE in (report / "leaderboard_classification.md").read_text(encoding="utf-8")
    for name in ("roc_overlay.png", "pr_overlay.png", "confusion_matrices.png", "performance_curves.png",
                 "regression_predicted_vs_actual.png", "regression_residuals.png"):
        assert (figures / name).stat().st_size > 0
    assert all(path.is_file() for path in result["written"])


def test_stale_artifact_is_rejected(tmp_path):
    source = run_directory(OUTPUT_ROOT, "with_spike", "multiple_linear")
    target = run_directory(tmp_path, "with_spike", "multiple_linear")
    shutil.copytree(source, target)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    config["validation_sha256"] = "0" * 64
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="different Validation file"):
        build_report("with_spike", artifact_root=tmp_path, report_root=tmp_path / "r",
                     figure_root=tmp_path / "f", replicates=20, require_all=False)
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.core.report'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/report.py
"""Turn the saved validation runs into leaderboards and figures.

Everything here reads artifacts; nothing is fitted and the held-out split is never
opened. Validation numbers are selection scores (each winner was picked on these same
rows), so every table carries block-bootstrap intervals and says which models cannot
be told apart from the leader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from . import figures
from .artifacts import Run, read_runs
from .bootstrap import REPLICATES, block_indices, difference, interval, replicate
from .contracts import FEATURE_COLUMNS
from .data import load
from .metrics import classification_metrics, loss, regression_metrics
from .trainer import OUTPUT_ROOT

FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "modeling"
CLASSIFICATION_FIELDS = ("average_precision", "roc_auc", "f1", "recall", "precision", "specificity", "accuracy")
REGRESSION_FIELDS = ("rmse", "mae", "r2")
INTERVAL_FIELDS = {"classification": ("average_precision", "roc_auc", "f1", "recall"),
                   "regression": ("rmse", "r2")}
# task -> (metric the models were selected on, whether larger is better)
PRIMARY = {"classification": ("average_precision", True), "regression": ("rmse", False)}
PERSISTENCE_FEATURE = "historical_volatility_5d"
SELECTION_NOTE = (
    "ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ "
    "จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% "
    "(moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น"
)


def classification_scores(y, score, label) -> dict:
    metrics = classification_metrics(y, score, predicted_labels=label)
    return {name: metrics[name]["value"] for name in CLASSIFICATION_FIELDS}


def regression_scores(y, prediction) -> dict:
    metrics = regression_metrics(y, prediction)
    return {name: metrics[name]["value"] for name in REGRESSION_FIELDS}


def baselines(task: str, train, evaluation, threshold: float) -> dict:
    """{name: (score, final prediction)} from rules that need nothing but Train statistics.

    train_mean and majority use the variant's own Train targets. persistence repeats
    the volatility realised over the five days ending today, which is a feature known
    at today's close; for classification it is compared with the Train-fitted Q75.
    """
    rows = len(evaluation.X)
    realised = evaluation.X[:, FEATURE_COLUMNS.index(PERSISTENCE_FEATURE)].copy()
    if task == "regression":
        mean = np.full(rows, train.y_regression.mean())
        return {"baseline_train_mean": (mean, mean), "baseline_persistence": (realised, realised)}
    majority = float(train.y_classification.mean() >= 0.5)
    constant = np.full(rows, majority)
    return {"baseline_majority": (constant, constant),
            "baseline_persistence": (realised, (realised > threshold).astype(np.float64))}


def as_entry(run: Run) -> dict:
    return {"model": run.name, "kind": "model", "target": run.target, "score": run.score,
            "prediction": run.prediction}


def score_entries(task: str, entries: list[dict], indices) -> list[dict]:
    """Add point scores and bootstrap replicates; every entry sees the same resampled rows."""
    scored = []
    for entry in entries:
        if task == "classification":
            arrays, statistic = (entry["target"], entry["score"], entry["prediction"]), classification_scores
        else:
            arrays, statistic = (entry["target"], entry["prediction"]), regression_scores
        scored.append({**entry, "point": statistic(*arrays), "replicates": replicate(statistic, arrays, indices)})
    return scored


def leaderboard(task: str, scored: list[dict]) -> list[dict]:
    field, higher = PRIMARY[task]
    models = [entry for entry in scored if entry["kind"] == "model" and entry["point"][field] is not None]
    if not models:
        raise ValueError("No model rows to rank")
    leader = (max if higher else min)(models, key=lambda entry: entry["point"][field])
    rows = []
    for entry in scored:
        row = {"model": entry["model"], "kind": entry["kind"]}
        for name, value in entry["point"].items():
            row[name] = value
            if name in INTERVAL_FIELDS[task]:
                summary = interval(value, entry["replicates"][name])
                row[f"{name}_low"], row[f"{name}_high"] = summary["low"], summary["high"]
                row[f"{name}_valid_replicates"] = summary["n_valid"]
        gap = difference(leader["point"][field], leader["replicates"][field],
                         entry["point"][field], entry["replicates"][field])
        row["gap_to_leader"], row["gap_low"], row["gap_high"] = gap["point"], gap["low"], gap["high"]
        # Tied unless the paired interval of the gap clears zero.
        row["tied_with_leader"] = bool(entry is leader or gap["excludes_zero"] is not True)
        rows.append(row)
    return rows


def ranked(rows: list[dict], field: str, higher: bool) -> list[dict]:
    """Best first; rows where the metric is undefined go last."""
    defined = [row for row in rows if row[field] is not None]
    undefined = [row for row in rows if row[field] is None]
    return sorted(defined, key=lambda row: row[field], reverse=higher) + undefined


def with_interval(row: dict, name: str) -> str:
    if row.get(name) is None:
        return "n/a"
    if row.get(f"{name}_low") is None:
        return f"{row[name]:.4f} [n/a]"
    return f"{row[name]:.4f} [{row[f'{name}_low']:.4f}, {row[f'{name}_high']:.4f}]"


def _check_current(runs: list[Run], validation) -> None:
    for run in runs:
        if run.config["validation_sha256"] != validation.sha256:
            raise ValueError(f"{run.variant}/{run.name} was scored on a different Validation file; retrain it")
        if not np.array_equal(run.dates, validation.dates.astype("datetime64[D]")):
            raise ValueError(f"{run.variant}/{run.name} predictions do not cover the Validation dates")


def _classification_extras(row: dict, entry: dict, run: Run | None, losses: dict) -> None:
    matrix = classification_metrics(entry["target"], entry["score"],
                                    predicted_labels=entry["prediction"])["confusion_matrix"]
    (tn, fp), (fn, tp) = matrix
    row.update(sensitivity=row["recall"], tnr=row["specificity"], tn=tn, fp=fp, fn=fn, tp=tp)
    if run is None:
        row.update(loss_name=None, loss=None, score_kind=None)
        return
    kind = run.metrics["score_kind"]
    measured = loss(losses[run.name], run.target,
                    scores=run.score if kind == "decision" else None,
                    probabilities=run.score if kind == "probability" else None)
    row.update(loss_name=losses[run.name], loss=measured["value"], score_kind=kind)


def build_report(variant: str, *, artifact_root=None, report_root=None, figure_root=None,
                 replicates: int = REPLICATES, require_all: bool = True) -> dict:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    report_root = Path(report_root or artifact_root / "report") / variant
    figure_root = Path(figure_root or FIGURE_ROOT) / variant
    data = load(variant)
    validation = data.validation
    indices = block_indices(len(validation.X), replicates=replicates)
    losses = json.loads((PROJECT_ROOT / "configs" / "modeling.json").read_text(encoding="utf-8"))["losses"]
    written, tables = [], {}

    for task in ("regression", "classification"):
        runs = read_runs(artifact_root, variant, task=task, require_all=require_all)
        if not runs:
            continue
        _check_current(runs, validation)
        target = (validation.y_regression if task == "regression" else validation.y_classification).astype(np.float64)
        entries = [as_entry(run) for run in runs]
        entries += [{"model": name, "kind": "baseline", "target": target, "score": score, "prediction": prediction}
                    for name, (score, prediction) in baselines(task, data.train, validation, data.threshold).items()]
        rows = leaderboard(task, score_entries(task, entries, indices))
        by_name = {run.name: run for run in runs}
        field, higher = PRIMARY[task]
        for row, entry in zip(rows, entries):
            row["variant"] = variant
            if task == "classification":
                _classification_extras(row, entry, by_name.get(row["model"]), losses)
            else:
                row["mse"] = None if row["rmse"] is None else row["rmse"] ** 2
            for name in INTERVAL_FIELDS[task]:
                row[f"{name}_ci"] = with_interval(row, name)
        tables[task] = rows
        csv_path, md_path = report_root / f"leaderboard_{task}.csv", report_root / f"leaderboard_{task}.md"
        figures.write_leaderboard(rows, csv_path, sort_by=field, descending=higher)
        columns = (["model", "kind", "average_precision_ci", "roc_auc_ci", "f1_ci", "recall_ci", "precision",
                    "specificity", "accuracy", "loss", "tied_with_leader"] if task == "classification"
                   else ["model", "kind", "rmse_ci", "mae", "r2_ci", "tied_with_leader"])
        tied = sum(1 for row in rows if row["kind"] == "model" and row["tied_with_leader"])
        figures.write_markdown_table(
            ranked(rows, field, higher), md_path, columns=columns,
            title=f"Validation leaderboard — {task} — {variant}",
            notes=(SELECTION_NOTE, f"{tied} จาก {len(runs)} โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้"),
        )
        written += [csv_path, md_path]

        if task == "classification":
            defined = [run for run in runs if run.metrics["roc_curve"]["value"] is not None]
            figures.plot_roc_curves(
                {run.name: (run.metrics["roc_curve"]["value"]["fpr"], run.metrics["roc_curve"]["value"]["tpr"],
                            run.metrics["roc_auc"]["value"]) for run in defined},
                figure_root / "roc_overlay.png", title=f"ROC on Validation — {variant}")
            figures.plot_pr_curves(
                {run.name: (run.metrics["pr_curve"]["value"]["recall"], run.metrics["pr_curve"]["value"]["precision"],
                            run.metrics["average_precision"]["value"]) for run in defined},
                figure_root / "pr_overlay.png", prevalence=float(target.mean()),
                title=f"Precision-recall on Validation — {variant}")
            figures.plot_confusion_grid({run.name: run.metrics["confusion_matrix"] for run in runs},
                                        figure_root / "confusion_matrices.png",
                                        title=f"Confusion matrices on Validation — {variant}")
            written += [figure_root / name for name in ("roc_overlay.png", "pr_overlay.png", "confusion_matrices.png")]
        else:
            figures.plot_predicted_vs_actual({run.name: (run.target, run.prediction) for run in runs},
                                             figure_root / "regression_predicted_vs_actual.png",
                                             title=f"Predicted vs actual on Validation — {variant}")
            figures.plot_residuals({run.name: (run.dates, run.target - run.prediction) for run in runs},
                                   figure_root / "regression_residuals.png",
                                   title=f"Residuals on Validation — {variant}")
            written += [figure_root / "regression_predicted_vs_actual.png", figure_root / "regression_residuals.png"]

    everything = read_runs(artifact_root, variant, require_all=require_all)
    curves = {run.name: run.history for run in everything if run.history and len(run.history["loss"]) > 1}
    if curves:
        figures.plot_performance_grid(curves, figure_root / "performance_curves.png",
                                      title=f"Training loss per iteration — {variant}")
        written.append(figure_root / "performance_curves.png")

    clusters = [run for run in everything if run.task == "clustering"]
    if clusters:
        rows = [{"model": run.name, "variant": variant, "k": run.config["params"]["k"],
                 "silhouette": run.metrics["silhouette"]["value"], "inertia": run.metrics["inertia"]["value"],
                 "cluster_sizes": json.dumps(run.metrics["cluster_sizes"], sort_keys=True)} for run in clusters]
        tables["clustering"] = rows
        csv_path, md_path = report_root / "leaderboard_clustering.csv", report_root / "leaderboard_clustering.md"
        figures.write_leaderboard(rows, csv_path, sort_by="silhouette", descending=True)
        figures.write_markdown_table(rows, md_path, columns=list(rows[0]),
                                     title=f"Validation clustering metrics — {variant}",
                                     notes=("silhouette/inertia คำนวณบน Validation ไม่มีช่วงความเชื่อมั่น",))
        written += [csv_path, md_path]
    return {"written": written, "tables": tables}
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_report.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/report.py tests/ml/core/test_report.py
git commit -m "feat: build validation leaderboards with baselines and block-bootstrap intervals" -m "Audit E9/E10: the top classifiers sit 0.016 AP apart while the bootstrap spread is about 0.1, so the table marks which rows cannot be separated from the leader instead of implying a ranking."
```

### Task C1.5: ต่อ `report` เข้า CLI แล้วรันจริง

**Files:**
- Modify: `src/ml/run.py:64-70` (`_report`), `src/ml/run.py:98-101` (parser)
- Modify: `tests/ml/test_run.py` (แทน `test_report_is_honest_about_not_being_implemented`)

**Interfaces:**
- Consumes: `build_report(...)` (C1.4)
- Produces: `python -m src.ml.run report --variant {with_spike,non_spike} [--output-root DIR] [--report-root DIR] [--figure-root DIR] [--replicates N]` — exit 0 พร้อมพิมพ์ path ที่เขียน, exit 1 เมื่อ run ไม่ครบหรือ artifact ล้าสมัย

- [ ] **Step 1: เขียน test ที่ fail ก่อน** — ลบ `test_report_is_honest_about_not_being_implemented` ใน `tests/ml/test_run.py` แล้วใส่แทน

```python
def test_report_fails_cleanly_when_runs_are_missing(tmp_path, capsys):
    assert main(["report", "--variant", "with_spike", "--output-root", str(tmp_path)]) == 1
    assert "no finished run" in capsys.readouterr().err


def test_report_writes_leaderboards_from_the_committed_artifacts(tmp_path, capsys):
    assert main(["report", "--variant", "non_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--replicates", "20"]) == 0
    assert (tmp_path / "r" / "non_spike" / "leaderboard_classification.csv").is_file()
    assert "leaderboard_regression.md" in capsys.readouterr().out
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k report -v`
Expected: FAIL — ทั้งสอง test ได้ `1` กับข้อความ `report is not implemented yet`

- [ ] **Step 3: แทน `_report` และ parser ของมันใน `src/ml/run.py`**

```python
def _report(args) -> int:
    from .core.report import build_report

    try:
        result = build_report(
            args.variant,
            artifact_root=Path(args.output_root) if args.output_root else None,
            report_root=Path(args.report_root) if args.report_root else None,
            figure_root=Path(args.figure_root) if args.figure_root else None,
            replicates=args.replicates,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path in result["written"]:
        print(path)
    return 0
```

```python
    report = sub.add_parser("report", help="build figures and the validation leaderboards")
    report.add_argument("--variant", choices=VARIANTS, required=True)
    report.add_argument("--output-root", help="where the trained runs live")
    report.add_argument("--report-root", help="where tables are written (default <output-root>/report)")
    report.add_argument("--figure-root", help="where figures are written (default outputs/figures/modeling)")
    report.add_argument("--replicates", type=int, default=2000, help="bootstrap replicates")
    report.set_defaults(handler=_report)
```

แก้ docstring บรรทัดบนสุดของ `run.py` ให้ตรงความจริงด้วย: ลบประโยคที่บอกว่า report คืน 1

- [ ] **Step 4: รัน test แล้วรันของจริงทั้งสอง variant**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k "report or finalize" -v`
Expected: PASS — 4 passed (2 ตัวใหม่ + `test_train_reports_reload_status_on_stdout` ที่ชื่อมีคำว่า report + test ของ finalize เดิม)

Run: `.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike; .\.venv\Scripts\python.exe -m src.ml.run report --variant non_spike`
Expected: แต่ละคำสั่งพิมพ์ path 12 บรรทัด (leaderboard 3 ชุด × csv/md = 6 + รูป 6 ไฟล์) และ exit 0

Run: `Get-Content outputs/modeling/report/with_spike/leaderboard_classification.md`
Expected: ตาราง 15 แถว แถว `baseline_majority` มี accuracy `0.9084` และ AP `0.0916` · บรรทัดล่างสุดบอกจำนวนโมเดลที่ `tied_with_leader`

- [ ] **Step 5: ตรวจทั้งชุดและ guard**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: ทั้งหมด passed, 0 failed (เดิม 1,002 + test ใหม่ของ C1) — ถ้า `test_modeling_artifacts_contain_no_test_period_dates` fail แปลว่ามี CSV ใหม่ที่คอลัมน์แรกเป็นวันที่ ให้แก้ลำดับคอลัมน์ ไม่ใช่แก้ test

- [ ] **Step 6: เปิดรูปดูด้วยตา แล้ว Commit (ขออนุมัติก่อน)**

เปิด `outputs/figures/modeling/with_spike/roc_overlay.png` และ `confusion_matrices.png` ตรวจว่า legend ไม่ทับเส้น ตัวเลขอ่านออก ถ้าไม่ผ่านให้ปรับตาม skill `dataviz` ก่อน commit

```powershell
git add src/ml/run.py tests/ml/test_run.py outputs/modeling/report outputs/figures/modeling
git commit -m "feat: implement run.py report and publish the validation leaderboards"
```

---

# PHASE C2 — Paired comparison `with_spike` ↔ `non_spike` (3 ชม.)

**เกณฑ์รับงานของ Phase**

- `.\.venv\Scripts\python.exe -m src.ml.run compare` exit 0 และเขียน `outputs/modeling/report/comparison_a_vs_b.{csv,md}` (19 แถว) + `outputs/figures/modeling/comparison_delta_{classification,regression}.png`
- ทุกแถว regression/classification มี `*_delta`, `*_delta_low`, `*_delta_high`, `*_excludes_zero` ของ metric หลักและ `verdict` ∈ {`non_spike better`, `with_spike better`, `inconclusive`}
- ไฟล์ Markdown ขึ้นต้นด้วยข้อความ confound (ตรวจด้วย test) และไม่มีคำว่า "spike ทำให้" ที่อ้างเหตุเดี่ยว
- ผลต่างคำนวณจาก bootstrap index ชุดเดียวกันทั้งสอง variant (paired) บน Validation 371 แถวเดียวกัน

### Task C2.1: `comparison.py`

**Files:**
- Create: `src/ml/core/comparison.py`
- Test: `tests/ml/core/test_comparison.py`

**Interfaces:**
- Consumes: `read_runs`, `Run` (C1.1) · `block_indices`, `difference`, `REPLICATES` (C1.2) · `as_entry`, `score_entries`, `PRIMARY`, `INTERVAL_FIELDS` (C1.4) · `plot_interval_bars`, `write_leaderboard`, `write_markdown_table` (C1.3)
- Produces:
  - `CONFOUND_NOTE: str`
  - `paired_comparison(*, artifact_root=None, replicates: int = REPLICATES, require_all: bool = True) -> list[dict]`
  - `write_comparison(rows: list[dict], *, report_root: Path, figure_root: Path) -> list[Path]`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_comparison.py
import numpy as np
import pytest

from src.ml.core.artifacts import read_run
from src.ml.core.comparison import CONFOUND_NOTE, paired_comparison, write_comparison
from src.ml.core.trainer import OUTPUT_ROOT


@pytest.fixture(scope="module")
def rows():
    return paired_comparison(replicates=100)


def test_every_model_gets_exactly_one_row(rows):
    assert len(rows) == 19
    assert sum(row["task"] == "classification" for row in rows) == 13


def test_delta_is_non_spike_minus_with_spike(rows):
    row = next(row for row in rows if row["model"] == "knn")
    a = read_run(OUTPUT_ROOT, "with_spike", "knn").metrics["average_precision"]["value"]
    b = read_run(OUTPUT_ROOT, "non_spike", "knn").metrics["average_precision"]["value"]
    assert row["average_precision_with_spike"] == pytest.approx(a)
    assert row["average_precision_delta"] == pytest.approx(b - a)
    assert row["average_precision_delta_low"] <= row["average_precision_delta"] <= row["average_precision_delta_high"]


def test_verdict_is_inconclusive_whenever_the_interval_covers_zero(rows):
    for row in rows:
        if row["task"] == "clustering":
            assert row["verdict"] == "no interval"
        elif row["primary_excludes_zero"] is not True:
            assert row["verdict"] == "inconclusive"
        else:
            assert row["verdict"] in ("non_spike better", "with_spike better")


def test_lower_rmse_counts_as_better(rows):
    for row in rows:
        if row["task"] == "regression" and row["primary_excludes_zero"] is True:
            assert (row["rmse_delta"] < 0) == (row["verdict"] == "non_spike better")


def test_written_markdown_leads_with_the_confound_statement(rows, tmp_path):
    written = write_comparison(rows, report_root=tmp_path / "r", figure_root=tmp_path / "f")
    text = (tmp_path / "r" / "comparison_a_vs_b.md").read_text(encoding="utf-8")
    assert CONFOUND_NOTE in text
    for needle in ("25.0%", "18.4%", "1,733", "1,520"):
        assert needle in CONFOUND_NOTE
    assert all(path.stat().st_size > 0 for path in written)


def test_mismatched_validation_rows_are_rejected(monkeypatch):
    import src.ml.core.comparison as module

    original = module.read_runs

    def shifted(root, variant, **kwargs):
        runs = original(root, variant, **kwargs)
        if variant == "non_spike":
            runs = [type(run)(**{**run.__dict__, "target": np.roll(run.target, 1)}) for run in runs]
        return runs

    monkeypatch.setattr(module, "read_runs", shifted)
    with pytest.raises(ValueError, match="same Validation rows"):
        paired_comparison(replicates=10)
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.core.comparison'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/comparison.py
"""with_spike vs non_spike, scored on the one Validation set both variants share.

The two variants differ in more than the spike rows themselves, so a difference here
is the effect of the whole removal policy and must be worded that way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import figures
from .artifacts import read_runs
from .bootstrap import REPLICATES, block_indices, difference
from .report import INTERVAL_FIELDS, PRIMARY, as_entry, score_entries
from .trainer import OUTPUT_ROOT

CONFOUND_NOTE = (
    "ผลต่างในตารางนี้คือผลของ **นโยบายตัดแถวทั้งชุด** ไม่ใช่ผลของ spike เพียงอย่างเดียว: "
    "การตัด 213 แถวทำให้ class prior ของ Train เปลี่ยนจาก 25.0% เป็น 18.4%, จำนวนแถวลดจาก 1,733 เป็น 1,520 "
    "และ regime ที่ผันผวนสูงหายไปจาก Train พร้อมกัน ทั้งสาม confound นี้แยกจากกันไม่ได้ด้วยการทดลองนี้ "
    "delta = non_spike − with_spike บน Validation 371 แถวชุดเดียวกัน ช่วง 95% จาก paired moving-block bootstrap "
    "(block 20 วัน) และเมื่อเทียบ 17 โมเดลพร้อมกัน ควรคาดว่าจะมีราว 1 แถวที่ช่วงไม่คร่อมศูนย์โดยบังเอิญ"
)


def paired_comparison(*, artifact_root=None, replicates: int = REPLICATES, require_all: bool = True) -> list[dict]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    rows = []
    for task in ("regression", "classification"):
        field, higher = PRIMARY[task]
        first = {run.name: run for run in read_runs(artifact_root, "with_spike", task=task, require_all=require_all)}
        second = {run.name: run for run in read_runs(artifact_root, "non_spike", task=task, require_all=require_all)}
        indices = None
        for name in sorted(set(first) & set(second)):
            a, b = first[name], second[name]
            if not (np.array_equal(a.dates, b.dates) and np.array_equal(a.target, b.target)):
                raise ValueError(f"{name}: the two variants were not scored on the same Validation rows")
            if indices is None:
                indices = block_indices(len(a.target), replicates=replicates)
            scored_a, scored_b = score_entries(task, [as_entry(a), as_entry(b)], indices)
            row = {"model": name, "task": task, "primary_metric": field}
            for metric in INTERVAL_FIELDS[task]:
                gap = difference(scored_a["point"][metric], scored_a["replicates"][metric],
                                 scored_b["point"][metric], scored_b["replicates"][metric])
                row[f"{metric}_with_spike"] = scored_a["point"][metric]
                row[f"{metric}_non_spike"] = scored_b["point"][metric]
                row[f"{metric}_delta"] = gap["point"]
                row[f"{metric}_delta_low"], row[f"{metric}_delta_high"] = gap["low"], gap["high"]
                row[f"{metric}_excludes_zero"] = gap["excludes_zero"]
            row["primary_excludes_zero"] = row[f"{field}_excludes_zero"]
            if row["primary_excludes_zero"] is not True:
                row["verdict"] = "inconclusive"
            else:
                row["verdict"] = "non_spike better" if (row[f"{field}_delta"] > 0) == higher else "with_spike better"
            rows.append(row)

    first = {run.name: run for run in read_runs(artifact_root, "with_spike", task="clustering", require_all=require_all)}
    second = {run.name: run for run in read_runs(artifact_root, "non_spike", task="clustering", require_all=require_all)}
    for name in sorted(set(first) & set(second)):
        a, b = first[name].metrics["silhouette"]["value"], second[name].metrics["silhouette"]["value"]
        rows.append({"model": name, "task": "clustering", "primary_metric": "silhouette",
                     "silhouette_with_spike": a, "silhouette_non_spike": b, "silhouette_delta": b - a,
                     "primary_excludes_zero": None, "verdict": "no interval"})
    return rows


def write_comparison(rows: list[dict], *, report_root: Path, figure_root: Path) -> list[Path]:
    report_root, figure_root = Path(report_root), Path(figure_root)
    csv_path, md_path = report_root / "comparison_a_vs_b.csv", report_root / "comparison_a_vs_b.md"
    figures.write_leaderboard(rows, csv_path, sort_by="model")
    display = []
    for row in rows:
        metric = row["primary_metric"]
        low, high = row.get(f"{metric}_delta_low"), row.get(f"{metric}_delta_high")
        display.append({
            "model": row["model"], "task": row["task"], "metric": metric,
            "with_spike": row[f"{metric}_with_spike"], "non_spike": row[f"{metric}_non_spike"],
            "delta": row[f"{metric}_delta"],
            "delta_95": "n/a" if low is None else f"[{low:+.4f}, {high:+.4f}]",
            "verdict": row["verdict"],
        })
    figures.write_markdown_table(display, md_path, columns=list(display[0]),
                                 title="Paired comparison: non_spike − with_spike (Validation)",
                                 notes=(CONFOUND_NOTE,))
    written = [csv_path, md_path]
    for task, (metric, _) in PRIMARY.items():
        bars = [(row["model"], row[f"{metric}_delta"], row[f"{metric}_delta_low"], row[f"{metric}_delta_high"])
                for row in rows if row["task"] == task]
        path = figure_root / f"comparison_delta_{task}.png"
        figures.plot_interval_bars(bars, path, title=f"non_spike − with_spike ({metric}, Validation)",
                                   xlabel=f"delta in {metric} with 95% paired block-bootstrap interval")
        written.append(path)
    return written
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_comparison.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: เพิ่ม subcommand `compare` (ใหม่) ใน `src/ml/run.py`** — test ก่อน (ต่อท้าย `tests/ml/test_run.py`)

```python
def test_compare_writes_the_paired_table(tmp_path, capsys):
    assert main(["compare", "--report-root", str(tmp_path / "r"), "--figure-root", str(tmp_path / "f"),
                 "--replicates", "20"]) == 0
    assert (tmp_path / "r" / "comparison_a_vs_b.csv").is_file()
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k compare -v` → Expected: FAIL (`SystemExit: 2`, invalid choice `compare`)

```python
def _compare(args) -> int:
    from .core.comparison import paired_comparison, write_comparison
    from .core.report import FIGURE_ROOT
    from .core.trainer import OUTPUT_ROOT

    artifact_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    try:
        rows = paired_comparison(artifact_root=artifact_root, replicates=args.replicates)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path in write_comparison(
        rows,
        report_root=Path(args.report_root) if args.report_root else artifact_root / "report",
        figure_root=Path(args.figure_root) if args.figure_root else FIGURE_ROOT,
    ):
        print(path)
    return 0
```

```python
    compare = sub.add_parser("compare", help="paired with_spike vs non_spike table on Validation")
    compare.add_argument("--output-root")
    compare.add_argument("--report-root")
    compare.add_argument("--figure-root")
    compare.add_argument("--replicates", type=int, default=2000)
    compare.set_defaults(handler=_compare)
```

- [ ] **Step 6: รันของจริงและอ่านผล**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k compare -v; .\.venv\Scripts\python.exe -m src.ml.run compare; Get-Content outputs/modeling/report/comparison_a_vs_b.md`
Expected: test PASS · คำสั่งพิมพ์ 4 path · ตาราง 19 แถว ทิศของ delta ต้องสอดคล้องกับที่บันทึกไว้ใน `docs/tasks/claude-model-progress.md` (regression: RMSE ลดลงทุกตัว · `knn` AP/AUC แย่ลง) ถ้าทิศกลับด้านให้ตรวจลำดับ a/b ก่อนไปต่อ

- [ ] **Step 7: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/comparison.py src/ml/run.py tests/ml/core/test_comparison.py tests/ml/test_run.py outputs/modeling/report outputs/figures/modeling
git commit -m "feat: add the paired with_spike vs non_spike comparison with block-bootstrap deltas" -m "Audit F15: dropping 213 rows also moves the class prior and the regime mix, so the table states that it measures the removal policy as a whole."
```

---

# PHASE C3 — Feature importance และ walk-forward robustness (6 ชม.)

**เกณฑ์รับงานของ Phase**

- `.\.venv\Scripts\python.exe -m src.ml.run analyze --variant <v>` เขียน `feature_importance.csv` (17 โมเดล × 8 ฟีเจอร์ = 136 แถว), `linear_coefficients.csv`, `walk_forward.csv` (13 โมเดล × (4 fold + 1 pooled) = 65 แถว) และรูป heatmap ต่อ variant
- importance ทุกค่ามี `importance_std` คู่ · ฟีเจอร์ที่โมเดลไม่ใช้ (เช่น 7 ฟีเจอร์ที่เหลือของ `simple_linear`) ได้ 0 พอดี
- walk-forward ใช้ hyperparameter จาก `config.json` เท่านั้น · preprocessing fit ใหม่ภายใน fold · แถวเทรนของทุก fold อยู่ก่อน block อย่างน้อย `purge` = 5 วันทำการบน timeline เดิม
- เอกสารผลลัพธ์เขียนชัดว่าเป็น **ตาราง robustness ไม่ใช่การเลือกโมเดลใหม่**

### Task C3.1: `importance.py`

**Files:**
- Create: `src/ml/core/importance.py`
- Test: `tests/ml/core/test_importance.py`

**Interfaces:**
- Consumes: `read_runs`, `load_model` (C1.1) · `data.load` · `trainer.evaluate` · `SEED`, `BLOCK` (C1.2) · `load_npz`
- Produces:
  - `block_shuffled(column, *, block: int, rng: np.random.RandomState) -> np.ndarray`
  - `block_permutation_importance(score, X, *, block: int = BLOCK, repeats: int = 30, seed: int = SEED) -> dict` คีย์ `baseline: float`, `mean: np.ndarray (p,)`, `std: np.ndarray (p,)` — `score(X) -> float` ยิ่งมากยิ่งดี
  - `feature_importance(variant: str, *, artifact_root=None, repeats: int = 30, require_all: bool = True) -> list[dict]`
  - `linear_coefficients(variant: str, *, artifact_root=None) -> list[dict]`
  - `LINEAR_MODELS = ("multiple_linear", "elastic_net", "logistic", "slp", "perceptron", "svm")`

**ระวัง static guard:** ห้ามเขียน `rng.permutation(` — ใช้ `np.argsort(rng.random_sample(k))` แทน และชื่อฟังก์ชัน `block_permutation_importance(` ไม่ชนเพราะหลังคำว่า permutation เป็น `_`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_importance.py
import numpy as np
import pytest

from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.importance import (
    block_permutation_importance,
    block_shuffled,
    feature_importance,
    linear_coefficients,
)


def test_block_shuffle_moves_whole_blocks_and_keeps_every_value():
    column = np.arange(100.0)
    shuffled = block_shuffled(column, block=20, rng=np.random.RandomState(3))
    assert sorted(shuffled) == sorted(column)
    assert not np.array_equal(shuffled, column)
    assert (np.diff(shuffled.reshape(5, 20), axis=1) == 1).all()


def test_only_the_feature_the_score_depends_on_is_important():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    X.setflags(write=False)
    y = X[:, 0].copy()

    def score(matrix):
        return -float(np.mean((matrix[:, 0] - y) ** 2))

    result = block_permutation_importance(score, X, block=20, repeats=5, seed=1)
    assert result["baseline"] == 0.0
    assert result["mean"][0] > 0.5
    assert result["mean"][1] == 0.0 and result["mean"][2] == 0.0
    assert result["std"].shape == (3,)


def test_importance_is_reproducible_for_a_fixed_seed():
    X = np.random.RandomState(0).normal(size=(120, 2))

    def score(matrix):
        return -float(np.abs(matrix[:, 1] - X[:, 1]).mean())

    first = block_permutation_importance(score, X, repeats=4, seed=9)
    second = block_permutation_importance(score, X, repeats=4, seed=9)
    np.testing.assert_array_equal(first["mean"], second["mean"])


@pytest.fixture(scope="module")
def importance_rows():
    return feature_importance("with_spike", repeats=3)


def test_real_importance_covers_every_supervised_model_and_feature(importance_rows):
    assert len(importance_rows) == 17 * 8
    assert {row["feature"] for row in importance_rows} == set(FEATURE_COLUMNS)
    assert all(row["metric"] in ("average_precision", "negative_rmse") for row in importance_rows)


def test_a_single_feature_model_ignores_the_other_seven(importance_rows):
    rows = [row for row in importance_rows if row["model"] == "simple_linear"]
    unused = [row for row in rows if row["feature"] != "historical_volatility_20d"]
    assert all(row["importance_mean"] == 0.0 for row in unused)


def test_linear_coefficients_are_reported_per_original_feature():
    rows = linear_coefficients("with_spike")
    assert len(rows) == 6 * 8
    svm = [row for row in rows if row["model"] == "svm"]
    assert [row["feature"] for row in svm] == list(FEATURE_COLUMNS)
    assert all(np.isfinite(row["coefficient"]) for row in rows)
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_importance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.core.importance'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/importance.py
"""Model-agnostic feature importance on Validation, plus linear-model coefficients.

A feature column is reordered in whole blocks of consecutive days rather than row by
row: the features are strongly autocorrelated, and a row-wise reordering would also
destroy that serial structure and overstate how much the model relies on the feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import load_model, read_runs, run_directory
from .bootstrap import BLOCK, SEED
from .contracts import FEATURE_COLUMNS
from .data import load
from .persist import load_npz
from .trainer import OUTPUT_ROOT, evaluate

LINEAR_MODELS = ("multiple_linear", "elastic_net", "logistic", "slp", "perceptron", "svm")


def block_shuffled(column, *, block: int, rng) -> np.ndarray:
    """Reorder whole blocks of consecutive rows, keeping each block's inner order."""
    column = np.asarray(column)
    pieces = np.array_split(np.arange(len(column)), max(1, -(-len(column) // block)))
    order = np.argsort(rng.random_sample(len(pieces)), kind="stable")
    return column[np.concatenate([pieces[index] for index in order])]


def block_permutation_importance(score, X, *, block: int = BLOCK, repeats: int = 30, seed: int = SEED) -> dict:
    """Drop in score(X) when one column is block-reordered; larger means more important."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.RandomState(seed)
    baseline = float(score(X))
    drops = np.empty((repeats, X.shape[1]))
    for repeat in range(repeats):
        for feature in range(X.shape[1]):
            shuffled = X.copy()
            shuffled[:, feature] = block_shuffled(X[:, feature], block=block, rng=rng)
            drops[repeat, feature] = baseline - float(score(shuffled))
    return {"baseline": baseline, "mean": drops.mean(axis=0),
            "std": drops.std(axis=0, ddof=1) if repeats > 1 else np.zeros(X.shape[1])}


def feature_importance(variant: str, *, artifact_root=None, repeats: int = 30, require_all: bool = True) -> list[dict]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    validation = load(variant).validation
    rows = []
    for run in read_runs(artifact_root, variant, require_all=require_all):
        if run.task == "clustering":
            continue
        model, transform = load_model(run.directory)

        def score(matrix, model=model, transform=transform, task=run.task):
            metrics, _, _ = evaluate(model, transform(matrix), validation.y_regression,
                                     validation.y_classification, task)
            if task == "regression":
                return -metrics["clipped_metrics"]["rmse"]["value"]
            return metrics["average_precision"]["value"]

        result = block_permutation_importance(score, validation.X, repeats=repeats)
        metric = "negative_rmse" if run.task == "regression" else "average_precision"
        for index, feature in enumerate(FEATURE_COLUMNS):
            rows.append({"model": run.name, "variant": variant, "task": run.task, "feature": feature,
                         "metric": metric, "baseline": result["baseline"],
                         "importance_mean": float(result["mean"][index]),
                         "importance_std": float(result["std"][index])})
    return rows


def linear_coefficients(variant: str, *, artifact_root=None) -> list[dict]:
    """Weights on the standardised features; svm's are mapped back from its PCA basis."""
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    rows = []
    for name in LINEAR_MODELS:
        directory = run_directory(artifact_root, variant, name)
        arrays, _ = load_npz(directory / "model.npz")
        scaler, metadata = load_npz(directory / "preprocessor.npz")
        coefficient = np.asarray(arrays["coef"], dtype=np.float64)
        if metadata["pca_components"]:
            coefficient = scaler["components"].T @ coefficient
        for feature, value in zip(FEATURE_COLUMNS, coefficient):
            rows.append({"model": name, "variant": variant, "feature": feature, "coefficient": float(value)})
    return rows
```

- [ ] **Step 4: รันให้ผ่าน + guard**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_importance.py tests/test_audit_guards.py::test_src_never_shuffles_or_splits_at_random -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/importance.py tests/ml/core/test_importance.py
git commit -m "feat: add block-permutation feature importance and linear coefficients" -m "The brief asks for feature importance and insight; only target correlations existed. Blocks are reordered, not rows, so the measure respects the series' autocorrelation."
```

### Task C3.2: `walk_forward.py`

**Files:**
- Create: `src/ml/core/walk_forward.py`
- Test: `tests/ml/core/test_walk_forward.py`

**Interfaces:**
- Consumes: `src.ml.classification.stacking.fold_plan(positions, n_positions, folds, initial_fraction, purge) -> list[(train_mask, block_mask, start, stop)]` · `trainer.trading_day_positions(variant, dates) -> (positions, n_positions)` · `trainer.UNSCALED_MODES`, `evaluate` · `preprocess.Standardizer`, `PCA` · `classification_scores` (C1.4) · `block_indices`, `replicate`, `interval` (C1.2) · `run_directory` (C1.1)
- Produces:
  - `walk_forward(name: str, *, variant: str, artifact_root=None, model_class=None, folds: int = 4, initial_fraction: float = 0.4, purge: int = 5, replicates: int = REPLICATES) -> list[dict]` — 4 แถว fold + 1 แถว `fold == "pooled"` (มี `average_precision_low/high`)
  - `walk_forward_table(variant: str, *, artifact_root=None, models: list[str] | None = None, replicates: int = REPLICATES) -> list[dict]`
  - `spearman(a, b) -> float`

**รายละเอียดที่พลาดง่าย:** โมเดลที่ `fit` รับ `positions` (`slp`, `mlp`, `stacking`) ต้องได้ `positions[train]` และ `n_positions = positions[train][-1] + 1` — ถ้าส่ง `n_positions` ของ timeline เต็ม `Stacking.fit` จะสร้าง fold ภายในที่ไม่มีแถวแล้ว raise `Fold for positions [...) has no rows to predict`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_walk_forward.py
import json

import numpy as np
import pytest

from src.ml.core.artifacts import run_directory
from src.ml.core.base import BaseModel
from src.ml.core.trainer import OUTPUT_ROOT
from src.ml.core.walk_forward import spearman, walk_forward
from src.ml.registry import MODELS, models_for_task


class Recorder(BaseModel):
    """Stands in for a model that asks for trading-day positions."""

    name = "gaussian_nb"
    task = "classification"
    calls: list = []

    def get_params(self):
        return {}

    def fit(self, X, y=None, positions=None, n_positions=None):
        type(self).calls.append({"rows": len(X), "last": int(positions[-1]), "n_positions": n_positions,
                                 "mean": np.asarray(X).mean(axis=0)})
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate_)

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)

    def state_dict(self):
        return {}

    def load_state(self, arrays, metadata):
        return self


@pytest.fixture()
def recorded():
    Recorder.calls = []
    rows = walk_forward("gaussian_nb", variant="with_spike", model_class=Recorder, replicates=20)
    return rows, Recorder.calls


def test_four_folds_and_one_pooled_row(recorded):
    rows, _ = recorded
    assert [row["fold"] for row in rows] == [0, 1, 2, 3, "pooled"]
    assert rows[-1]["block_rows"] == sum(row["block_rows"] for row in rows[:4])
    assert rows[-1]["positives"] == sum(row["positives"] for row in rows[:4])


def test_training_history_grows_and_stops_before_the_purge_gap(recorded):
    rows, calls = recorded
    assert [call["rows"] for call in calls] == sorted(call["rows"] for call in calls)
    for row, call in zip(rows[:4], calls):
        assert call["last"] < row["block_start_position"] - 5
        assert call["n_positions"] == call["last"] + 1


def test_blocks_are_consecutive_and_cover_the_last_sixty_percent(recorded):
    rows, _ = recorded
    assert rows[0]["block_start_position"] == int(1733 * 0.4)
    for earlier, later in zip(rows[:3], rows[1:4]):
        assert earlier["block_stop_position"] == later["block_start_position"]
    assert rows[3]["block_stop_position"] == 1733


def test_non_spike_counts_days_on_the_original_timeline():
    Recorder.calls = []
    rows = walk_forward("gaussian_nb", variant="non_spike", model_class=Recorder, replicates=20)
    assert rows[0]["block_start_position"] == int(1733 * 0.4)
    assert rows[-1]["block_rows"] < 1040


def test_real_model_returns_metrics_with_a_pooled_interval():
    rows = walk_forward("gaussian_nb", variant="with_spike", replicates=50)
    pooled = rows[-1]
    assert 0 < pooled["average_precision"] <= 1
    assert pooled["average_precision_low"] <= pooled["average_precision"] <= pooled["average_precision_high"]
    assert pooled["positives"] > 300


@pytest.mark.parametrize("name", models_for_task("classification"))
def test_saved_winner_params_rebuild_every_classifier(name):
    config = json.loads((run_directory(OUTPUT_ROOT, "with_spike", name) / "config.json").read_text(encoding="utf-8"))
    MODELS[name](**config["params"])


def test_spearman_of_reversed_ranks_is_minus_one():
    assert spearman([0.1, 0.2, 0.3, 0.4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([0.1, 0.2, 0.3], [10, 20, 30]) == pytest.approx(1.0)
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_walk_forward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.core.walk_forward'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/core/walk_forward.py
"""Expanding-window check inside Train, with each model's winning hyperparameters fixed.

This is a robustness table, not a second model selection: Validation is one calm
regime with 34 positive days, while these four blocks span 2019-2023 and hold about
ten times as many. Nothing chosen here feeds back into which model or setting is used.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from ..classification.stacking import fold_plan
from ..registry import MODELS, models_for_task
from .artifacts import run_directory
from .bootstrap import REPLICATES, block_indices, interval, replicate
from .data import load
from .preprocess import PCA, Standardizer
from .report import classification_scores
from .trainer import OUTPUT_ROOT, UNSCALED_MODES, evaluate, trading_day_positions

FIELDS = ("average_precision", "roc_auc", "f1", "recall")


def spearman(a, b) -> float:
    """Rank correlation without tie handling; the inputs here are distinct floats."""
    rank_a = np.argsort(np.argsort(np.asarray(a, dtype=np.float64)))
    rank_b = np.argsort(np.argsort(np.asarray(b, dtype=np.float64)))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def walk_forward(name: str, *, variant: str, artifact_root=None, model_class=None, folds: int = 4,
                 initial_fraction: float = 0.4, purge: int = 5, replicates: int = REPLICATES) -> list[dict]:
    directory = run_directory(Path(artifact_root or OUTPUT_ROOT), variant, name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model_class = model_class or MODELS[name]
    params = config["params"] if model_class is MODELS[name] else {}
    train = load(variant).train
    positions, n_positions = trading_day_positions(variant, train.dates)
    scaled = config["preprocess"] not in UNSCALED_MODES or bool(config["pca_components"])
    wants_positions = "positions" in inspect.signature(model_class.fit).parameters

    rows, pooled = [], {"y": [], "score": [], "label": []}
    for fold, (history, block, start, stop) in enumerate(
        fold_plan(positions, n_positions, folds, initial_fraction, purge)
    ):
        history_X, block_X = train.X[history], train.X[block]
        if scaled:
            scaler = Standardizer().fit(history_X)
            history_X, block_X = scaler.transform(history_X), scaler.transform(block_X)
        if config["pca_components"]:
            pca = PCA(config["pca_components"]).fit(history_X)
            history_X, block_X = pca.transform(history_X), pca.transform(block_X)
        extras = {}
        if wants_positions:
            # The model's own inner folds must end where this fold's history ends.
            extras = {"positions": positions[history], "n_positions": int(positions[history][-1]) + 1}
        model = model_class(**params).fit(history_X, train.y_classification[history], **extras)
        _, label, score = evaluate(model, block_X, train.y_regression[block],
                                   train.y_classification[block], "classification")
        y = train.y_classification[block].astype(np.float64)
        label, score = np.asarray(label, dtype=np.float64), np.asarray(score, dtype=np.float64)
        scores = classification_scores(y, score, label)
        rows.append({"model": name, "variant": variant, "fold": fold,
                     "block_start": str(train.dates[block][0])[:10], "block_end": str(train.dates[block][-1])[:10],
                     "block_start_position": start, "block_stop_position": stop,
                     "train_rows": int(history.sum()), "block_rows": int(block.sum()),
                     "positives": int(y.sum()), **{field: scores[field] for field in FIELDS}})
        for key, values in (("y", y), ("score", score), ("label", label)):
            pooled[key].append(values)

    y, score, label = (np.concatenate(pooled[key]) for key in ("y", "score", "label"))
    point = classification_scores(y, score, label)
    summary = interval(point["average_precision"], replicate(
        classification_scores, (y, score, label), block_indices(len(y), replicates=replicates)
    )["average_precision"])
    rows.append({"model": name, "variant": variant, "fold": "pooled",
                 "block_start": rows[0]["block_start"], "block_end": rows[-1]["block_end"],
                 "block_start_position": rows[0]["block_start_position"],
                 "block_stop_position": rows[-1]["block_stop_position"],
                 "train_rows": None, "block_rows": int(len(y)), "positives": int(y.sum()),
                 **{field: point[field] for field in FIELDS},
                 "average_precision_low": summary["low"], "average_precision_high": summary["high"]})
    return rows


def walk_forward_table(variant: str, *, artifact_root=None, models: list[str] | None = None,
                       replicates: int = REPLICATES) -> list[dict]:
    rows = []
    for name in models or models_for_task("classification"):
        rows += walk_forward(name, variant=variant, artifact_root=artifact_root, replicates=replicates)
    return rows
```

หมายเหตุสำหรับ `non_spike`: แถวใน block ไม่ใช่วันติดกันทุกแถว block bootstrap ของ pooled จึงนับ block ตามลำดับแถว ไม่ใช่ตามวันปฏิทิน — ต้องเขียนข้อจำกัดนี้ใต้ตารางในรายงาน

- [ ] **Step 4: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_walk_forward.py -v`
Expected: PASS — 19 passed (6 + 13 จาก parametrize)

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/walk_forward.py tests/ml/core/test_walk_forward.py
git commit -m "feat: add an expanding-window robustness check inside Train" -m "Audit E10: Validation is a single calm regime with 34 positives. The purged folds already used by stacking give four blocks with roughly ten times as many, at fixed hyperparameters."
```

### Task C3.3: subcommand `analyze` (ใหม่) และรันจริง

**Files:**
- Modify: `src/ml/run.py` (เพิ่ม `_analyze` + parser)
- Modify: `tests/ml/test_run.py`

**Interfaces:**
- Consumes: `feature_importance`, `linear_coefficients` (C3.1) · `walk_forward_table`, `spearman` (C3.2) · `write_leaderboard`, `plot_heatmap` (C1.3)
- Produces: `python -m src.ml.run analyze --variant V [--output-root DIR] [--report-root DIR] [--figure-root DIR] [--repeats N] [--replicates N] [--models a,b] [--skip-walk-forward]`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
def test_analyze_writes_importance_and_coefficients(tmp_path):
    assert main(["analyze", "--variant", "with_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--repeats", "2", "--skip-walk-forward"]) == 0
    report = tmp_path / "r" / "with_spike"
    assert (report / "feature_importance.csv").is_file() and (report / "linear_coefficients.csv").is_file()
    assert (tmp_path / "f" / "with_spike" / "feature_importance.png").stat().st_size > 0
    assert not (report / "walk_forward.csv").exists()


def test_analyze_can_walk_forward_a_chosen_model(tmp_path):
    assert main(["analyze", "--variant", "with_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--repeats", "2", "--replicates", "20",
                 "--models", "gaussian_nb"]) == 0
    lines = (tmp_path / "r" / "with_spike" / "walk_forward.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k analyze -v` → Expected: FAIL (`SystemExit: 2`)

- [ ] **Step 2: เพิ่มโค้ดใน `src/ml/run.py`**

```python
def _analyze(args) -> int:
    import numpy as np

    from .core import figures
    from .core.contracts import FEATURE_COLUMNS
    from .core.importance import feature_importance, linear_coefficients
    from .core.report import FIGURE_ROOT
    from .core.trainer import OUTPUT_ROOT
    from .core.walk_forward import walk_forward_table

    artifact_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    report = (Path(args.report_root) if args.report_root else artifact_root / "report") / args.variant
    figure = (Path(args.figure_root) if args.figure_root else FIGURE_ROOT) / args.variant

    importance = feature_importance(args.variant, artifact_root=artifact_root, repeats=args.repeats)
    figures.write_leaderboard(importance, report / "feature_importance.csv", sort_by="model")
    names = sorted({row["model"] for row in importance if row["task"] == "classification"})
    matrix = np.array([[next(row["importance_mean"] for row in importance
                             if row["model"] == name and row["feature"] == feature)
                        for feature in FEATURE_COLUMNS] for name in names])
    figures.plot_heatmap(matrix, names, list(FEATURE_COLUMNS), figure / "feature_importance.png",
                         title=f"Block-permutation importance on Validation — {args.variant}",
                         colorbar_label="drop in average precision")
    figures.write_leaderboard(linear_coefficients(args.variant, artifact_root=artifact_root),
                              report / "linear_coefficients.csv", sort_by="model")
    written = [report / "feature_importance.csv", figure / "feature_importance.png",
               report / "linear_coefficients.csv"]
    if not args.skip_walk_forward:
        rows = walk_forward_table(args.variant, artifact_root=artifact_root,
                                  models=args.models.split(",") if args.models else None,
                                  replicates=args.replicates)
        figures.write_leaderboard(rows, report / "walk_forward.csv", sort_by="model")
        written.append(report / "walk_forward.csv")
    for path in written:
        print(path)
    return 0
```

```python
    analyze = sub.add_parser("analyze", help="feature importance and walk-forward robustness")
    analyze.add_argument("--variant", choices=VARIANTS, required=True)
    analyze.add_argument("--output-root")
    analyze.add_argument("--report-root")
    analyze.add_argument("--figure-root")
    analyze.add_argument("--repeats", type=int, default=30, help="reorderings per feature")
    analyze.add_argument("--replicates", type=int, default=2000, help="bootstrap replicates")
    analyze.add_argument("--models", help="comma-separated classifiers for the walk-forward table")
    analyze.add_argument("--skip-walk-forward", action="store_true")
    analyze.set_defaults(handler=_analyze)
```

- [ ] **Step 3: รัน test แล้วรันของจริง (จับเวลา)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k analyze -v`
Expected: PASS — 2 passed

Run: `Measure-Command { .\.venv\Scripts\python.exe -m src.ml.run analyze --variant with_spike | Out-Default }`
Expected: พิมพ์ 4 path · ถ้าเกิน 30 นาทีให้หยุดแล้วรันใหม่ด้วย `--models` ทีละกลุ่มตามลำดับตัดของ Phase (ตัด `mlp`, `slp`, `stacking` ก่อน) และบันทึกในรายงานว่าตารางไม่ครบ 13 ตัวเพราะอะไร

Run: `.\.venv\Scripts\python.exe -m src.ml.run analyze --variant non_spike`
Expected: 4 path เช่นกัน

- [ ] **Step 4: ตรวจผลด้วยตา**

Run: `.\.venv\Scripts\python.exe -c "import csv; rows=[r for r in csv.DictReader(open('outputs/modeling/report/with_spike/walk_forward.csv', encoding='utf-8')) if r['fold']=='pooled']; print(len(rows)); [print(r['model'], r['positives'], r['average_precision'][:6], r['average_precision_low'][:6], r['average_precision_high'][:6]) for r in rows]"`
Expected: 13 บรรทัด · `positives` เท่ากันทุกโมเดล (ราว 353 ตาม audit) · AP ทุกตัวสูงกว่า prevalence ของ block

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/run.py tests/ml/test_run.py outputs/modeling/report outputs/figures/modeling
git commit -m "feat: add run.py analyze for feature importance and walk-forward robustness"
```

---

# PHASE C4 — `run.py finalize --allow-test` (5 ชม. + ⛔ STOP gate)

**เกณฑ์รับงานของ Phase**

- ก่อน gate: `finalize --dry-run` เขียน `outputs/modeling/selection_lock.json` (38 runs + `headline` ต่อ variant/task) โดย **ไม่เปิดไฟล์ Test** และ lock ถูก commit แล้ว
- หลัง gate: `finalize --allow-test` รัน **ครั้งเดียว** เขียน `test_metrics.json` + `test_predictions.csv` ครบ 38 โฟลเดอร์, `final/<variant>/leaderboard_{regression,classification}.{csv,md}`, `final/segmented_metrics.csv`, `finalize_receipt.json`
- `test_metrics.json` ของ regression/classification มีคีย์ `segments` → `spike_affected` (54 แถว) และ `non_spike` (319 แถว)
- รันซ้ำโดยไม่มี `--rerun-reason` → exit 1 **ก่อน** อ่าน Test · artifact ที่ hash ต่างจาก lock → exit 1 ก่อนอ่าน Test
- `pytest -q` เขียว รวม guard ที่แก้แล้ว

**การแก้ `tests/test_audit_guards.py` (ต้องทำใน Task C4.2 ก่อนรัน finalize จริง)**

| Test | ทำอะไร | เหตุผล |
| --- | --- | --- |
| `test_only_the_data_module_and_finalize_reference_the_test_split` | **ไม่แก้** `TEST_SPLIT_ALLOWED` ยังเป็น `{"core/data.py", "core/contracts.py", "run.py"}` | `src/ml/finalize.py` รับ `Split` ที่ `run.py` โหลดมา จึงไม่มีสตริงต้องห้าม — ถ้า test นี้ fail แปลว่ามีสตริงหลุดเข้า `finalize.py` ให้แก้โค้ด ห้ามเพิ่มไฟล์ใน allow-list |
| `test_modeling_artifacts_contain_no_test_period_dates` | **แก้** เป็น `test_only_finalize_outputs_contain_test_period_dates` ตามโค้ดใน C4.2 Step 5 | หลัง finalize `test_predictions.csv` 38 ไฟล์จะมีวันที่ช่วง Test โดยชอบธรรม — ยอมเฉพาะชื่อไฟล์นี้ และเฉพาะเมื่อมี `finalize_receipt.json` |
| (ใหม่) `test_finalize_outputs_exist_only_with_a_receipt` | **เพิ่ม** | `test_metrics.json` ที่โผล่มาโดยไม่มี receipt = มีคนอ่าน Test นอกขั้นตอน |
| `test_src_never_shuffles_or_splits_at_random` | ไม่แก้ | โมดูลใหม่ทุกตัวหลีกเลี่ยงสตริงต้องห้ามแล้ว |

### Task C4.1: `load_segments()` — อ่าน `diagnostic_segment`

**Files:**
- Modify: `src/ml/core/contracts.py` (ต่อท้าย)
- Modify: `src/ml/core/data.py` (ต่อท้าย + import)
- Test: `tests/ml/core/test_data.py` (ต่อท้าย)

**Interfaces:**
- Produces:
  - `contracts.SEGMENT_PATHS: dict[str, Path]` คีย์ `"validation"`, `"test"`
  - `contracts.SEGMENT_COLUMN = "diagnostic_segment"`
  - `data.load_segments(split: str, *, allow_test: bool = False) -> tuple[np.ndarray, np.ndarray]` (= `(dates datetime64, segment str)`, read-only, ตรวจ SHA-256 กับ `experiment_dataset_report.json` คีย์ `<split>_flagged`)

- [ ] **Step 1: เขียน test ที่ fail ก่อน** (ไม่มี test ใดอ่าน segment ของ Test)

```python
from src.ml.core.data import load_segments


def test_validation_segments_align_with_the_validation_split():
    dates, segments = load_segments("validation")
    np.testing.assert_array_equal(dates, load("with_spike").validation.dates)
    assert segments.shape == (371,)
    assert set(segments) == {"non_spike"}
    assert not segments.flags.writeable


def test_test_segments_are_denied_by_default():
    with pytest.raises(PermissionError, match="finalize"):
        load_segments("test")
    with pytest.raises(PermissionError):
        load_segments("test", allow_test=1)


def test_unknown_segment_split_is_rejected():
    with pytest.raises(ValueError, match="Unknown split"):
        load_segments("train")
```

(ถ้า `tests/ml/core/test_data.py` ยังไม่ import `numpy as np` ให้เพิ่ม)

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_data.py -k segment -v`
Expected: FAIL — `ImportError: cannot import name 'load_segments'`

- [ ] **Step 3: implementation**

ต่อท้าย `src/ml/core/contracts.py` (เพิ่ม `TEST_FLAGGED_PATH`, `VALIDATION_FLAGGED_PATH` ใน import จาก `config`)

```python
SEGMENT_COLUMN = "diagnostic_segment"
SEGMENT_PATHS = {"validation": VALIDATION_FLAGGED_PATH, "test": TEST_FLAGGED_PATH}
```

ต่อท้าย `src/ml/core/data.py` (เพิ่ม `SEGMENT_COLUMN`, `SEGMENT_PATHS` ใน import จาก `.contracts`)

```python
def load_segments(split: str, *, allow_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(dates, diagnostic_segment) of an evaluation split; Test stays default-deny."""
    if split not in SEGMENT_PATHS:
        raise ValueError(f"Unknown split: {split!r}; expected one of {sorted(SEGMENT_PATHS)}")
    if split == "test" and allow_test is not True:
        raise PermissionError(
            "Test segments are only available during finalize; "
            "pass allow_test=True from `src.ml.run finalize --allow-test`"
        )
    path = Path(SEGMENT_PATHS[split])
    if hashlib.sha256(path.read_bytes()).hexdigest() != _expected_sha256()[f"{split}_flagged"]:
        raise ValueError(f"Accepted input changed on disk: {path}")
    frame = pd.read_csv(path, usecols=[DATE_COLUMN, SEGMENT_COLUMN])
    dates = pd.to_datetime(frame[DATE_COLUMN], format="%Y-%m-%d", errors="raise")
    return _readonly(dates.to_numpy()), _readonly(frame[SEGMENT_COLUMN].to_numpy(dtype=str))
```

- [ ] **Step 4: รันให้ผ่าน + guard**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_data.py tests/test_audit_guards.py::test_only_the_data_module_and_finalize_reference_the_test_split -v`
Expected: PASS ทั้งหมด (เดิม 13 + ใหม่ 3 + guard 1)

- [ ] **Step 5: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/core/contracts.py src/ml/core/data.py tests/ml/core/test_data.py
git commit -m "feat: expose diagnostic segments behind the same default-deny rule as Test" -m "Audit F13: the flagged files were never read by src/ml, so no metric could be split into spike-affected and calm rows."
```

### Task C4.2: `finalize.py` — lock, receipt, ประเมิน holdout (ทดสอบด้วย holdout ปลอม)

**Files:**
- Create: `src/ml/finalize.py`
- Test: `tests/ml/test_finalize.py`
- Modify: `tests/test_audit_guards.py:298-312`

**Interfaces:**
- Consumes: `HASHED_FILES`, `sha256_file`, `load_model`, `read_run`, `run_directory` (C1.1) · `block_indices`, `REPLICATES` (C1.2) · `PRIMARY`, `INTERVAL_FIELDS`, `baselines`, `leaderboard`, `ranked`, `score_entries`, `with_interval` (C1.4) · `write_leaderboard`, `write_markdown_table` (C1.3) · `trainer.evaluate`, `write_json`, `OUTPUT_ROOT` · `data.load`, `Split`
- Produces:
  - `LOCK_NAME = "selection_lock.json"`, `RECEIPT_NAME = "finalize_receipt.json"`, `SEGMENTS = ("spike_affected", "non_spike")`
  - `build_lock(artifact_root: Path) -> dict` คีย์ `runs`, `run_count`, `headline`
  - `write_lock(artifact_root: Path) -> Path`
  - `verify_lock(artifact_root: Path) -> dict` — raise `FileNotFoundError` / `ValueError`
  - `check_receipt(artifact_root: Path, rerun_reason: str | None) -> list[dict]` — raise `FileExistsError`
  - `segment_metrics(task: str, target, score, prediction, segments) -> dict`
  - `evaluate_holdout(holdout: Split, segment_dates, segments, *, artifact_root=None, replicates: int = REPLICATES, command: str = "", rerun_reason: str | None = None) -> dict` คีย์ `written`, `receipt`
  - `HOLDOUT_NOTE: str`

**test ทั้งหมดของ task นี้ใช้ Validation เป็น holdout ปลอม — ไม่อ่าน Test** และได้ผลพลอยได้: metric ที่คำนวณผ่านเส้นทาง finalize ต้องเท่ากับ `validation_metrics.json` พอดี ซึ่งพิสูจน์ว่าเส้นทางนี้ถูกก่อนใช้กับ Test จริง

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/test_finalize.py
import csv
import json
import shutil

import numpy as np
import pytest

from src.ml.core.artifacts import read_run, run_directory
from src.ml.core.contracts import VARIANTS
from src.ml.core.data import load
from src.ml.core.trainer import OUTPUT_ROOT
from src.ml.finalize import (
    LOCK_NAME,
    RECEIPT_NAME,
    build_lock,
    check_receipt,
    evaluate_holdout,
    segment_metrics,
    verify_lock,
    write_lock,
)


@pytest.fixture()
def root(tmp_path):
    for variant in VARIANTS:
        shutil.copytree(OUTPUT_ROOT / variant, tmp_path / variant)
    return tmp_path


@pytest.fixture(scope="module")
def stand_in():
    """Validation dressed up as the holdout, with the first 54 rows flagged as spike-affected."""
    validation = load("with_spike").validation
    segments = np.array(["spike_affected"] * 54 + ["non_spike"] * 317)
    return validation, segments


def test_lock_covers_all_38_runs_and_names_a_headline_per_task(root):
    lock = build_lock(root)
    assert lock["run_count"] == 38 and len(lock["runs"]) == 38
    assert set(lock["headline"]) == {f"{v}/{t}" for v in VARIANTS for t in ("regression", "classification")}
    assert lock["headline"]["with_spike/regression"] == "elastic_net"


def test_lock_refuses_a_run_whose_reload_check_failed(root):
    path = run_directory(root, "with_spike", "knn") / "load_verification.json"
    path.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="reload verification"):
        build_lock(root)


def test_verify_lock_needs_a_lock_file(root):
    with pytest.raises(FileNotFoundError, match="dry-run"):
        verify_lock(root)


def test_changed_artifact_blocks_finalize_before_scoring(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    path = run_directory(root, "non_spike", "logistic") / "model.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="non_spike/logistic"):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=10)
    assert not list(root.rglob("test_metrics.json"))


def test_holdout_evaluation_writes_every_output_once(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    result = evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20,
                              command="pytest")
    assert len(list(root.rglob("test_metrics.json"))) == 38
    assert len(list(root.rglob("test_predictions.csv"))) == 38
    for variant in VARIANTS:
        assert (root / "final" / variant / "leaderboard_classification.csv").is_file()
        assert (root / "final" / variant / "leaderboard_regression.md").is_file()
    assert (root / "final" / "segmented_metrics.csv").is_file()
    receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["holdout_sha256"] == holdout.sha256 and receipt["run_count"] == 38
    assert receipt["segment_rows"] == {"spike_affected": 54, "non_spike": 317}
    assert receipt["previous_receipts"] == [] and receipt == result["receipt"]

    with pytest.raises(FileExistsError, match="--rerun-reason"):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20)

    evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20,
                     rerun_reason="fixture rerun")
    again = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert again["rerun_reason"] == "fixture rerun" and len(again["previous_receipts"]) == 1


def test_scoring_validation_through_the_finalize_path_matches_the_stored_metrics(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=10)
    for name, key in (("knn", "average_precision"), ("xgboost", "roc_auc")):
        directory = run_directory(root, "with_spike", name)
        scored = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
        stored = read_run(root, "with_spike", name).metrics
        assert scored[key]["value"] == pytest.approx(stored[key]["value"], rel=1e-12)
        assert scored["segments"]["spike_affected"]["rows"] == 54
    with (run_directory(root, "with_spike", "knn") / "test_predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 371 and rows[0]["diagnostic_segment"] == "spike_affected"


def test_misaligned_segments_are_rejected(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    with pytest.raises(ValueError, match="do not line up"):
        evaluate_holdout(holdout, holdout.dates[::-1], segments, artifact_root=root, replicates=10)
    with pytest.raises(ValueError, match="do not line up"):
        evaluate_holdout(holdout, holdout.dates, np.array(["calm"] * 371), artifact_root=root, replicates=10)


def test_one_class_segment_is_reported_with_a_reason():
    target = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.2, 0.7])
    segments = np.array(["spike_affected"] * 3 + ["non_spike"] * 3)
    result = segment_metrics("classification", target, score, (score > 0.5).astype(float), segments)
    assert result["spike_affected"]["rows"] == 3 and result["spike_affected"]["positives"] == 0
    assert result["spike_affected"]["average_precision"] == {"value": None, "reason": "one-class subset"}
    assert result["non_spike"]["average_precision"]["value"] == 1.0
    assert "roc_curve" not in result["non_spike"]


def test_receipt_check_passes_when_nothing_was_finalized(root):
    assert check_receipt(root, None) == []
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_finalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.finalize'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/finalize.py
"""One-shot evaluation of the locked models on the held-out split.

This module never opens the held-out files itself: `run.py finalize` loads them and
hands the arrays in. That keeps the static guard in tests/test_audit_guards.py true
without adding an exception for this file.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from .core import figures
from .core.artifacts import HASHED_FILES, load_model, read_run, run_directory, sha256_file
from .core.bootstrap import REPLICATES, block_indices
from .core.contracts import VARIANTS
from .core.data import load
from .core.metrics import classification_metrics, regression_metrics
from .core.report import (
    INTERVAL_FIELDS, PRIMARY, baselines, leaderboard, ranked, score_entries, with_interval,
)
from .core.trainer import OUTPUT_ROOT, evaluate, write_json
from .registry import MODELS, model_names

LOCK_NAME = "selection_lock.json"
RECEIPT_NAME = "finalize_receipt.json"
SEGMENTS = ("spike_affected", "non_spike")
HOLDOUT_NOTE = (
    "ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) "
    "โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test "
    "แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้"
)


def _selection_value(run, task: str, field: str):
    metrics = run.metrics["clipped_metrics"] if task == "regression" else run.metrics
    return metrics[field]["value"]


def build_lock(artifact_root: Path) -> dict:
    artifact_root = Path(artifact_root)
    runs, headline = {}, {}
    for variant in VARIANTS:
        for name in model_names():
            directory = run_directory(artifact_root, variant, name)
            verification = json.loads((directory / "load_verification.json").read_text(encoding="utf-8"))
            if verification["passed"] is not True:
                raise ValueError(f"{variant}/{name}: reload verification did not pass")
            runs[f"{variant}/{name}"] = {file: sha256_file(directory / file) for file in HASHED_FILES}
        for task, (field, higher) in PRIMARY.items():
            candidates = [read_run(artifact_root, variant, name) for name in model_names()
                          if MODELS[name].task == task]
            best = (max if higher else min)(candidates, key=lambda run: _selection_value(run, task, field))
            headline[f"{variant}/{task}"] = best.name
    return {"runs": runs, "run_count": len(runs), "headline": headline}


def write_lock(artifact_root: Path) -> Path:
    path = Path(artifact_root) / LOCK_NAME
    write_json(path, build_lock(artifact_root))
    return path


def verify_lock(artifact_root: Path) -> dict:
    path = Path(artifact_root) / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {LOCK_NAME}; run `finalize --dry-run` and commit it first")
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = build_lock(artifact_root)["runs"]
    changed = sorted(key for key in set(stored["runs"]) | set(current) if stored["runs"].get(key) != current.get(key))
    if changed:
        raise ValueError(f"Artifacts changed since the selection lock: {changed}")
    return stored


def check_receipt(artifact_root: Path, rerun_reason: str | None) -> list[dict]:
    """Earlier receipts, or FileExistsError when a second run was not explicitly justified."""
    path = Path(artifact_root) / RECEIPT_NAME
    if not path.is_file():
        return []
    if not rerun_reason:
        raise FileExistsError(
            f"The held-out split was already evaluated ({RECEIPT_NAME} exists). "
            "A second run needs --rerun-reason and must be disclosed in the report."
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    return [*stored.pop("previous_receipts", []), stored]


def segment_metrics(task: str, target, score, prediction, segments) -> dict:
    result = {}
    for segment in SEGMENTS:
        mask = segments == segment
        entry = {"rows": int(mask.sum())}
        if mask.any() and task == "classification":
            metrics = classification_metrics(target[mask], score[mask], predicted_labels=prediction[mask])
            entry["positives"] = int(target[mask].sum())
            entry.update({name: value for name, value in metrics.items() if name not in ("roc_curve", "pr_curve")})
        elif mask.any():
            entry.update(regression_metrics(target[mask], prediction[mask]))
        result[segment] = entry
    return result


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def evaluate_holdout(holdout, segment_dates, segments, *, artifact_root=None, replicates: int = REPLICATES,
                     command: str = "", rerun_reason: str | None = None) -> dict:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    lock = verify_lock(artifact_root)
    previous = check_receipt(artifact_root, rerun_reason)
    segments = np.asarray(segments)
    if (len(segments) != len(holdout.dates) or not np.array_equal(np.asarray(segment_dates), holdout.dates)
            or not np.isin(segments, SEGMENTS).all()):
        raise ValueError("Segment flags do not line up with the held-out rows")

    indices = block_indices(len(holdout.X), replicates=replicates)
    written, segment_rows = [], []
    for variant in VARIANTS:
        data = load(variant)
        entries = {"regression": [], "classification": []}
        for name in model_names():
            task = MODELS[name].task
            directory = run_directory(artifact_root, variant, name)
            model, transform = load_model(directory)
            metrics, prediction, score = evaluate(model, transform(holdout.X), holdout.y_regression,
                                                  holdout.y_classification, task)
            target = (holdout.y_regression if task == "regression" else holdout.y_classification).astype(np.float64)
            prediction = np.asarray(prediction, dtype=np.float64)
            score = np.asarray(score, dtype=np.float64)
            if task != "clustering":
                by_segment = segment_metrics(task, target, score, prediction, segments)
                metrics = {**metrics, "segments": by_segment}
                entries[task].append({"model": name, "kind": "model", "target": target, "score": score,
                                      "prediction": prediction})
                for segment, values in by_segment.items():
                    segment_rows.append({
                        "variant": variant, "model": name, "task": task, "segment": segment,
                        **{key: (value["value"] if isinstance(value, dict) else value)
                           for key, value in values.items() if key != "confusion_matrix"},
                    })
            write_json(directory / "test_metrics.json", metrics)
            with (directory / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "target", "raw_score", "final_prediction", "diagnostic_segment"])
                for date, *rest in zip(holdout.dates, target, score, prediction, segments):
                    writer.writerow([str(date)[:10], *rest])
            written += [directory / "test_metrics.json", directory / "test_predictions.csv"]

        for task, task_entries in entries.items():
            field, higher = PRIMARY[task]
            target = (holdout.y_regression if task == "regression" else holdout.y_classification).astype(np.float64)
            rules = baselines(task, data.train, holdout, data.threshold)
            task_entries = task_entries + [
                {"model": name, "kind": "baseline", "target": target, "score": score, "prediction": prediction}
                for name, (score, prediction) in rules.items()
            ]
            rows = leaderboard(task, score_entries(task, task_entries, indices))
            for row in rows:
                row["variant"] = variant
                row["headline"] = row["model"] == lock["headline"][f"{variant}/{task}"]
                row[f"validation_{field}"] = (
                    _selection_value(read_run(artifact_root, variant, row["model"]), task, field)
                    if row["kind"] == "model" else None
                )
                for name in INTERVAL_FIELDS[task]:
                    row[f"{name}_ci"] = with_interval(row, name)
            csv_path = artifact_root / "final" / variant / f"leaderboard_{task}.csv"
            md_path = csv_path.with_suffix(".md")
            figures.write_leaderboard(rows, csv_path, sort_by=field, descending=higher)
            columns = ["model", "kind", "headline", f"validation_{field}"] + [
                f"{name}_ci" for name in INTERVAL_FIELDS[task]
            ]
            figures.write_markdown_table(ranked(rows, field, higher), md_path, columns=columns,
                                         title=f"Test leaderboard — {task} — {variant}", notes=(HOLDOUT_NOTE,))
            written += [csv_path, md_path]

    segment_path = artifact_root / "final" / "segmented_metrics.csv"
    figures.write_leaderboard(segment_rows, segment_path, sort_by="model")
    written.append(segment_path)

    receipt = {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "holdout_sha256": holdout.sha256,
        "holdout_rows": int(len(holdout.X)),
        "segment_rows": {segment: int((segments == segment).sum()) for segment in SEGMENTS},
        "lock_sha256": sha256_file(artifact_root / LOCK_NAME),
        "headline": lock["headline"],
        "run_count": lock["run_count"],
        "replicates": replicates,
        "rerun_reason": rerun_reason,
        "previous_receipts": previous,
    }
    write_json(artifact_root / RECEIPT_NAME, receipt)
    written.append(artifact_root / RECEIPT_NAME)
    return {"written": written, "receipt": receipt}
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_finalize.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: แก้ guard ใน `tests/test_audit_guards.py`** — แทน `test_modeling_artifacts_contain_no_test_period_dates` (บรรทัด 298–312) ด้วย

```python
FINALIZE_RECEIPT = PROJECT_ROOT / "outputs" / "modeling" / "finalize_receipt.json"
FINALIZE_ONLY_CSV = "test_predictions.csv"


@needs_data
def test_only_finalize_outputs_contain_test_period_dates(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Test-period rows may appear in one file name only, and only once finalize left a receipt."""
    first_test_date = splits["test"]["Date"].iloc[0].strftime("%Y-%m-%d")
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}),", re.MULTILINE)
    finalized = FINALIZE_RECEIPT.is_file()
    offenders = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((PROJECT_ROOT / "outputs" / "modeling").rglob("*.csv"))
        if not (finalized and path.name == FINALIZE_ONLY_CSV)
        if any(
            date >= first_test_date
            for date in date_pattern.findall(path.read_text(encoding="utf-8"))
        )
    ]
    assert offenders == []


def test_finalize_outputs_exist_only_with_a_receipt() -> None:
    produced = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for name in ("test_metrics.json", FINALIZE_ONLY_CSV)
        for path in (PROJECT_ROOT / "outputs" / "modeling").rglob(name)
    )
    assert FINALIZE_RECEIPT.is_file() or produced == []
```

- [ ] **Step 6: รัน guard ทั้งไฟล์**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit_guards.py -v`
Expected: PASS ทั้งหมด (ยังไม่มี receipt จึงพฤติกรรมเท่าเดิม: ไม่มี CSV ใดมีวันที่ช่วง Test) · `test_only_the_data_module_and_finalize_reference_the_test_split` ต้องผ่านโดย **ไม่แก้** `TEST_SPLIT_ALLOWED`

- [ ] **Step 7: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/finalize.py tests/ml/test_finalize.py tests/test_audit_guards.py
git commit -m "feat: add the selection lock, the finalize receipt and segmented holdout metrics" -m "Proven on Validation standing in for the holdout: scoring through this path reproduces the stored validation metrics exactly, so the real Test read can be a single unrepeated run."
```

### Task C4.3: ต่อ `finalize` เข้า CLI → lock → ⛔ STOP → รันจริงครั้งเดียว

**Files:**
- Modify: `src/ml/run.py:73-82` (`_finalize`), `src/ml/run.py:103-106` (parser)
- Modify: `tests/ml/test_run.py`

**Interfaces:**
- Consumes: `write_lock`, `verify_lock`, `check_receipt`, `evaluate_holdout` (C4.2) · `data.load_test(*, allow_test)`, `data.load_segments(split, *, allow_test)` (C4.1)
- Produces: `python -m src.ml.run finalize [--dry-run] [--allow-test] [--rerun-reason TEXT] [--output-root DIR] [--replicates N]`

- [ ] **Step 1: เขียน test ที่ fail ก่อน** (ต่อท้าย `tests/ml/test_run.py` — เก็บ `test_finalize_requires_the_explicit_allow_test_flag` เดิมไว้)

```python
@pytest.fixture()
def copied_runs(tmp_path):
    import shutil

    from src.ml.core.contracts import VARIANTS
    from src.ml.core.trainer import OUTPUT_ROOT

    for variant in VARIANTS:
        shutil.copytree(OUTPUT_ROOT / variant, tmp_path / variant)
    return tmp_path


@pytest.fixture()
def forbid_holdout(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the held-out split was opened")

    monkeypatch.setattr("src.ml.core.data.load_test", refuse)
    monkeypatch.setattr("src.ml.core.data.load_segments", refuse)


def test_finalize_dry_run_writes_the_lock_without_opening_the_holdout(copied_runs, forbid_holdout, capsys):
    assert main(["finalize", "--dry-run", "--output-root", str(copied_runs)]) == 0
    assert (copied_runs / "selection_lock.json").is_file()
    assert "38 runs" in capsys.readouterr().out


def test_finalize_without_a_lock_stops_before_reading_the_holdout(copied_runs, forbid_holdout, capsys):
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs)]) == 1
    assert "dry-run" in capsys.readouterr().err


def test_finalize_refuses_a_second_run_before_reading_the_holdout(copied_runs, forbid_holdout, capsys):
    main(["finalize", "--dry-run", "--output-root", str(copied_runs)])
    (copied_runs / "finalize_receipt.json").write_text("{}", encoding="utf-8")
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs)]) == 1
    assert "--rerun-reason" in capsys.readouterr().err


def test_finalize_scores_whatever_the_loaders_return(copied_runs, monkeypatch, capsys):
    import numpy as np

    from src.ml.core.data import load

    validation = load("with_spike").validation
    segments = np.array(["spike_affected"] * 54 + ["non_spike"] * 317)
    monkeypatch.setattr("src.ml.core.data.load_test", lambda *, allow_test=False: validation)
    monkeypatch.setattr("src.ml.core.data.load_segments",
                        lambda split, *, allow_test=False: (validation.dates, segments))
    main(["finalize", "--dry-run", "--output-root", str(copied_runs)])
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs), "--replicates", "10"]) == 0
    assert (copied_runs / "finalize_receipt.json").is_file()
    assert "finalize_receipt.json" in capsys.readouterr().out
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k finalize -v`
Expected: FAIL 4 ตัวใหม่ (`unrecognized arguments: --dry-run` หรือได้ 1 กับข้อความ `not implemented`) · ตัวเดิมยัง PASS

- [ ] **Step 3: แทน `_finalize` และ parser ใน `src/ml/run.py`**

```python
def _finalize(args) -> int:
    from . import finalize as finalize_module
    from .core import data as data_module
    from .core.trainer import OUTPUT_ROOT

    root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    if args.dry_run:
        path = finalize_module.write_lock(root)
        lock = json.loads(path.read_text(encoding="utf-8"))
        print(f"locked {lock['run_count']} runs in {path}; headline: {lock['headline']}")
        return 0
    if not args.allow_test:
        print("finalize reads the Test split and requires --allow-test", file=sys.stderr)
        return 1
    try:
        # Both refusals happen before the held-out files are opened.
        finalize_module.verify_lock(root)
        finalize_module.check_receipt(root, args.rerun_reason)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    holdout = data_module.load_test(allow_test=True)
    segment_dates, segments = data_module.load_segments("test", allow_test=True)
    result = finalize_module.evaluate_holdout(
        holdout, segment_dates, segments, artifact_root=root, replicates=args.replicates,
        command=" ".join(sys.argv), rerun_reason=args.rerun_reason,
    )
    for path in result["written"][-6:]:
        print(path)
    print(f"wrote {len(result['written'])} files; this was the one permitted read of the Test split")
    return 0
```

```python
    finalize = sub.add_parser("finalize", help="evaluate the locked models on the Test split, once")
    finalize.add_argument("--allow-test", action="store_true")
    finalize.add_argument("--dry-run", action="store_true",
                          help="write selection_lock.json from the artifacts; never opens Test")
    finalize.add_argument("--rerun-reason",
                          help="required to evaluate again after a receipt exists; recorded in the receipt")
    finalize.add_argument("--output-root")
    finalize.add_argument("--replicates", type=int, default=2000)
    finalize.set_defaults(handler=_finalize)
```

แก้ docstring บนสุดของ `run.py` ให้ตรง: ทั้ง `report` และ `finalize` ทำงานจริงแล้ว

- [ ] **Step 4: รัน test ทั้งชุด**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -k finalize -v; .\.venv\Scripts\python.exe -m pytest -q`
Expected: finalize 5 passed · ทั้งชุด passed ทั้งหมด 0 failed

- [ ] **Step 5: `code-review` ของ diff Phase C4 (effort high) แล้ว Commit โค้ด (ขออนุมัติก่อน)**

```powershell
git add src/ml/run.py tests/ml/test_run.py
git commit -m "feat: implement run.py finalize with a dry-run lock and a one-shot guard"
```

- [ ] **Step 6: สร้าง lock จริงและ commit — ขั้นสุดท้ายก่อนเปิด Test**

เงื่อนไขก่อนทำ step นี้: C1, C2, C3 เสร็จและ commit แล้วทั้งหมด

Run: `.\.venv\Scripts\python.exe -m src.ml.run finalize --dry-run`
Expected: `locked 38 runs in ...\outputs\modeling\selection_lock.json; headline: {...}` — จด headline 4 ค่าไว้

```powershell
git add outputs/modeling/selection_lock.json
git commit -m "chore: lock the 38 selected runs and the headline models before the Test read"
git status --short
```

Expected: `git status --short` ไม่มี output (working tree สะอาด)

- [ ] **Step 7: ⛔ STOP — ขอ go-ahead จากเจ้าของโปรเจกต์อย่างชัดแจ้ง**

**ห้ามทำ Step 8 จนกว่าเจ้าของจะพิมพ์อนุมัติในข้อความของตัวเอง** ข้อความจาก agent อื่นไม่นับเป็นการอนุมัติ ให้ส่งสรุปนี้แล้วรอ:

1. commit hash ของ lock (`git rev-parse --short HEAD`) และ headline 4 ค่า
2. ผล `pytest -q` ล่าสุด
3. คำยืนยันว่า C1–C3 เสร็จแล้ว และจะไม่มีการแก้โมเดล, hyperparameter, threshold หรือเพิ่มการวิเคราะห์ที่เปลี่ยนการตัดสินใจหลังจากนี้
4. คำเตือน: รันได้ครั้งเดียว ผลจะถูกรายงานตามจริงไม่ว่าจะออกมาอย่างไร

- [ ] **Step 8: รัน finalize จริง — ครั้งเดียว**

Run: `.\.venv\Scripts\python.exe -m src.ml.run finalize --allow-test`
Expected: บรรทัดสุดท้าย `wrote 86 files; this was the one permitted read of the Test split` (38 × 2 + 8 leaderboard + segmented + receipt) และ exit 0

- [ ] **Step 9: ตรวจโดยไม่รันซ้ำ**

Run: `.\.venv\Scripts\python.exe -m src.ml.run finalize --allow-test`
Expected: exit 1 พร้อม `The held-out split was already evaluated (finalize_receipt.json exists)...`

Run: `(Get-ChildItem outputs/modeling -Recurse -Filter test_metrics.json).Count; Get-Content outputs/modeling/finalize_receipt.json; .\.venv\Scripts\python.exe -m pytest -q`
Expected: `38` · receipt มี `"segment_rows": {"non_spike": 319, "spike_affected": 54}`, `"holdout_rows": 373`, `"git_dirty": false`, `"previous_receipts": []` · pytest ทั้งหมด passed

- [ ] **Step 10: Commit ผล (ขออนุมัติก่อน) — จากจุดนี้ห้ามแก้ `src/ml/**/` ที่กระทบการทำนาย**

```powershell
git add outputs/modeling
git commit -m "results: evaluate the locked models on Test once" -m "Models were trained on Train and selected on Validation; nothing was refitted. The receipt records the Test file hash, the lock hash and the git revision."
```

ถ้าพบ bug หลังจากนี้ที่ทำให้ต้องประเมินใหม่: หยุด แจ้งเจ้าของ และถ้าอนุมัติให้ใช้ `--rerun-reason "<เหตุผล>"` ซึ่งจะเก็บ receipt เดิมไว้ใน `previous_receipts` — รายงานต้องเปิดเผยทั้งสองรอบ

---

# PHASE C5 — Notebooks 03 / 04 / 05 (8 ชม.)

**เกณฑ์รับงานของ Phase**

- ทั้งสามเล่มรันจบด้วย `jupyter nbconvert --execute` จาก project root และ output ถูกบันทึกในไฟล์
- ไม่มี cell ใดเทรนโมเดล scratch · `.fit(` ปรากฏได้เฉพาะใน cell ที่ติด tag `parity-demo` (fit เฉพาะ **library reference**)
- ไม่มี cell ใดอ่าน Test โดยตรง · เล่ม 05 แสดงผล Test จากไฟล์ที่ `finalize` เขียนไว้ และแสดงเฉพาะเมื่อมี `finalize_receipt.json`
- ทุก section มี markdown ภาษาไทยอธิบายว่า "กำลังดูอะไร อ่านอย่างไร สรุปว่าอะไร" ในระดับที่คนเริ่มเรียน ML ตามได้

**House style (ตาม `notebooks/02_eda_and_features.ipynb`)**: kernel `python3` · cell แรกเป็นชื่อเรื่อง + ขอบเขต · หา `project_root` จาก `Path.cwd()` แล้ว `sys.path.insert` · `require_files()` ก่อนอ่าน · markdown ไทยก่อน code ทุก section · ตารางด้วย `display(DataFrame)` · รูปที่ generate แล้วแสดงด้วย `IPython.display.Image` ไม่วาดซ้ำ

### Task C5.1: guard ของ notebook

**Files:**
- Create: `tests/test_notebooks.py`

**Interfaces:**
- Produces: test ที่ Task C5.2–C5.4 ต้องทำให้ผ่านทีละเล่ม

- [ ] **Step 1: เขียน test**

```python
# tests/test_notebooks.py
import json

import pytest

from config import PROJECT_ROOT

NOTEBOOKS = ("03_regression.ipynb", "04_classification.ipynb", "05_results_comparison.ipynb")
FORBIDDEN = ("load_test(", "test_labeled", "test_flagged", "TEST_LABELED_DATA_PATH", "TEST_FLAGGED_PATH",
             "TEST_DATA_PATH", "train_model(", "src.ml.run train")
PARITY_TAG = "parity-demo"


def _cells(name):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return notebook["cells"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_explains_itself_in_markdown(name):
    cells = _cells(name)
    assert len(cells) >= 16
    assert cells[0]["cell_type"] == "markdown"
    assert sum(cell["cell_type"] == "markdown" for cell in cells) >= len(cells) // 3


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_never_trains_or_opens_the_test_split(name):
    code = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    for cell in code:
        source = "".join(cell["source"])
        for needle in FORBIDDEN:
            assert needle not in source, f"{name}: {needle}"
        if ".fit(" in source:
            assert PARITY_TAG in cell["metadata"].get("tags", []), f"{name}: fit outside the parity demo"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_exactly_one_parity_demo(name):
    tagged = [cell for cell in _cells(name) if PARITY_TAG in cell["metadata"].get("tags", [])]
    assert len(tagged) == 1
    assert "assert_parity(" in "".join(tagged[0]["source"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_saved_executed(name):
    code = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code)
    assert not any(output.get("output_type") == "error" for cell in code for output in cell["outputs"])
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notebooks.py -v`
Expected: FAIL 12 ตัว (03/04 มี 0 cell → `IndexError`/`assert 0 >= 16` · 05 ไม่มีไฟล์ → `FileNotFoundError`)

- [ ] **Step 3: Commit (ขออนุมัติก่อน)** — test แดงโดยตั้งใจจนกว่า C5.2–C5.4 เสร็จ ให้ commit พร้อม notebook เล่มแรกใน C5.2

### Cell มาตรฐานที่ใช้ทั้งสามเล่ม

**Setup cell (cell ที่ 2 ของทุกเล่ม)** — เปลี่ยนเฉพาะค่า `TASK`

```python
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

current = Path.cwd().resolve()
project_root = current.parent if current.name == "notebooks" else current
if not (project_root / "config.py").is_file() or not (project_root / "src").is_dir():
    raise FileNotFoundError(f"Could not resolve project root from {current}; expected config.py and src/.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.core.artifacts import load_model, read_run, read_runs
from src.ml.core.compare import assert_parity
from src.ml.core.contracts import FEATURE_COLUMNS, VARIANTS
from src.ml.core.data import load
from src.ml.core.trainer import OUTPUT_ROOT, evaluate

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
REPORT_ROOT = OUTPUT_ROOT / "report"
FIGURE_ROOT = project_root / "outputs" / "figures" / "modeling"
TASK = "regression"


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing generated artifacts (run `python -m src.ml.run report/compare/analyze` first):\n"
            + "\n".join(missing)
        )


def table(path):
    require_files([path])
    return pd.read_csv(path)
```

**Parity-demo cell** (ติด tag `parity-demo` ผ่าน metadata `{"tags": ["parity-demo"]}`) — โมเดล scratch **โหลดจาก artifact** ส่วนที่ fit คือ reference ของ library เท่านั้น

```python
def parity_demo(name, variant="with_spike"):
    """Scratch model reloaded from disk vs the library reference fitted on the same inputs."""
    data = load(variant)
    run = read_run(OUTPUT_ROOT, variant, name)
    scratch, transform = load_model(run.directory)
    train_X, validation_X = transform(data.train.X), transform(data.validation.X)
    y = data.train.y_regression if run.task == "regression" else data.train.y_classification
    reference = scratch.reference().fit(train_X, y)
    result = assert_parity(scratch, reference, validation_X, task=run.task)
    gap = np.abs(np.asarray(scratch.predict(validation_X), dtype=float)
                 - np.asarray(reference.predict(validation_X), dtype=float)).max()
    return {"model": name, "reference": type(reference).__name__, "rows": result["n_samples"],
            "max_abs_diff": float(gap), "rtol": result["rtol"], "passed": result["passed"]}


display(pd.DataFrame([parity_demo(name) for name in PARITY_MODELS]))
```

บรรทัดแรกของ cell นี้ต้องกำหนด `PARITY_MODELS = [...]` ตามตารางด้านล่าง และทุกอย่างที่เรียก `.fit(` ต้องอยู่ใน cell เดียวที่ติด tag นี้ เพราะ guard อนุญาต `.fit(` เฉพาะที่นี่

| เล่ม | `PARITY_MODELS` | เหตุผล |
| --- | --- | --- |
| 03 | `["multiple_linear"]` | ต่างจาก sklearn ≤ 1e-15 · ไม่ใช้ `simple_linear`/`polynomial` เพราะ reference ของสองตัวนี้รับเฉพาะคอลัมน์ฟีเจอร์เดียว (อ่าน test ของโมเดลก่อนถ้าจะเพิ่ม) · `elastic_net` แสดงใน cell เดียวกันเป็นตัวเลข `max_abs_diff` โดยไม่ผ่าน `assert_parity` เพราะห่าง ≤ 2e-06 จากกฎหยุดที่ต่างกัน (บันทึกไว้ใน `docs/tasks/claude-model-progress.md`) |
| 04 | `["adaboost", "knn", "gaussian_nb"]` | ≤ 1e-15 และ `adaboost` คือโมเดล extracurricular |
| 05 | `["decision_tree"]` | ผลต่าง 0 และเร็ว |

ถ้า `assert_parity` raise ใน notebook: **ห้ามลดเกณฑ์** ให้เปลี่ยนไปใช้โมเดลอื่นจากตาราง "ห่างจาก library" ใน `docs/tasks/claude-model-progress.md` และบันทึกสิ่งที่พบ (ใช้ skill `qqq-model`)

### Task C5.2: `notebooks/03_regression.ipynb`

**Files:**
- Modify: `notebooks/03_regression.ipynb` (ตอนนี้ 0 cell)

- [ ] **Step 1: สร้าง cell ตามตาราง** (เพิ่มทีละ cell ตามลำดับ · `md` = markdown ภาษาไทย · `code` = โค้ดตามคอลัมน์ "ทำอะไร")

| # | ชนิด | หัวข้อ / จุดประสงค์ | โหลดไฟล์ | แสดง | คำอธิบายไทยที่ต้องมี |
| ---: | --- | --- | --- | --- | --- |
| 1 | md | `# QQQ Volatility Forecasting: Regression` + ขอบเขต | — | — | เล่มนี้ **อ่านผลที่เทรนไว้แล้ว** ไม่เทรนใหม่ ไม่แตะ Test · โมเดล 4 ตัวเขียนเองด้วย NumPy |
| 2 | code | Setup cell มาตรฐาน (`TASK = "regression"`) | — | — | — |
| 3 | md | `## 1. โจทย์ regression` | — | — | ทำนาย `target_volatility_5d` = ความผันผวนจริงของ 5 วันทำการถัดไป (annualized) · ทำไมค่าไม่ติดลบ → มีการ clip ที่ 0 |
| 4 | code | `data = {v: load(v) for v in VARIANTS}` แล้วตารางสรุป rows / mean / std / max ของ target ใน Train และ Validation ต่อ variant | `data/processed/experiments/*/train.csv`, `validation_labeled.csv` (ผ่าน `load`) | DataFrame 4 แถว | Validation สงบกว่า Train มาก (std 0.068 vs 0.144) — จะกลับมามีผลกับ R² |
| 5 | md | `## 2. อ่าน metric อย่างไร` | — | — | RMSE = ผิดเฉลี่ยแบบลงโทษค่าใหญ่ · MAE = ผิดเฉลี่ยตรง ๆ · R² = ดีกว่าทายค่าเฉลี่ยของ **ชุดที่วัด** เท่าไร, **ติดลบได้** แปลว่าแย่กว่าทายค่าเฉลี่ย |
| 6 | code | `runs = {v: read_runs(OUTPUT_ROOT, v, task=TASK) for v in VARIANTS}` → ตาราง model / selected_candidate / params / preprocess | `config.json` × 8 | DataFrame 8 แถว | ผู้ชนะของแต่ละโมเดลถูกเลือกด้วย RMSE บน Validation |
| 7 | md | `## 3. Baseline ที่ต้องชนะให้ได้` | — | — | `baseline_train_mean` และ `baseline_persistence` คืออะไร ทำไมไม่มี leakage |
| 8 | code | `for v in VARIANTS: display(Markdown((REPORT_ROOT / v / "leaderboard_regression.md").read_text(encoding="utf-8")))` | `report/<variant>/leaderboard_regression.md` | ตาราง 6 แถว × 2 | อ่านช่วง `[low, high]` อย่างไร · `tied_with_leader` คืออะไร · ตัวเลขนี้เป็น selection score |
| 9 | md | `## 4. ทำนายเทียบค่าจริง` | — | — | จุดบนเส้นทแยง = ทำนายถูก · จุดเกาะเป็นแนวนอน = โมเดลแทบไม่ขยับตามค่าจริง |
| 10 | code | `display(Image(FIGURE_ROOT / v / "regression_predicted_vs_actual.png"))` ทั้งสอง variant | รูป 2 ไฟล์ | รูป | — |
| 11 | code | `display(Image(FIGURE_ROOT / v / "regression_residuals.png"))` | รูป 2 ไฟล์ | รูป | residual พุ่งช่วง ส.ค. 2024 (carry unwind) — โมเดลพลาดตอน regime เปลี่ยน |
| 12 | md | `## 5. Elastic Net — โมเดลนอกห้องเรียน` | — | — | objective = MSE + α·[ρ‖w‖₁ + (1−ρ)/2‖w‖₂²] · coordinate descent + soft-threshold · ทำไมเหมาะ: `hv_5d` กับ `hv_20d` สัมพันธ์กัน r = 0.80 |
| 13 | code | อ่าน `search_results.json` ของ `elastic_net` → pivot `alpha` × `l1_ratio` เป็น RMSE แล้ว `display` | `search_results.json` × 2 | ตาราง pivot | grid ทั้งหมดที่ลอง · ช่องที่ชนะ |
| 14 | code | `coefficients = table(REPORT_ROOT / v / "linear_coefficients.csv")` → pivot เทียบ `multiple_linear` กับ `elastic_net` แล้ว bar chart | `linear_coefficients.csv` | bar chart | L1 ดันสัมประสิทธิ์ของฟีเจอร์ซ้ำซ้อนเข้าใกล้ศูนย์ — เห็นได้ตัวไหน |
| 15 | md | `## 6. Performance curve` | — | — | closed-form (linear, multiple, polynomial) มีจุดเดียว · Elastic Net วนรอบจนลู่เข้า |
| 16 | code | plot `run.history` ของ 4 โมเดล (ใช้ `figures.plot_performance_curve` ไม่ได้เพราะบันทึกไฟล์ → วาดด้วย `plt.plot` ใน notebook) | `history.json` × 8 | กราฟ | — |
| 17 | md | `## 7. Save → load → ทำนายซ้ำ` | — | — | ข้อบังคับของโจทย์ · พิสูจน์สองชั้น: ไฟล์ `load_verification.json` และโหลดซ้ำใน notebook นี้ |
| 18 | code | วน 8 run: `load_model` → `evaluate` → เทียบ `run.prediction` ด้วย `np.allclose` + อ่าน `load_verification.json` | `model.npz`, `preprocessor.npz`, `load_verification.json` | ตาราง 8 แถว คอลัมน์ `stored_passed`, `reproduced_here` | ทุกแถวต้องเป็น True |
| 19 | md | `## 8. เทียบกับ library (parity demo)` | — | — | โจทย์อนุญาต library เพื่อยืนยันผล · scratch ของเราโหลดจากไฟล์ ส่วน sklearn ถูก fit ตรงนี้บนข้อมูลเดียวกัน |
| 20 | code **[parity-demo]** | Parity-demo cell มาตรฐาน + แถว `elastic_net` ที่รายงาน `max_abs_diff` | artifact 2 run | ตาราง | `max_abs_diff` ระดับ 1e-15 = ตรงกันถึงทศนิยมสุดท้ายของ float64 |
| 21 | md | `## 9. ฟีเจอร์ไหนสำคัญ` | — | — | วิธี: สลับลำดับฟีเจอร์ทีละ **ก้อน 20 วัน** แล้วดู RMSE แย่ลงเท่าไร · ข้อจำกัด: ฟีเจอร์ที่สัมพันธ์กันแบ่งความสำคัญกัน |
| 22 | code | `table(REPORT_ROOT / v / "feature_importance.csv")` กรอง `task == "regression"` → pivot mean ± std | `feature_importance.csv` | ตาราง pivot | — |
| 23 | md | `## 10. with_spike เทียบ non_spike` | — | — | คัดลอก `CONFOUND_NOTE` · อธิบายว่า delta ติดลบของ RMSE = ดีขึ้น |
| 24 | code | `comparison = table(REPORT_ROOT / "comparison_a_vs_b.csv")` กรอง regression + `display(Image(FIGURE_ROOT / "comparison_delta_regression.png"))` | `comparison_a_vs_b.csv`, รูป | ตาราง 4 แถว + รูป | ช่วงไหนไม่คร่อมศูนย์ |
| 25 | md | `## 11. สรุป` | — | — | 3 ข้อค้นพบ + 2 ข้อจำกัด (Validation regime เดียว, selection score) |
| 26 | code | provenance: `train_sha256` / `validation_sha256` จาก `config.json` + เวอร์ชัน NumPy/pandas | `config.json` | ตาราง | ใช้ตรวจว่า notebook อ่าน artifact ชุดเดียวกับรายงาน |

- [ ] **Step 2: รัน notebook จากต้นจนจบ**

Run: `.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/03_regression.ipynb`
Expected: `[NbConvertApp] Writing ... bytes to notebooks\03_regression.ipynb` ไม่มี traceback

- [ ] **Step 3: รัน guard เฉพาะเล่มนี้**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notebooks.py -k 03_regression -v`
Expected: PASS — 4 passed

- [ ] **Step 4: Commit (ขออนุมัติก่อน)**

```powershell
git add notebooks/03_regression.ipynb tests/test_notebooks.py
git commit -m "docs: write the regression notebook from the saved artifacts"
```

### Task C5.3: `notebooks/04_classification.ipynb`

**Files:**
- Modify: `notebooks/04_classification.ipynb` (ตอนนี้ 0 cell)

- [ ] **Step 1: สร้าง cell ตามตาราง**

| # | ชนิด | หัวข้อ / จุดประสงค์ | โหลดไฟล์ | แสดง | คำอธิบายไทยที่ต้องมี |
| ---: | --- | --- | --- | --- | --- |
| 1 | md | `# QQQ Volatility Forecasting: Classification and Clustering` | — | — | 13 classifier + 2 clustering เขียนเอง · อ่าน artifact เท่านั้น |
| 2 | code | Setup cell (`TASK = "classification"`) | — | — | — |
| 3 | md | `## 1. โจทย์และความไม่สมดุลของคลาส` | — | — | label = 1 เมื่อ target > Q75 ของ **Train** (0.2531) |
| 4 | code | ตาราง positive rate ของ Train (2 variant) และ Validation | ผ่าน `load` | ตาราง: 25.0% / 18.4% / 9.2% | Validation มี positive แค่ 34 จาก 371 |
| 5 | md | `## 2. ทำไม accuracy ใช้เดี่ยว ๆ ไม่ได้` | — | — | ทาย "ไม่ผันผวน" ทุกวันได้ 337/371 = 90.8% โดยไม่รู้อะไรเลย → ต้องดู AP, ROC-AUC, F1, recall |
| 6 | md | `## 3. อภิธานศัพท์ metric` | — | — | ตาราง: precision · sensitivity (= recall) · specificity (= TNR) · F1 · ROC-AUC · average precision · confusion matrix อ่านอย่างไร — ระบุว่าโจทย์ลิสต์ specificity กับ TNR แยกกันจึงพิมพ์สองคอลัมน์ |
| 7 | code | ตาราง model / selected_candidate / จำนวน candidate / params / preprocess / pca_components | `config.json`, `search_results.json` × 26 | ตาราง 26 แถว | threshold คงที่ 0.5 (probability) หรือ 0 (decision score) ไม่มีการจูน |
| 8 | md | `## 4. Leaderboard บน Validation` | — | — | selection score · ช่วง 95% จาก block bootstrap · กลุ่ม `tied_with_leader` |
| 9 | code | `display(Markdown(...leaderboard_classification.md))` × 2 | `report/<v>/leaderboard_classification.md` | ตาราง 15 แถว × 2 | baseline สองตัวอยู่ตรงไหนของตาราง |
| 10 | code | ตารางเต็มจาก CSV เฉพาะคอลัมน์ที่โจทย์ระบุ: `loss_name, loss, tn, fp, fn, tp, accuracy, precision, sensitivity, specificity, tnr, f1, roc_auc` | `leaderboard_classification.csv` | ตาราง | ครบทุก metric ตามโจทย์หน้า 3 |
| 11 | md | `## 5. ROC และ Precision-Recall` | — | — | ROC: เส้นทแยง = เดาสุ่ม · PR: เส้นประ = positive rate · ทำไม PR บอกความจริงมากกว่าเมื่อคลาสไม่สมดุล |
| 12 | code | `Image(roc_overlay.png)`, `Image(pr_overlay.png)` ทั้งสอง variant | รูป 4 ไฟล์ | รูป | — |
| 13 | md | `## 6. Confusion matrix` | — | — | หลายโมเดลแทบไม่ทำนาย positive เลย → recall ต่ำ |
| 14 | code | `Image(confusion_matrices.png)` × 2 | รูป 2 ไฟล์ | รูป | — |
| 15 | md | `## 7. Performance curve` | — | — | loss ต่อ iteration ของโมเดลที่เทรนวนรอบ · `slp`/`perceptron`/`logistic` จบเพราะหมดโควตา ไม่ใช่ลู่เข้า |
| 16 | code | `Image(performance_curves.png)` × 2 | รูป 2 ไฟล์ | รูป | — |
| 17 | md | `## 8. AdaBoost — โมเดลนอกห้องเรียน` | — | — | SAMME: น้ำหนักตัวอย่าง, error ε_m, α_m = lr·½·ln((1−ε)/ε) · exponential loss ไวต่อ outlier |
| 18 | code | `load_npz(model.npz)` → plot `estimator_errors` และ `estimator_weights` ต่อรอบ ทั้งสอง variant | `adaboost/model.npz` × 2 | กราฟ 2 เส้น × 2 | stump ช่วงหลัง error เข้าใกล้ 0.5 → α เล็กลง |
| 19 | md | `## 9. PCA กับ Random Forest และ SVM` | — | — | โจทย์ขอ dimensionality reduction สำหรับสองตัวนี้ |
| 20 | code | `search_results.json` ของ `random_forest`, `svm` → ตาราง `pca_components` × AP | `search_results.json` × 4 | ตาราง | ตัวไหนชนะด้วย PCA กี่ component |
| 21 | md | `## 10. Clustering: k-Means และ Agglomerative` | — | — | unsupervised: ไม่เห็น label ตอน fit · silhouette คืออะไร · Agglomerative ไม่มี predict ตามธรรมชาติ ใช้ nearest-centroid |
| 22 | code | `display(Markdown(leaderboard_clustering.md))` + crosstab cluster id × label บน Validation จาก `validation_predictions.csv` | `leaderboard_clustering.md`, `validation_predictions.csv` | ตาราง | cluster แยก regime ผันผวนได้แค่ไหน |
| 23 | md | `## 11. Save → load → ทำนายซ้ำ` | — | — | เหมือนเล่ม 03 |
| 24 | code | วน 30 run (13 + 2) × 2: `load_model` → `evaluate` → `np.array_equal` กับ `run.prediction` | `model.npz`, `load_verification.json` | ตาราง 30 แถว | ทุกแถว True |
| 25 | md | `## 12. เทียบกับ library (parity demo)` | — | — | เหมือนเล่ม 03 + หมายเหตุว่า sklearn ใช้ `RandomState` (MT19937) จึงต้องใช้ตัวเดียวกัน |
| 26 | code **[parity-demo]** | Parity-demo cell มาตรฐาน (`adaboost`, `knn`, `gaussian_nb`) | artifact 3 run | ตาราง | — |
| 27 | md | `## 13. ฟีเจอร์ไหนสำคัญ` | — | — | วิธี block-permutation · ค่าติดลบ = สลับแล้วดีขึ้น = ฟีเจอร์นั้นไม่ช่วยบน Validation · อ่านคู่กับ std เสมอ |
| 28 | code | `Image(feature_importance.png)` + ตาราง 3 อันดับแรกต่อโมเดล | `feature_importance.csv`, รูป | รูป + ตาราง | — |
| 29 | md | `## 14. Robustness: walk-forward ภายใน Train` | — | — | **ไม่ใช่การเลือกโมเดลใหม่** · 4 block ครอบ 2019–2023 มี positive ~353 แถว เทียบกับ 34 ใน Validation |
| 30 | code | `walk = table(walk_forward.csv)` → pivot AP ต่อ fold + แถว pooled พร้อมช่วง | `walk_forward.csv` | ตาราง 13 × 5 | อันดับคงที่ข้าม regime หรือไม่ |
| 31 | md | `## 15. สรุป` | — | — | 3 ข้อค้นพบ + ข้อจำกัด |
| 32 | code | provenance เหมือนเล่ม 03 | `config.json` | ตาราง | — |

- [ ] **Step 2: รัน notebook**

Run: `.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/04_classification.ipynb`
Expected: เขียนไฟล์สำเร็จ ไม่มี traceback

- [ ] **Step 3: guard**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notebooks.py -k 04_classification -v`
Expected: PASS — 4 passed

- [ ] **Step 4: Commit (ขออนุมัติก่อน)**

```powershell
git add notebooks/04_classification.ipynb
git commit -m "docs: write the classification and clustering notebook from the saved artifacts"
```

### Task C5.4: `notebooks/05_results_comparison.ipynb` (ใหม่)

**Files:**
- Create: `notebooks/05_results_comparison.ipynb`

ทำหลัง C4.3 (ส่วน Test ต้องมี receipt) ถ้า C4.3 ยังไม่ได้ go-ahead ให้เขียน cell 1–18 ก่อน แล้วเติม 19–23 ภายหลัง

- [ ] **Step 1: สร้าง cell ตามตาราง**

| # | ชนิด | หัวข้อ / จุดประสงค์ | โหลดไฟล์ | แสดง | คำอธิบายไทยที่ต้องมี |
| ---: | --- | --- | --- | --- | --- |
| 1 | md | `# QQQ Volatility Forecasting: Results Comparison` | — | — | เล่มนี้ตอบคำถามวิจัย: นโยบายตัดแถว spike ช่วยหรือทำร้ายโมเดล · และโมเดลไหนดีกว่า baseline จริง |
| 2 | code | Setup cell (`TASK = None`) + `from src.ml.core.bootstrap import block_indices, difference` + `from src.ml.core.report import as_entry, score_entries` | — | — | — |
| 3 | md | `## 1. ภาพรวม 19 โมเดล × 2 variant` | — | — | 38 run · ทุกตัวเขียนเอง |
| 4 | code | รวม leaderboard classification + regression ของสอง variant เป็นตารางเดียว (`pd.concat`) | `leaderboard_*.csv` × 4 | ตาราง | — |
| 5 | md | `## 2. โมเดลไหน "ชนะ" ได้จริง` | — | — | ช่วงของผลต่างคร่อมศูนย์ = ยังบอกไม่ได้ว่าใครดีกว่า · จำนวนโมเดลในกลุ่มนำ |
| 6 | code | นับ `tied_with_leader` ต่อ variant/task + รายชื่อกลุ่มนำ | `leaderboard_*.csv` | ตาราง | ถ้าเกือบทุกตัวอยู่กลุ่มนำ = ข้อมูล 34 positive ไม่พอแยก ไม่ใช่โมเดลเท่ากัน |
| 7 | md | `## 3. ดีกว่าโมเดลในห้องเรียนเท่าไร` | — | — | เทียบ headline กับ `logistic` (classification) และ `multiple_linear` (regression) แบบ paired |
| 8 | code | `score_entries` + `difference` ระหว่าง headline กับ baseline เชิงเส้น บน index ชุดเดียว (`block_indices(371)`) | `validation_predictions.csv` (ผ่าน `read_run`) | ตาราง delta + ช่วง + % improvement | อ้าง "+5" ได้เฉพาะเมื่อช่วงไม่คร่อมศูนย์ |
| 9 | md | `## 4. with_spike ↔ non_spike` | — | — | `CONFOUND_NOTE` เต็ม · ตาราง confound 3 แถว |
| 10 | code | ตาราง confound: rows, positive rate, target max ของ Train สอง variant (คำนวณจาก `load`) | ผ่าน `load` | ตาราง | 1,733 → 1,520 · 25.0% → 18.4% · max 1.50 → 0.55 |
| 11 | code | `display(Markdown(comparison_a_vs_b.md))` + `Image(comparison_delta_classification.png)`, `Image(comparison_delta_regression.png)` | `comparison_a_vs_b.md`, รูป 2 | ตาราง 19 แถว + รูป | — |
| 12 | md | `### อ่านผล A↔B` | — | — | นับ verdict แต่ละแบบ · regression กับ classification ไปทางเดียวกันหรือไม่ · สมมติฐาน AdaBoost (exponential loss) ได้รับการสนับสนุนหรือไม่ — **เขียนตามตัวเลขที่ออกมา** |
| 13 | md | `## 5. Robustness ข้าม regime` | — | — | อันดับบน Validation กับอันดับ walk-forward ตรงกันแค่ไหน |
| 14 | code | join AP Validation กับ AP pooled walk-forward → scatter + `spearman` จาก `src.ml.core.walk_forward` | `leaderboard_classification.csv`, `walk_forward.csv` | scatter + ค่า ρ | ρ ต่ำ = อันดับบน Validation ไม่เสถียร ยิ่งต้องไม่อ้างอันดับ |
| 15 | md | `## 6. ฟีเจอร์ที่โมเดลพึ่งพา` | — | — | เทียบระหว่างโมเดลและระหว่าง variant |
| 16 | code | importance เฉลี่ยข้ามโมเดลต่อฟีเจอร์ ทั้งสอง variant → grouped bar | `feature_importance.csv` × 2 | bar chart | เชื่อมกับ storytelling ของฟีเจอร์ใน notebook 02 |
| 17 | md | `## 7. พลาดตรงไหน (error analysis)` | — | — | positive 20 จาก 34 อยู่ใน 2024Q3 |
| 18 | code | จาก `validation_predictions.csv` ของ headline: นับ TP/FN ต่อไตรมาส + plot score ตามเวลาเทียบ label | `validation_predictions.csv` | ตาราง + กราฟ | โมเดลจับช่วง carry unwind ได้กี่วัน |
| 19 | md | `## 8. ผลบน Test (อ่านครั้งเดียว)` | — | — | เทรนจาก Train · เลือกบน Validation · ไม่ refit · headline ถูกล็อกก่อนเปิด Test |
| 20 | code | `receipt_path = OUTPUT_ROOT / "finalize_receipt.json"`; ถ้าไม่มี → `display(Markdown("ยังไม่ได้ finalize"))`; ถ้ามี → แสดง receipt + `final/<v>/leaderboard_*.md` | `finalize_receipt.json`, `final/**/leaderboard_*.md` | receipt + ตาราง 4 | ตัวเลขนี้คือค่าประมาณที่ไม่ลำเอียงตัวเดียวของโปรเจกต์ |
| 21 | code | (เมื่อมี receipt) scatter Validation AP เทียบ Test AP จากคอลัมน์ `validation_average_precision` + ตาราง headline | `final/<v>/leaderboard_classification.csv` | scatter + ตาราง | จุดใต้เส้นทแยง = Validation มองโลกดีเกินจริง (selection bias) |
| 22 | code | (เมื่อมี receipt) `table(final/segmented_metrics.csv)` → pivot segment × metric ของ headline | `final/segmented_metrics.csv` | ตาราง | spike-affected 54 แถว vs สงบ 319 แถว · ไม่มีช่วงความเชื่อมั่นเพราะแถวน้อย |
| 23 | md | `### อ่านผล Test` | — | — | เขียนตามจริง รวมกรณีที่แย่กว่า Validation |
| 24 | md | `## 9. เทียบกับ library (parity demo)` | — | — | — |
| 25 | code **[parity-demo]** | Parity-demo cell มาตรฐาน (`decision_tree`) | artifact 1 run | ตาราง | — |
| 26 | md | `## 10. ข้อจำกัด` | — | — | ตัวอย่างไม่อิสระ (ESS ≈ 50–200) · Validation regime เดียว · A↔B ถูก confound · importance ของฟีเจอร์ที่สัมพันธ์กัน · สินทรัพย์เดียวช่วงเวลาเดียว |
| 27 | code | provenance: hash ของ lock + receipt + เวอร์ชัน | `selection_lock.json`, `finalize_receipt.json` | ตาราง | — |

- [ ] **Step 2: รัน notebook**

Run: `.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/05_results_comparison.ipynb`
Expected: เขียนไฟล์สำเร็จ ไม่มี traceback

- [ ] **Step 3: guard ทั้งสามเล่ม + รันซ้ำทั้งชุดจาก project root**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notebooks.py -v`
Expected: PASS — 12 passed

Run: `foreach ($n in "03_regression","04_classification","05_results_comparison") { .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --stdout "notebooks/$n.ipynb" | Out-Null; "$n exit=$LASTEXITCODE" }`
Expected: `03_regression exit=0`, `04_classification exit=0`, `05_results_comparison exit=0`

- [ ] **Step 4: Commit (ขออนุมัติก่อน)**

```powershell
git add notebooks/05_results_comparison.ipynb
git commit -m "docs: add the results-comparison notebook with the one-shot Test section"
```

---

# PHASE C6 — Documentation pass (5 ชม.)

**เกณฑ์รับงานของ Phase**

- คนที่ไม่เคยเห็นรีโป clone แล้วทำตาม `README.md` → `docs/guides/RUNBOOK.md` ได้ถึง `pytest -q` เขียวและ `run.py report` exit 0 โดยไม่ต้องถามใคร
- ไม่มีลิงก์ภายในที่ขาด · ไม่มีประโยคที่บอกว่า `report`/`finalize` "ยังไม่ implement"
- model card ทั้ง 19 ใบและ progress index มีผลจริงจากไฟล์ generate (ไม่พิมพ์มือ)
- เอกสารที่ถูกแทนที่อยู่ใน `docs/archive/` พร้อมหัวกระดาษบอกว่าเลิกใช้

### Task C6.1: test ตรวจลิงก์ + เขียน README และ RUNBOOK ใหม่

**Files:**
- Create: `tests/test_docs_links.py`
- Modify: `README.md` (เขียนใหม่ทั้งไฟล์ ~120 บรรทัด), `docs/guides/RUNBOOK.md` §4, §10, §11, `docs/README.md`
- Modify: `AGENTS.md` §8–§9 (แก้เฉพาะข้อความที่ล้าสมัย — ดูตาราง)

- [ ] **Step 1: เขียน test**

```python
# tests/test_docs_links.py
import re

import pytest

from config import PROJECT_ROOT

LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")
DOCUMENTS = sorted(
    path for path in [PROJECT_ROOT / "README.md", PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "CLAUDE.md",
                      *(PROJECT_ROOT / "docs").rglob("*.md")]
    if "archive" not in path.parts and "superpowers" not in path.parts
)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_relative_links_resolve(document):
    broken = [target for target in LINK.findall(document.read_text(encoding="utf-8"))
              if not (document.parent / target).exists()]
    assert broken == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_documents_do_not_claim_report_or_finalize_are_missing(document):
    text = document.read_text(encoding="utf-8")
    for stale in ("ยังไม่ implement", "not implemented yet", "exit 1) — เป็นงาน Phase C"):
        assert stale not in text
```

- [ ] **Step 2: รันดูรายการที่ต้องแก้**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs_links.py -v`
Expected: FAIL บางไฟล์ — ใช้รายการที่ fail เป็น to-do ของ step ถัดไป

- [ ] **Step 3: เขียน `README.md` ใหม่เป็นทางเข้าเดียว** — โครงตามนี้ (เนื้อหาเดิมที่เป็นประวัติย้ายไป `docs/archive/README_2026-10-10.md`)

```markdown
# QQQ Volatility Forecasting
<1 ย่อหน้า: โจทย์, 2 task, คำถามวิจัยเรื่อง spike>

## ผลลัพธ์โดยย่อ
<!-- BEGIN GENERATED: outputs/modeling/final/with_spike/leaderboard_classification.md -->
<!-- END GENERATED -->
<ลิงก์ไป notebook 05 และรายงาน>

## เริ่มใช้งานใน 5 คำสั่ง
<venv → pip install → run_data_pipeline --output-root tmp/phase1-reproduction → pytest -q → run.py report>

## โครงสร้างรีโป            <ตาราง 8 แถว: src/ml, notebooks, outputs/modeling, docs, ...>
## ทำซ้ำผลทั้งหมด            <train --all ×2 → report ×2 → compare → analyze ×2; finalize ห้ามรันซ้ำ>
## โมเดล 19 ตัว               <ตาราง: ชื่อ · task · ไฟล์ · reference · ลิงก์ model card>
## ข้อจำกัดที่ต้องรู้ก่อนอ่านตัวเลข  <5 bullet จาก notebook 05 §10>
## เอกสาร                    <ชี้ docs/README.md>
```

- [ ] **Step 4: แก้ RUNBOOK และ AGENTS ตามตาราง**

| ไฟล์ | ตำแหน่ง | แก้เป็น |
| --- | --- | --- |
| `README.md` | ประโยค "ผู้ที่ clone ใหม่ต้องรับ `data/raw/qqq_daily.csv` แยกต่างหาก" | ลบ — snapshot ถูก track ใน git แล้ว (commit `7690862`) |
| `docs/guides/RUNBOOK.md` §4 | ขั้นตอนขอไฟล์ snapshot | "ไฟล์มากับ clone · ตรวจด้วยคำสั่ง verify ที่มีอยู่" |
| `docs/guides/RUNBOOK.md` §10 | ย่อหน้า "`report` และ `finalize` ยังไม่ implement" | คำสั่ง `report`, `compare`, `analyze`, `finalize --dry-run` พร้อม output ที่คาดหวัง + คำเตือนว่า `finalize --allow-test` ถูกรันแล้วและจะปฏิเสธการรันซ้ำ |
| `docs/guides/RUNBOOK.md` | เพิ่ม §12 "จาก clone เปล่าถึงผลครบ" | ลำดับคำสั่งเต็ม + เวลาโดยประมาณต่อขั้น (วัดจริง) |
| `AGENTS.md` §9 | comment `⚠️ ... report และ finalize ยังไม่ implement` | ลบ comment · เพิ่มคำสั่ง `compare`, `analyze`, `finalize --dry-run` |
| `AGENTS.md` §8 | แผนภาพ branch และตารางแบ่งงาน | เพิ่มบรรทัด "⚠️ แก้ไข 2026-10-12: Phase C ทำร่วมกันบน `feat/report-and-finalize` ไม่มีการแบ่ง owner" — **ไม่ลบตารางเดิม** (เป็นประวัติว่าใครเขียนโมเดลไหน) |
| `AGENTS.md` §7 | บรรทัด "ตอน finalize จึงเพิ่ม ..." | เพิ่ม `selection_lock.json`, `finalize_receipt.json`, `report/`, `final/` |
| `docs/plan/ARCHITECTURE.md` §2, §5 | ตารางโมดูล · "ไม่ทำ cross-validation" | เพิ่ม 6 โมดูลใหม่ · แก้เป็น "ไม่ใช้ CV **เลือก**โมเดล; มีตาราง walk-forward เป็น robustness เท่านั้น" · เพิ่ม ADR-007 (lock + receipt) |
| `docs/plan/REPORT_AND_SLIDES_OUTLINE.md` §0, §9 | path ของ artifact · "ไม่ได้ทำ walk-forward retraining" | path จริงใต้ `outputs/figures/modeling/` และ `outputs/modeling/report/` · แก้ข้อจำกัดให้ตรง |
| `docs/plan/RISKS_AND_DOD.md` §1.2 | checklist | ติ๊กตามหลักฐานจริง |

- [ ] **Step 5: ย้ายเอกสารที่ถูกแทนที่** (ใช้ `git mv` เพื่อเก็บประวัติ) แล้วเพิ่มหัวกระดาษ `> ⚠️ เลิกใช้ 2026-10-12 — เก็บเป็นประวัติ ห้ามใช้เป็นคำสั่งงาน` ในแต่ละไฟล์

```powershell
git mv docs/status/HANDOFF.md docs/archive/HANDOFF.md
git mv docs/status/PROJECT_STATUS.md docs/archive/PROJECT_STATUS.md
git mv docs/audit/REALITY_AUDIT.md docs/archive/REALITY_AUDIT.md
git mv docs/tasks/claude-linear-family.md docs/archive/claude-linear-family.md
git mv docs/tasks/codex-tree-family.md docs/archive/codex-tree-family.md
```

จากนั้นแก้ทุกลิงก์ที่ชี้ไฟล์เหล่านี้ (`CLAUDE.md`, `AGENTS.md` ส่วนหัว, `docs/README.md`, `docs/models/*/MODEL_PROGRESS.md`) — หาด้วย `git grep -n "status/HANDOFF\|status/PROJECT_STATUS\|REALITY_AUDIT\|claude-linear-family\|codex-tree-family"` · `outputs/reports/*.json` ที่อ้าง path เหล่านี้ **ห้ามแก้** (hash ผูกไว้)

- [ ] **Step 6: รันให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs_links.py -v`
Expected: PASS ทุกไฟล์

- [ ] **Step 7: ทดสอบ RUNBOOK จาก clone เปล่า** (ทำหลัง commit งานล่าสุดแล้ว)

```powershell
git clone . ..\qqq-fresh-clone
Set-Location ..\qqq-fresh-clone
```

จากนั้น **เปิด `docs/guides/RUNBOOK.md` ใน clone แล้วพิมพ์ตามทีละคำสั่งตั้งแต่ §2 ถึง §5 และ §12 โดยไม่ใช้ความจำ** — `data/interim/**`, `data/processed/**` และ `tmp/` ถูก gitignore จึงต้องถูกสร้างใหม่จาก snapshot ที่ track ไว้ จุดประสงค์ของ step นี้คือพิสูจน์ว่า RUNBOOK ครบ ไม่ใช่พิสูจน์ว่าเราทำเป็น ปิดท้ายด้วย

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike --report-root tmp/report --figure-root tmp/figures
git status --short
```

Expected: pytest ทั้งหมด passed 0 failed (ข้อมูลที่ regenerate ต้องได้ SHA-256 ตรงกับที่ผูกไว้ ไม่งั้น `load()` จะ raise `Accepted input changed on disk`) · report exit 0 · `git status` ไม่มีไฟล์ที่ track แล้วเปลี่ยน
ถ้าขั้นไหน fail: แก้ RUNBOOK (หรือโค้ด) แล้วทำซ้ำจาก clone ใหม่ — ห้ามข้าม · เสร็จแล้ว `Set-Location` กลับและลบ `..\qqq-fresh-clone`

- [ ] **Step 8: Commit (ขออนุมัติก่อน)**

```powershell
git add README.md AGENTS.md CLAUDE.md docs tests/test_docs_links.py
git commit -m "docs: make the README the single entry point and archive superseded documents"
```

### Task C6.2: ใส่ผลจริงลง model card ด้วย block ที่ generate

**Files:**
- Create: `src/ml/docs_sync.py`, `tests/test_docs_sync.py`
- Modify: `docs/plan/MODEL_CARDS.md`, `docs/tasks/claude-model-progress.md`, `docs/models/*/MODEL_PROGRESS.md` (19 ไฟล์), `README.md`

**Interfaces:**
- Consumes: `read_run`, `run_directory` (C1.1) · `write_markdown_table` (C1.3) · ไฟล์ใต้ `outputs/modeling/report/` และ `outputs/modeling/final/`
- Produces:
  - `MARKER: re.Pattern` — block รูป `<!-- BEGIN GENERATED: <path จาก project root> -->` … `<!-- END GENERATED -->`
  - `write_cards(*, artifact_root=None, report_root=None) -> list[Path]` — เขียน `<report_root>/cards/<model>.md` (default `outputs/modeling/report/cards/`) (ตาราง 2 แถวต่อ variant: params, metric หลักบน Validation, และ metric บน Test เมื่อมี `test_metrics.json`)
  - `sync(document: Path) -> bool` — แทนเนื้อใน block ด้วยเนื้อไฟล์ต้นทาง คืน `True` ถ้าไฟล์เปลี่ยน
  - `python -m src.ml.docs_sync` — เขียน card ทั้ง 19 ใบแล้ว sync เอกสารทุกไฟล์ที่มี marker

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/test_docs_sync.py
import pytest

from config import PROJECT_ROOT
from src.ml.docs_sync import MARKER, sync, write_cards
from src.ml.registry import model_names

DOCUMENTS = [path for path in [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").rglob("*.md")]
             if "archive" not in path.parts and "superpowers" not in path.parts
             and MARKER.search(path.read_text(encoding="utf-8"))]


def test_sync_replaces_only_the_generated_block(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ml.docs_sync.PROJECT_ROOT", tmp_path)
    (tmp_path / "table.md").write_text("| a |\n| --- |\n| 1 |\n", encoding="utf-8")
    document = tmp_path / "doc.md"
    document.write_text("before\n<!-- BEGIN GENERATED: table.md -->\nstale\n<!-- END GENERATED -->\nafter\n",
                        encoding="utf-8")
    assert sync(document) is True
    assert document.read_text(encoding="utf-8") == (
        "before\n<!-- BEGIN GENERATED: table.md -->\n| a |\n| --- |\n| 1 |\n<!-- END GENERATED -->\nafter\n")
    assert sync(document) is False


def test_a_block_pointing_at_a_missing_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ml.docs_sync.PROJECT_ROOT", tmp_path)
    document = tmp_path / "doc.md"
    document.write_text("<!-- BEGIN GENERATED: nope.md -->\n<!-- END GENERATED -->\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        sync(document)


def test_one_card_is_written_per_model(tmp_path):
    written = write_cards(report_root=tmp_path)
    assert sorted(path.stem for path in written) == model_names()
    assert "with_spike" in written[0].read_text(encoding="utf-8")


def test_at_least_the_readme_and_the_model_cards_carry_generated_blocks():
    names = {path.name for path in DOCUMENTS}
    assert {"README.md", "MODEL_CARDS.md"} <= names


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_generated_blocks_match_their_source_files(document):
    for match in MARKER.finditer(document.read_text(encoding="utf-8")):
        source = (PROJECT_ROOT / match["path"]).read_text(encoding="utf-8")
        assert match["body"] == source, f"{document.name}: run `python -m src.ml.docs_sync`"
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.docs_sync'`

- [ ] **Step 3: เขียน implementation**

```python
# src/ml/docs_sync.py
"""Copy generated result tables into the documents that quote them.

Numbers typed by hand drifted from the artifacts once already. A document now quotes a
table by naming its file between two markers, and a test fails when the two differ.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import PROJECT_ROOT

from .core.artifacts import read_run
from .core.contracts import VARIANTS
from .core.figures import write_markdown_table
from .core.trainer import OUTPUT_ROOT
from .registry import MODELS, model_names

MARKER = re.compile(
    r"<!-- BEGIN GENERATED: (?P<path>[^ ]+) -->\n(?P<body>.*?)<!-- END GENERATED -->", re.DOTALL
)
FIELDS = {"regression": ("rmse", "mae", "r2"),
          "classification": ("average_precision", "roc_auc", "f1", "recall", "precision", "specificity", "accuracy"),
          "clustering": ("silhouette", "inertia")}


def _values(metrics: dict, task: str) -> dict:
    source = metrics["clipped_metrics"] if task == "regression" else metrics
    return {field: source[field]["value"] for field in FIELDS[task]}


def write_cards(*, artifact_root=None, report_root=None) -> list[Path]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    report_root = Path(report_root or artifact_root / "report")
    written = []
    for name in model_names():
        task, rows = MODELS[name].task, []
        for variant in VARIANTS:
            run = read_run(artifact_root, variant, name)
            params = json.dumps(run.config["params"], sort_keys=True)
            rows.append({"variant": variant, "split": "validation (selection score)", "params": params,
                         **_values(run.metrics, task)})
            holdout_path = run.directory / "test_metrics.json"
            if holdout_path.is_file():
                holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
                rows.append({"variant": variant, "split": "test (one-shot)", "params": params,
                             **_values(holdout, task)})
        path = report_root / "cards" / f"{name}.md"
        write_markdown_table(rows, path, columns=list(rows[0]), title=f"ผลจริงของ `{name}`")
        written.append(path)
    return written


def sync(document: Path) -> bool:
    document = Path(document)
    text = document.read_text(encoding="utf-8")

    def fill(match: re.Match) -> str:
        body = (PROJECT_ROOT / match["path"]).read_text(encoding="utf-8")
        return f"<!-- BEGIN GENERATED: {match['path']} -->\n{body}<!-- END GENERATED -->"

    updated = MARKER.sub(fill, text)
    if updated != text:
        document.write_text(updated, encoding="utf-8", newline="\n")
    return updated != text


def main() -> int:
    write_cards()
    documents = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]
    for document in documents:
        if "archive" in document.parts or "superpowers" in document.parts:
            continue
        if sync(document):
            print(f"updated {document.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: ใส่ marker ลงเอกสาร**

| เอกสาร | ตำแหน่ง | marker ชี้ไปที่ |
| --- | --- | --- |
| `docs/plan/MODEL_CARDS.md` | ท้าย section ของโมเดลแต่ละตัว (19 จุด) ใต้หัวข้อย่อย `#### ผลจริง` | `outputs/modeling/report/cards/<model>.md` |
| `docs/models/<model>/MODEL_PROGRESS.md` | ท้าย section `## ผล (validation)` (19 ไฟล์) — **ไม่ลบตารางเดิม** ตามกฎ "ห้ามลบผลเก่า" | `outputs/modeling/report/cards/<model>.md` |
| `docs/tasks/claude-model-progress.md` | section ใหม่ `## Leaderboard (generate)` | `outputs/modeling/report/with_spike/leaderboard_classification.md`, `.../leaderboard_regression.md`, `outputs/modeling/report/comparison_a_vs_b.md` |
| `README.md` | `## ผลลัพธ์โดยย่อ` | `outputs/modeling/final/with_spike/leaderboard_classification.md` |

รูปแบบที่ใส่ (บรรทัดว่างระหว่าง marker ไม่ต้องมี — `docs_sync` จะเติมเนื้อให้):

```markdown
<!-- BEGIN GENERATED: outputs/modeling/report/cards/knn.md -->
<!-- END GENERATED -->
```

ในดัชนี `docs/tasks/claude-model-progress.md` เปลี่ยนสถานะ 8 แถวฝั่ง Codex จาก "🟡 … รอ leaderboard" เป็น ✅ (DoD ข้อ 5 ครบแล้วหลัง C1)

- [ ] **Step 5: sync แล้วรัน test**

Run: `.\.venv\Scripts\python.exe -m src.ml.docs_sync; .\.venv\Scripts\python.exe -m pytest tests/test_docs_sync.py tests/test_docs_links.py -v`
Expected: พิมพ์ `updated ...` ราว 22 ไฟล์ · test PASS ทั้งหมด · รัน `docs_sync` ซ้ำอีกครั้งต้องไม่พิมพ์อะไร (idempotent)

- [ ] **Step 6: Commit (ขออนุมัติก่อน)**

```powershell
git add src/ml/docs_sync.py tests/test_docs_sync.py docs README.md outputs/modeling/report/cards
git commit -m "docs: fill the model cards from generated tables and guard them against drift"
```

---

# PHASE C7 — รายงานภาษาไทย + 22 สไลด์ (12 ชม.)

**เกณฑ์รับงานของ Phase**

- `docs/report/REPORT_DRAFT.md` ครบ §1–§10 + ภาคผนวกตาม `REPORT_AND_SLIDES_OUTLINE.md` Part A — "ร่างที่อ่านรู้เรื่อง" ไม่ต้องขัดเกลาสำนวน
- `docs/report/TRACEABILITY.md` มีทุกแถวของเกณฑ์ให้คะแนนในโจทย์หน้า 5 และทุกช่อง "หลักฐาน" ชี้ไฟล์ที่ **มีอยู่จริง**
- ตัวเลขทุกตัวในร่างอยู่ใน block `GENERATED` หรือมี path ของไฟล์ต้นทางกำกับ
- `deliverables/report.docx` และ `deliverables/slides.pptx` (22 สไลด์) เปิดได้ และรูปทุกรูปมาจาก `outputs/figures/`

### Task C7.1: ตาราง traceability เกณฑ์ → หลักฐาน

**Files:**
- Create: `docs/report/TRACEABILITY.md`
- Modify: `tests/test_docs_links.py` (เพิ่ม 1 test)

- [ ] **Step 1: เขียน test ที่บังคับให้ทุกหลักฐานมีอยู่จริง** (ต่อท้าย `tests/test_docs_links.py`)

```python
EVIDENCE = re.compile(r"`((?:outputs|src|notebooks|docs|tests|deliverables)/[^`*<>]+)`")


def test_every_traceability_evidence_path_exists():
    text = (PROJECT_ROOT / "docs" / "report" / "TRACEABILITY.md").read_text(encoding="utf-8")
    paths = EVIDENCE.findall(text)
    assert len(paths) >= 25
    assert [path for path in paths if not (PROJECT_ROOT / path).exists()] == []
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs_links.py -k traceability -v` → Expected: FAIL (`FileNotFoundError`)

- [ ] **Step 2: เขียน `docs/report/TRACEABILITY.md`** — ตารางนี้คือเนื้อหา (path ต้องเป็นไฟล์เดี่ยวที่มีจริง ไม่ใช้ wildcard)

| เกณฑ์ (โจทย์หน้า 5) | คะแนน | รายงาน | สไลด์ | รูป/ตาราง | หลักฐาน |
| --- | ---: | --- | --- | --- | --- |
| 1. Problem setting | 30 (ร่วม) | §1 | S2–S3 | — | `notebooks/02_eda_and_features.ipynb` |
| 2. Feature selection & insight — why ML/DL | 30 (ร่วม) | §2 | S4–S6 | correlation heatmap, importance heatmap | `outputs/reports/feature_report.json`, `outputs/modeling/report/with_spike/feature_importance.csv`, `outputs/figures/modeling/with_spike/feature_importance.png`, `outputs/modeling/report/with_spike/linear_coefficients.csv` |
| 3. Case analysis & data preparation | 30 (ร่วม) | §3 | S7–S9 | split timeline, spike figure | `outputs/reports/data_split_report.json`, `outputs/reports/spike_analysis.json`, `docs/audit/DATA_PIPELINE_AUDIT.md` |
| ≥ 5 features | เงื่อนไข | §2.1 | S4 | ตารางฟีเจอร์ | `src/ml/core/contracts.py` |
| Regression from scratch | 40 (ร่วม) | §5.1 | S10 | — | `src/ml/regression/multiple_linear.py`, `notebooks/03_regression.ipynb` |
| Classification from scratch | 40 (ร่วม) | §5.2 | S11 | — | `src/ml/classification/logistic.py`, `notebooks/04_classification.ipynb` |
| Better regression model + กลไก | 40 (ร่วม) | §6.1 | S12 | coefficient bar | `src/ml/regression/elastic_net.py`, `outputs/modeling/report/cards/elastic_net.md` |
| Better classification model + กลไก | 40 (ร่วม) | §6.2 | S13 | α/ε ต่อรอบ | `src/ml/classification/adaboost.py`, `outputs/modeling/report/cards/adaboost.md` |
| Scratch + ready-to-use ยืนยันผล | 40 (ร่วม) | §5.3 | S14 | ตาราง parity | `src/ml/core/compare.py`, `tests/ml/core/test_compare.py` |
| Training-testing iterations, batch runs | 40 (ร่วม) | §5.4 | S14 | — | `src/ml/core/trainer.py`, `outputs/modeling/with_spike/classification/xgboost/search_results.json` |
| Save-load model แล้ว test | 40 (ร่วม) | §5.5 | S14 | ตาราง reload | `tests/ml/core/test_artifacts.py`, `outputs/modeling/with_spike/classification/knn/load_verification.json` |
| Loss | 30 (ร่วม) | §7.1–7.2 | S15 | คอลัมน์ `loss` | `outputs/modeling/report/with_spike/leaderboard_classification.csv` |
| Confusion matrix | 30 (ร่วม) | §7.2 | S16 | grid 13 โมเดล | `outputs/figures/modeling/with_spike/confusion_matrices.png` |
| Accuracy, Precision, Sensitivity, Specificity, TNR, F1 | 30 (ร่วม) | §7.2 | S16 | leaderboard | `outputs/modeling/report/with_spike/leaderboard_classification.md` |
| ROC & AUC | 30 (ร่วม) | §7.3 | S17 | ROC ซ้อน | `outputs/figures/modeling/with_spike/roc_overlay.png` |
| Performance curve | 30 (ร่วม) | §7.4 | S18 | loss ต่อ iteration | `outputs/figures/modeling/with_spike/performance_curves.png` |
| R-Square | 30 (ร่วม) | §7.1 | S15 | leaderboard regression | `outputs/modeling/report/with_spike/leaderboard_regression.md` |
| Benchmarking, analysis, discussion | 30 (ร่วม) | §8 | S19–S21 | A↔B delta, walk-forward | `outputs/modeling/report/comparison_a_vs_b.md`, `outputs/figures/modeling/comparison_delta_classification.png`, `outputs/modeling/report/with_spike/walk_forward.csv` |
| ผลบน Test (one-shot) | 30 (ร่วม) | §7.6 | S19 | Test leaderboard, segment | `outputs/modeling/finalize_receipt.json`, `outputs/modeling/final/with_spike/leaderboard_classification.md`, `outputs/modeling/final/segmented_metrics.csv` |
| +5 creative problem / data prep | +5 | §1, §3.3 | S8, S22 | spike figure | `outputs/figures/spike_analysis/daily_return_spikes.png` |
| +5 much better than taught models | +5 | §8.3 | S20–S21 | delta vs baseline เชิงเส้น | `notebooks/05_results_comparison.ipynb` |
| Report | deliverable | ทั้งฉบับ | — | — | `deliverables/report.docx` |
| Slides | deliverable | — | 22 แผ่น | — | `deliverables/slides.pptx` |
| .ipynb / .py | deliverable | — | — | — | `notebooks/05_results_comparison.ipynb`, `src/ml/run.py` |

แถว "+5 much better" ต้องมีหมายเหตุใต้ตาราง: อ้างได้เฉพาะเมื่อช่วงของผลต่างไม่คร่อมศูนย์ — ถ้าคร่อม ให้รายงานว่า "ดีกว่าแต่ยังไม่ถึงระดับที่แยกจาก noise ได้"

- [ ] **Step 3: รัน test** (จะผ่านเต็มหลัง C7.3 สร้าง `deliverables/` — ระหว่างนี้ fail เฉพาะ 2 path นั้นถือว่าถูกต้อง)

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs_links.py -k traceability -v`
Expected: FAIL โดย list มีแค่ `deliverables/report.docx` และ `deliverables/slides.pptx`

### Task C7.2: ร่างรายงาน `docs/report/REPORT_DRAFT.md`

**Files:**
- Create: `docs/report/REPORT_DRAFT.md`

- [ ] **Step 1: เขียนตามลำดับคะแนนต่อชั่วโมง** (จาก outline) — แต่ละ section มีข้อกำหนดเนื้อหาขั้นต่ำ

| ลำดับ | Section | เนื้อหาขั้นต่ำ | ตัวเลขมาจาก |
| ---: | --- | --- | --- |
| 1 | §7 ผลการทดลอง | §7.1–7.5 ใช้ block `GENERATED` ของ leaderboard ทั้งสอง variant · ทุกตารางมีประโยค selection score · §7.2 ย่อหน้าแรกอธิบาย majority baseline 337/371 · §7.6 ผล Test + ตาราง segment + ประโยค "อ่านครั้งเดียว ไม่ refit" | `report/<v>/leaderboard_*.md`, `final/<v>/leaderboard_*.md`, `final/segmented_metrics.csv` |
| 2 | §8 Benchmarking | §8.1 กลุ่มนำ (ไม่จัดอันดับภายในกลุ่ม) · §8.2 A↔B ขึ้นต้นด้วย `CONFOUND_NOTE` + ตาราง delta · §8.3 เทียบ baseline เชิงเส้นแบบ paired · §8.4 error analysis ตามไตรมาส · §8.5 (ใหม่) walk-forward robustness + Spearman | `comparison_a_vs_b.md`, `walk_forward.csv`, notebook 05 |
| 3 | §6 โมเดลนอกห้องเรียน | Elastic Net: objective, coordinate descent, soft-threshold, ตัวอย่างคำนวณ 1 รอบ · AdaBoost: SAMME, น้ำหนัก, α, exponential loss · ยกจาก `docs/plan/MODEL_CARDS.md` #4, #15 แล้วเพิ่มผลจริง | `report/cards/elastic_net.md`, `report/cards/adaboost.md` |
| 4 | §2 ฟีเจอร์และเหตุผลที่ใช้ ML | §2.1 ตาราง 8 ฟีเจอร์ · §2.2 correlation + **permutation importance + coefficient** (ส่วนที่โจทย์ hint) พร้อมข้อจำกัด · §2.3 why ML ตอบด้วยผล baseline เทียบโมเดล | `feature_importance.csv`, `linear_coefficients.csv` |
| 5 | §3 การเตรียมข้อมูล | §3.1–3.4 ตาม outline + ข้อความที่ audit บังคับ: simple return (C3), fence จาก `abs(return_1d)` (F2), feature รู้หลังปิดตลาดวัน t (C7), Test ถูกอ่านนอก `src/ml` ที่ไหนบ้าง (E13) | `docs/audit/DATA_PIPELINE_AUDIT.md` |
| 6 | §5 การ implement | §5.1–5.2 ย่อหน้าละโมเดล · §5.3 ตาราง parity 19 แถว (จาก `docs/tasks/claude-model-progress.md` "ห่างจาก library") · §5.4 จำนวน candidate ต่อโมเดล · §5.5 save/load 38 ไฟล์ + test reload · ย่อหน้า "สามโมเดลที่ผลขึ้นกับโควตาการเทรน" | `search_results.json`, `load_verification.json` |
| 7 | §1, §4, §9, §10 | §4.3 protocol: Train → Validation → Test ครั้งเดียว, lock ก่อนเปิด · §9 ข้อจำกัด 6 ข้อ (ESS, regime เดียว, selection bias, confound, importance, สินทรัพย์เดียว) · §10 ตอบคำถามวิจัยตามตัวเลขจริง | — |
| 8 | ภาคผนวก | grid ทั้งหมด · receipt · วิธี reproduce ชี้ RUNBOOK | `configs/modeling.json`, `finalize_receipt.json` |

**กฎการเขียนที่บังคับ** (ตรวจด้วย grep ใน Step 2): ไม่มี "ดีที่สุด/อันดับ 1" สำหรับโมเดลที่ `tied_with_leader = yes` ร่วมกับตัวอื่น · ไม่มี accuracy เดี่ยว ๆ ในประโยคสรุป · ไม่มี "spike ทำให้…" โดยไม่มีคำว่า "นโยบายตัดแถว" ในย่อหน้าเดียวกัน · ไม่มี "ผล Validation พิสูจน์ว่า…"

- [ ] **Step 2: sync ตัวเลขและตรวจคำต้องห้าม**

Run: `.\.venv\Scripts\python.exe -m src.ml.docs_sync; .\.venv\Scripts\python.exe -m pytest tests/test_docs_sync.py tests/test_docs_links.py -q`
Expected: `updated docs/report/REPORT_DRAFT.md` · test passed (ยกเว้น 2 path ของ `deliverables/`)

Run: `git grep -n -E "พิสูจน์ว่า|ดีที่สุดในทุก|accuracy สูงสุด" -- docs/report/REPORT_DRAFT.md`
Expected: ไม่มี output — ถ้ามี ให้อ่านบริบททีละบรรทัดแล้วแก้ถ้าเป็นการอ้างเกินหลักฐาน

- [ ] **Step 3: Commit (ขออนุมัติก่อน)**

```powershell
git add docs/report tests/test_docs_links.py
git commit -m "docs: draft the Thai report and trace every grading criterion to its evidence"
```

### Task C7.3: สร้าง `report.docx` และ `slides.pptx`

**Files:**
- Create: `deliverables/report.docx`, `deliverables/slides.pptx`

- [ ] **Step 1: รายงาน** — invoke skill `anthropic-skills:docx` แปลง `docs/report/REPORT_DRAFT.md` เป็น `deliverables/report.docx`: ฟอนต์ไทยที่อ่านง่าย (TH Sarabun New 16pt หรือเทียบเท่า), หัวข้อมีเลข, สารบัญ, รูปฝังจาก `outputs/figures/**` พร้อมคำบรรยายและเลขรูป, ตารางจาก block `GENERATED` · เครื่องมือของ skill ทำงานนอก `.venv` ของโปรเจกต์ — **ห้าม `pip install` ลง `.venv`**

- [ ] **Step 2: สไลด์** — invoke skill `anthropic-skills:pptx` สร้าง 22 สไลด์ตามตาราง Part B ของ `REPORT_AND_SLIDES_OUTLINE.md` โดยปรับ 4 จุดให้ตรงผลจริง:

| สไลด์ | ปรับ |
| --- | --- |
| S16 | ขึ้นต้นด้วย majority baseline 337/371 ก่อนตาราง metric |
| S19 | leaderboard แสดง "กลุ่มนำ" + ช่วงความเชื่อมั่น ไม่ใช่อันดับ 1–13 · มีคอลัมน์ Test คู่ Validation |
| S20 | bar chart delta พร้อม whisker + ประโยค confound ใต้กราฟ |
| S22 | ข้อจำกัด 3 ข้อ: regime เดียว · ตัวอย่างไม่อิสระ · A↔B ถูก confound |

ห้ามตัด S8, S9, S12, S13, S14, S20 · speaker notes ภาษาไทยทุกสไลด์ · รูปทุกรูปมาจากไฟล์ใน `outputs/figures/` (ไม่วาดใหม่ใน PowerPoint)

- [ ] **Step 3: ตรวจ**

Run: `Get-ChildItem deliverables | Select-Object Name, Length; .\.venv\Scripts\python.exe -m pytest tests/test_docs_links.py -q`
Expected: 2 ไฟล์ ขนาด > 0 · test ทั้งหมด passed (รวม traceability)

เปิดไฟล์ทั้งสองด้วยตา: สารบัญถูก · ไม่มีตารางล้นหน้า · สไลด์ครบ 22 · ตัวเลขใน S15–S20 ตรงกับ `outputs/modeling/report/` (สุ่มเทียบ 5 ค่า)

- [ ] **Step 4: Commit (ขออนุมัติก่อน)**

```powershell
git add deliverables
git commit -m "docs: add the report and the 22-slide deck"
```

### Task C7.4: Final verification ก่อนส่ง (วันที่ 18)

- [ ] **Step 1:** invoke `superpowers:verification-before-completion` แล้วรันและแนบ output จริงของทุกคำสั่ง

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike --report-root tmp/verify/r --figure-root tmp/verify/f
.\.venv\Scripts\python.exe -m src.ml.run finalize --allow-test
foreach ($n in "03_regression","04_classification","05_results_comparison") { .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --stdout "notebooks/$n.ipynb" | Out-Null; "$n exit=$LASTEXITCODE" }
.\.venv\Scripts\python.exe -m src.ml.docs_sync
git status --short
```

Expected: pytest ทั้งหมด passed 0 failed · report exit 0 · **finalize exit 1 พร้อมข้อความ "already evaluated"** (พิสูจน์ว่า guard ยังทำงาน) · notebook 3 เล่ม `exit=0` · `docs_sync` ไม่พิมพ์อะไร · `git status` สะอาด (ยกเว้น `tmp/`)

- [ ] **Step 2:** ไล่ checklist `docs/plan/RISKS_AND_DOD.md` §1.2 ทีละข้อพร้อมหลักฐาน · ไล่คอลัมน์ ✓ ของ `docs/report/TRACEABILITY.md`

- [ ] **Step 3:** สรุปให้เจ้าของ: สิ่งที่ส่ง 4 อย่าง (report, slides, notebooks + `src/`, เอกสารอื่น) และขออนุมัติ merge/PR

---

## Self-Review (ทำแล้วตอนเขียนแผน)

**Spec coverage**

| ข้อกำหนด | Task |
| --- | --- |
| E9 selection bias → CI + ไม่อ้างอันดับ | C1.2, C1.4 (`tied_with_leader`, `SELECTION_NOTE`) |
| E10/D12 ห้าม accuracy เดี่ยว | C1.4 (baseline_majority + คอลัมน์ AP/AUC/F1/recall), C5.3 cell 5, C7.2 กฎการเขียน |
| D13 CI ต้องเป็น block | C1.2 (`test_blocks_widen_the_interval_of_an_autocorrelated_mean`) |
| F13 segment บน Test | C4.1, C4.2 (`segment_metrics`) |
| F15 confound ของ A↔B + paired bootstrap | C2.1 (`CONFOUND_NOTE`, `difference`) |
| Feature importance + coefficient | C3.1 |
| Walk-forward robustness ด้วย fold plan ของ stacking | C3.2 |
| Static guard ของ Test | C4 ตาราง guard, C4.2 Step 5–6 |
| `default_rng` / `permutation(` ถูกห้าม | Global Constraints, C1.2 Step 4, C3.1 |
| finalize: STOP gate, one-shot, hash, segment, final leaderboard | C4.2, C4.3 Step 6–9 |
| notebooks: load only, parity demo, nbconvert | C5.1–C5.4 |
| เอกสารสำหรับคนนอก + snapshot ใน git + archive + model card | C6.1, C6.2 |
| รายงานไทย + 22 สไลด์ + traceability | C7.1–C7.3 |

**Type consistency ที่ตรวจแล้ว:** `Run` ใช้ฟิลด์เดียวกันใน C1.1/C1.4/C2/C4 · entry dict (`model, kind, target, score, prediction`) สร้างที่ `as_entry` และใน `build_report`/`evaluate_holdout` ด้วยคีย์เดียวกัน · `PRIMARY[task] -> (field, higher)` ใช้เหมือนกันใน report/comparison/finalize · `difference(point_a, values_a, point_b, values_b)` คืน b − a ทุกที่ · `evaluate_holdout(holdout, segment_dates, segments, ...)` เรียกจาก `run.py` ด้วยลำดับ argument เดียวกับใน test

**สิ่งที่แผนนี้ตั้งใจไม่ทำ:** refit บน Train+Validation · เลือก hyperparameter ใหม่ · จูน threshold · CI ของ silhouette · แก้ `trainer.py` (คง provenance ของ 38 artifact) · เพิ่ม dependency
