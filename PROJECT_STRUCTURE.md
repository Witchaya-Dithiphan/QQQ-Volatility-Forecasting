# PROJECT STRUCTURE — โครงสร้างเป้าหมายและแผนการย้าย

> เอกสารนี้บอกว่าไฟล์ไหน **เก็บ / ย้าย / ลบ** และหน้าตาปลายทางเป็นอย่างไร
> เหตุผลเบื้องหลังอยู่ใน `ARCHITECTURE.md` (ADR-001)

---

## 1. โครงสร้างเป้าหมาย

```
QQQ-Volatility-Forecasting/
├── AGENTS.md                    ★ contract กลาง — Codex และ Claude อ่านไฟล์นี้
├── CLAUDE.md                    ★ ชี้ไป AGENTS.md (ไม่มีเนื้อหาซ้ำ)
├── ARCHITECTURE.md              ★ ดีไซน์ + ADR
├── MODEL_CARDS.md               ★ สเปก 19 โมเดล
├── ROADMAP.md                   ★ แผนรายวันถึง 18 ต.ค.
├── RISKS_AND_DOD.md             ★ ความเสี่ยง + definition of done
├── REPORT_AND_SLIDES_OUTLINE.md ★ โครงรายงาน/สไลด์ ผูกกับ rubric
├── README.md                    ปรับให้ตรงความจริง
├── RUNBOOK.md                   ปรับ: เพิ่มคำสั่งชั้น modeling
├── PROJECT_STATUS.md            แหล่งสถานะเดียว (ฉบับอื่นชี้มาที่นี่)
├── HANDOFF.md                   แก้ข้อที่ขัดกับโค้ด
├── PHASE2_SPIKE_READINESS.md    คงไว้ (หลักฐาน Phase 2)
├── docs/
│   ├── audit/REALITY_AUDIT.md   ★ สถานะจริง 2026-10-10
│   └── superpowers/specs/       ★ design spec
│
├── configs/modeling.json        คงไว้ — ใช้เป็น "ข้อมูล" (grid + inventory) ไม่ใช่ schema ที่ต้อง validate
│
├── src/
│   ├── config.py                คงไว้
│   ├── <data layer 13 ไฟล์>     🔒 ห้ามแตะ
│   └── ml/                      ★ บ้านใหม่ทั้งหมด
│       ├── core/
│       │   ├── base.py          BaseModel interface
│       │   ├── data.py          load(variant) / load_test(allow_test=)
│       │   ├── preprocess.py    Standardizer + PCA        ← ย้ายมา
│       │   ├── metrics.py       metrics + losses + curves ← ย้ายมา
│       │   ├── persist.py       save/load npz + verify    ← ย้ายมา
│       │   ├── trainer.py       grid search + artifacts
│       │   ├── compare.py       strict parity
│       │   ├── figures.py       ROC / confusion / perf curve / leaderboard
│       │   └── registry.py      ชื่อโมเดล → คลาส
│       ├── regression/          linear · multiple · polynomial · elastic_net
│       ├── classification/      13 ไฟล์ (1 โมเดล 1 ไฟล์)
│       ├── clustering/          kmeans · agglomerative
│       └── run.py               CLI: train / report / finalize
│
├── tests/
│   ├── <data layer tests>       🔒 คงไว้ทั้งหมด (387 tests)
│   └── ml/                      ★ mirror โครงสร้าง src/ml/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb   คงไว้ (21 cells, รันแล้ว)
│   ├── 02_eda_and_features.ipynb คงไว้ (32 cells, รันแล้ว)
│   ├── 03_regression.ipynb      ★ เขียนใหม่จาก 0 cells
│   ├── 04_classification.ipynb  ★ เขียนใหม่จาก 0 cells
│   └── 05_results_comparison.ipynb ★ ใหม่ — paired A↔B + leaderboard
│
└── outputs/
    ├── reports/ figures/ metrics/ models/   (gitignored)
    └── modeling/<variant>/<task>/<model>/   ★ artifact ต่อ run
```

★ = สร้างใหม่ในรอบนี้ · 🔒 = ห้ามแก้

---

## 2. ตารางย้าย/ลบ

### 2.1 ย้ายมาใช้ต่อ (โค้ดคณิตศาสตร์ที่มี test ผ่านแล้ว)

| จาก | ไป | หมายเหตุ |
| --- | --- | --- |
| `src/modeling/preprocessing.py` (115 บรรทัด) | `src/ml/core/preprocess.py` | ย้ายตรง ๆ ไม่แก้ตรรกะ |
| `src/modeling/metrics.py` (153 บรรทัด) | `src/ml/core/metrics.py` | ย้ายตรง ๆ ไม่แก้ตรรกะ |
| `src/modeling/persistence.py` บรรทัด 31–83 | `src/ml/core/persist.py` | เอาเฉพาะ `save_npz`, `load_npz`, `save_reference`, `load_reference`, `verify_reload`; เปลี่ยน `open("xb")` → `open("wb")` เพื่อให้รันซ้ำได้ |
| `src/modeling/datasets.py` (140 บรรทัด) | `src/ml/core/data.py` | เก็บการตรวจ schema/SHA-256/chronology; ตัด `_load_reports()` identity chain และ `FinalizeTestAuthorization` |
| `src/modeling/contracts.py` (9 บรรทัด) | รวมเข้า `src/ml/core/data.py` | สั้นเกินกว่าจะแยกไฟล์ |
| `tests/modeling/test_preprocessing.py` | `tests/ml/core/test_preprocess.py` | ย้ายพร้อมโค้ด |
| `tests/modeling/test_metrics.py` | `tests/ml/core/test_metrics.py` | ย้ายพร้อมโค้ด |
| `tests/modeling/test_datasets.py` | `tests/ml/core/test_data.py` | ปรับให้เข้ากับ loader ใหม่ |

### 2.2 ลบทิ้ง

| ไฟล์ | บรรทัด | เหตุผล |
| --- | ---: | --- |
| `src/modeling/config_schema.py` | 984 | validate config ที่โค้ดอ่านจริงเพียง ~29 path |
| `src/modeling/configuration.py` | 116 | integrity seal ทำให้แก้ grid ทีเดียวต้อง re-seal 3 ชั้น |
| `src/modeling/config_sync.py` | 63 | sync JSON → markdown ไม่มีผลต่อผลลัพธ์ ML |
| `src/modeling/expected_runs.py` | 61 | สร้างใบรายการ 688 runs สำหรับงานที่ต้องการ 38 |
| `src/modeling/dependency_smoke.py` | 37 | บังคับเวอร์ชันเป๊ะทุกแพ็กเกจ ทำให้ environment เปราะโดยไม่มีคะแนนผูก |
| `src/modeling/artifacts.py` | 82 | เอาเฉพาะแนวคิด `write_json`; `code_snapshot_hash` และ `runtime_fingerprint` คือต้นเหตุ deadlock |
| `src/modeling/persistence.py` บรรทัด 86–339 | 253 | lifecycle / resume / finalize gate / test ledger |
| `src/modeling/runner.py` | 84 | CLI ที่ปฏิเสธทุกคำสั่งที่ต้องใช้จริง |
| `src/models/` ทั้งโฟลเดอร์ | 3 | stub ว่าง ถูกแทนด้วย `src/ml/` |
| `tests/modeling/test_m2_fixes.py` | 224 | มี 3 `pytest.skip` placeholder และ assertion ที่ทดสอบว่า `make_manifest` คืนค่าที่เพิ่งส่งเข้าไป |
| `tests/modeling/test_config.py`, `test_expected_runs.py`, `test_finalize_gate.py`, `test_runner.py`, `test_artifacts.py`, `test_persistence.py`, `test_dependencies.py` | ~580 | ทดสอบ machinery ที่ถูกลบ (ส่วน save/load ของ `test_persistence.py` ย้ายไป `tests/ml/core/test_persist.py`) |
| `outputs/.test_ledger.json` | — | `git rm --cached` — ถูก commit ผิดกฎ `.gitignore` ของโปรเจกต์เอง |

**รวมลบ ~1,630 บรรทัด production + ~1,100 บรรทัด test ของ governance layer**

### 2.3 ห้ามแตะเด็ดขาด 🔒

`data/**` · `src/clean_data.py` · `build_features.py` · `build_targets.py` · `split_data.py` ·
`detect_spikes.py` · `audit_spikes.py` · `build_experiment_datasets.py` · `build_spike_reports.py` ·
`build_spike_input_contract.py` · `spike_contract.py` · `load_data.py` · `download_qqq_data.py` ·
`report_paths.py` · `run_data_pipeline.py` · `run_spike_analysis.py` · `tests/test_*.py` ของชั้นข้อมูล

---

## 3. ลำดับการย้าย (ทำบน `feat/ml-core` ก่อนแตกสายงาน)

1. สร้าง `src/ml/core/` + `tests/ml/` แล้วย้าย 3 ไฟล์คณิตศาสตร์พร้อม test → รัน `pytest -q` ต้องเขียว
2. เขียน `base.py`, `data.py`, `compare.py` + test
3. เขียน `trainer.py`, `registry.py`, `run.py` + test
4. เขียน `figures.py` + test
5. **pilot:** Multiple Linear Regression ครบลูป (train → grid → save → load → verify → figure → leaderboard)
   เพื่อพิสูจน์ว่า framework ใช้ได้จริงก่อนให้ 2 คนลงมือขนาน
6. ลบ `src/modeling/`, `src/models/`, tests ของ governance → รัน `pytest -q` ต้องเขียว
7. ตั้ง `pytest.ini` ให้ `--basetemp=.pytest_tmp` (ADR-006) + เพิ่ม `.pytest_tmp/` ใน `.gitignore`
8. merge เข้า `main` → แจ้งทั้งทีมให้ rebase แล้วจึงเริ่มเขียนโมเดล

**ห้ามเริ่มขั้นที่ 8 ก่อนที่ pilot ในขั้น 5 จะผ่านจริง** — ถ้า framework ผิด การแก้หลังจาก 2 คน
เขียนโมเดลไปแล้วจะกระทบทั้งสอง branch พร้อมกัน

---

## 4. กติกาเรื่องไฟล์ระหว่าง 2 branch

- ไฟล์ใน `src/ml/core/**` **freeze หลัง merge** — ถ้าจำเป็นต้องแก้ ให้แจ้งอีกฝั่งก่อนเสมอ
  และแก้บน `main` ไม่ใช่บน branch โมเดล
- `src/ml/registry.py` เป็นจุดเดียวที่ทั้งสอง branch ต้องแก้ร่วมกัน (เพิ่มชื่อโมเดล)
  → ออกแบบให้เป็น dict ที่เพิ่มบรรทัดเดียวต่อโมเดล เรียงตามตัวอักษร เพื่อให้ conflict แก้ง่าย
- `MODEL_CARDS.md` ก็แก้ร่วมกัน → ใช้กฎเดียวกัน คนละหัวข้อ ไม่แก้ของกันและกัน
