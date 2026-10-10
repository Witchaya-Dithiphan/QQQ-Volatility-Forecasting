# REALITY AUDIT — สถานะจริงของโปรเจกต์ ณ 2026-10-10

> **วิธีอ่านเอกสารนี้:** ทุกตัวเลขมาจากการรันคำสั่งจริงบน `main` ที่ commit `779c78d`
> ไม่มีตัวเลขใดคัดลอกมาจากเอกสารเดิม เอกสารเดิมหลายฉบับรายงานค่าที่พิสูจน์ซ้ำไม่ได้
> ซึ่งเป็นเหตุผลที่ต้องทำ audit นี้ตั้งแต่แรก
>
> เกณฑ์: ✅ ตรวจแล้วจริง · ⚠️ อ้างแต่ไม่มีหลักฐาน · ❌ ไม่มีอยู่จริง · 🔁 เอกสารขัดกับโค้ด

---

## 1. สรุปสำหรับคนที่อ่านบรรทัดเดียว

**ชั้นข้อมูลเสร็จจริงและแข็งแรง · ชั้นโมเดลเป็นศูนย์สมบูรณ์ · infrastructure ที่มีอยู่แก้ปัญหาผิดข้อ**

| ด้าน | สถานะ |
| --- | --- |
| Data pipeline (Phase 1 + 2) | ✅ เสร็จ ตรวจ SHA-256 ตรงทุกไฟล์ |
| Environment Python 3.14.8 | ✅ ใช้งานได้ 69 distributions |
| Test suite | ✅ **459 passed, 3 skipped, 0 failed** |
| โมเดล 19 ตัว | ❌ **0/19** |
| Notebook 03/04 | ❌ 0 cells ทั้งคู่ |
| `outputs/models`, `outputs/metrics` | ❌ ว่างเปล่า — ไม่เคยมีโมเดลถูกเทรนเลย |
| `src/modeling/` 1,930 บรรทัด | ⚠️ มีของใช้ได้ ~300 บรรทัด ที่เหลือเป็น governance → เลิกใช้ (ADR-001) |

---

## 2. ตัวเลขที่ยืนยันแล้วว่าถูกต้อง ✅

รันจริงด้วย `.\.venv\Scripts\python.exe`

| รายการ | ค่า | วิธีตรวจ |
| --- | --- | --- |
| Raw snapshot | 2,512 rows · 2016-08-31 → 2026-08-28 · SHA-256 `649e1db1…b6b13` | `src/download_qqq_data.py verify-snapshot` |
| Split | train 1,733 · validation 371 · test 373 | `run_data_pipeline` |
| Q75 threshold | `0.2530580184684854` | `classification_threshold.json` |
| Class balance (train) | 1,300 / 433 | อ่านจาก CSV |
| Spike threshold | `0.0465436445413787` | `run_spike_analysis` |
| Non-spike train | 1,520 rows (ตัดออก 213) | `experiments/non_spike/train.csv` |
| With-spike train | byte-identical กับ Original Train (`424F1BDA…4E64C`) | เทียบ SHA-256 |
| Config seal | `Modeling config and generated plan block match` | `config_sync --check` |
| Dependency gate | ผ่าน รายงาน `"python": "3.14.8"`, 69 locked distributions | `dependency_smoke` |
| Feature condition number | **35.6** (>30 = multicollinearity ชัด) | คำนวณจาก train set |
| `hist_vol_5d ~ hist_vol_20d` | \|r\| = **0.804** | คำนวณจาก train set |
| Regression target | skew **2.80** · kurtosis **16.26** | คำนวณจาก train set |

สองแถวท้ายเป็นข้อมูลใหม่ที่ audit นี้สร้างขึ้น และเป็นฐานของ ADR-004

---

## 3. สิ่งที่ไม่มีอยู่จริง ❌

| สิ่งที่เอกสารพูดถึง | ความจริง |
| --- | --- |
| โมเดล 19 ตัว | `src/models/__init__.py`, `regression.py`, `classification.py` = **3 บรรทัดรวมกัน** มีแต่ docstring |
| `notebooks/03_regression.ipynb` | 0 cells (ไฟล์ 234 bytes = skeleton เปล่า) |
| `notebooks/04_classification.ipynb` | 0 cells |
| `outputs/models/`, `outputs/metrics/` | มีแต่ `.gitkeep` → **ไม่เคยมีโมเดลใดถูกเทรนและบันทึก** |
| `src/modeling/registry.py`, `selection.py` | ไม่มี (แผนระบุเป็น Planned ถูกต้อง) |
| training/evaluation CLI | `runner.py:73-74` บล็อก `train/tune/evaluate/finalize-test` ด้วย `parser.error()` มีแต่ `check-config`/`check-data` |
| figure writer สำหรับ ROC/confusion/performance curve | ไม่มี matplotlib ใน `src/modeling/` เลยสักบรรทัด |
| leaderboard / results table | ไม่มี |
| scratch-vs-reference comparison harness | ไม่มี มีแต่ข้อความบรรยายใน config |

`outputs/.test_ledger.json` มี 2 entries ที่ `status: "intent"`, `run_id: "synthetic"` ซึ่งเป็น
**side-effect จาก test** (`test_finalize_gate.py` ไม่ได้ monkeypatch `_test_ledger_path`)
ไม่ใช่หลักฐานว่าเคยเทรนโมเดล และไฟล์นี้ถูก commit เข้า git ทั้งที่ `.gitignore` ของโปรเจกต์เองห้ามไว้

---

## 4. เอกสารที่ขัดกับโค้ดหรือขัดกันเอง 🔁

| เรื่อง | ขัดกันอย่างไร | ความจริง |
| --- | --- | --- |
| จำนวน classifier | `README.md:210` บอก 13 · `README.md:257`, `PROJECT_STATUS.md:57,338`, `HANDOFF.md:315` บอก 12 · `HANDOFF.md:302` บอก 13 | **13** (`configs/modeling.json` มี 19 entries = reg 4 / cls 13 / clustering 2) |
| จำนวน test | ทุกฉบับรายงาน `334 passed, 2 skipped` | **459 passed, 3 skipped** — เลข 334 เป็นค่าก่อน commit M2 |
| commit baseline | `PROJECT_STATUS.md:6`, `MODEL_TRAINING_PLAN.md:16` อ้าง `03dfada` | HEAD จริงคือ `779c78d` |
| Deadline | `MODEL_TRAINING_PLAN.md:5,46,426` ระบุ 12 ต.ค. ตาม PDF | **18 ต.ค.** (มีประกาศเลื่อน — ยืนยันโดยเจ้าของโปรเจกต์ 2026-10-10) |
| คำสั่งสร้าง venv | `HANDOFF.md:153,164` สั่ง `py -3.11 -m venv .venv` | ทำตามแล้ว **test fail ทันที** เพราะ gate บังคับ 3.14.8 |
| สถานะ M2 | `PROJECT_STATUS.md:355`, `HANDOFF.md:440` สั่งให้ "เริ่ม implement shared loader/metrics/persistence" | **ทำเสร็จไปแล้ว** ใน commit `0e8b360` ไม่มีเอกสารใดบันทึก |
| lockfile | `HANDOFF.md:125,141-143` บอก "ไม่ pin version" และ "ไม่มี lockfile" | `requirements.txt` เป็น shim ชี้ไป `requirements-lock.txt` ที่ pin ครบ 69 ตัว |
| `scikit-learn` | `HANDOFF.md:292` บอก "ไม่พบ dependency" | มีใน `requirements.in:9-11` พร้อม `xgboost`, `joblib` และติดตั้งแล้ว |

เอกสารเหล่านี้จะถูกแก้ในรอบเดียวกับที่ merge แผนนี้ โดยยึดหลัก **โค้ดคือความจริง เอกสารตามโค้ด**

---

## 5. กับดัก 2 อย่างที่เสียเวลาไปแล้วและจะเสียอีกถ้าไม่บันทึกไว้

### 5.1 Cross-drive pytest — ไม่ใช่ bug

โปรเจกต์อยู่ไดรฟ์ `D:` แต่ pytest สร้าง `tmp_path` บน `C:` ทำให้ 5 tests fail ด้วย
`ValueError: Report and referenced artifact must be on the same drive` เพราะ
`src/report_paths.py:23` ใช้ `os.path.relpath` ซึ่งคำนวณ path ข้ามไดรฟ์บน Windows ไม่ได้

**พิสูจน์:** รันด้วย `--basetemp=D:/tmp_pytest` แล้ว failure หายทันที 5 ข้อ
**แก้ถาวร:** ADR-006 → ตั้ง `--basetemp=.pytest_tmp` ใน `pytest.ini`

### 5.2 Provenance ของ Phase 1 reproduction

`outputs/reports/spike_input_contract.json` เก็บ
`saved_report_provenance.isolated_phase1_reproduction.output_root` เป็น path สัมพัทธ์
`run_spike_analysis` จะอ่านค่านี้เพื่อหา reproduction root ถ้าโฟลเดอร์นั้นหายไป
preflight จะ fail ด้วย `Missing Phase 1 reproduction root`

เหตุการณ์จริง: ระหว่างงาน re-freeze Python มีการรันด้วย `--reproduced-root tmp/py314check`
แล้วลบโฟลเดอร์ทิ้ง ทำให้ contract ชี้ไปที่ไม่มีอยู่ → test fail

**แก้แล้ว:** rerun ด้วยชื่อตามคู่มือ `tmp/phase1-reproduction` (RUNBOOK §5.2)
**กฎ:** ใช้ชื่อนี้เท่านั้น และห้ามลบทิ้ง

---

## 6. หมายเหตุเรื่องความน่าเชื่อถือของ audit นี้เอง

audit ทำด้วย agent 4 สายขนานกัน และ **สองสายรายงานขัดกันเรื่อง environment** — สายหนึ่งบอกว่า
`.venv` มีแค่ 35/69 packages และรัน pytest ไม่ได้ อีกสายบอกว่าติดตั้งครบและ 453 tests ผ่าน

สาเหตุคือ race condition ของตัว audit เอง: สายที่ตรวจ test เป็นผู้สร้าง `.venv` และติดตั้ง
dependencies ขณะที่อีกสองสายกำลังอ่าน `.venv` ที่ยังติดตั้งไม่เสร็จ

**ข้อสรุปที่ถูกต้องคือฝั่งที่รันคำสั่งหลังติดตั้งเสร็จ** และยืนยันซ้ำแล้วในเอกสารนี้ว่า
environment ใช้งานได้จริง บันทึกไว้เพื่อให้คนที่อ่าน transcript เดิมไม่สับสน และเป็นบทเรียนว่า
agent ที่รันขนานกันต้องไม่แตะ resource เดียวกัน
