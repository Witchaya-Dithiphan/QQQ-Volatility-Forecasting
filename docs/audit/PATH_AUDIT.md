# PATH AUDIT — 2026-10-11

> ส่วนหนึ่งของ [`DATA_PIPELINE_AUDIT.md`](DATA_PIPELINE_AUDIT.md)

path constant ทั้ง 39 ตัวใน `config.py` เป็น absolute ที่สร้างจาก `PROJECT_ROOT = Path(__file__).resolve().parent`
ไม่พบ path ที่ hard-code `C:\Users\...` หรือใช้ backslash อย่างเดียวใน `src/**`, `configs/`, `pytest.ini`, `tests/**`

| Path | นิยามที่ | มีจริง | ใช้โดย | คำตัดสิน |
| --- | --- | --- | --- | --- |
| constant 35 ตัวของ data / report / figure | `config.py:7-67` | มี | data modules, `src/ml/core/contracts.py`, `data.py` | OK |
| `data/raw/qqq_daily_latest.csv` | `config.py:9` | ไม่มี | `download_qqq_data.py` (refresh เท่านั้น) | OK เกิดเมื่อ refresh จริง |
| `outputs/reports/qqq_download_report.json` | `config.py:10` | ไม่มี | `download_qqq_data.py` | OK เหตุผลเดียวกัน |
| `"data/raw/qqq_daily.csv"` (relative ต่อ cwd) | `src/load_data.py:50` | มีเมื่อ cwd เป็น root | `load_qqq_data()` | 🟡 ซ้ำกับ `RAW_DATA_PATH` |
| `outputs/modeling` | `src/ml/core/trainer.py:23` | มี | trainer | OK แต่อยู่นอก `config.py` |
| `configs/modeling.json` | `src/ml/run.py:30`, `trainer.py:54` | มี | `src/ml` | OK |
| `src/modeling/**/*.py`, `src/models/**/*.py` | `configs/modeling.json:695-696` | ไม่มี | ไม่พบผู้อ่าน | 🟠 glob ล้าสมัย |
| `outputs/modeling/<variant>/<task>/<model>/<implementation>/<run_id>/` | `configs/modeling.json:700` | ไม่มี | ไม่มี | 🟡 layout จริงคือ `<variant>/<task>/<model>/` |
| `../../tmp/phase1-reproduction` | `outputs/reports/spike_input_contract.json:177` | มี (14 ไฟล์ เหมือนชุดตรวจรับ) | preflight ของ Phase 2 | 🟠 report ที่ track พึ่งโฟลเดอร์ที่ ignore |
| `outputs/models/`, `outputs/metrics/` | `.gitignore` | มีแค่ `.gitkeep` | ไม่มี | 🟡 ตาย |
| `--basetemp=.pytest_tmp` | `pytest.ini:4` | มี (ignore) | pytest | OK อยู่ไดรฟ์เดียวกับโปรเจกต์ |
| `C:\old-machine\...` | `tests/test_experiment_datasets.py:87-115` | — | fixture path ที่ไม่ portable โดยตั้งใจ | OK |
| `C:\Users\ADMIN\AppData\...` | output ของ notebook 01, 02 | — | ข้อความ stderr | 🟡 path ของเครื่องอื่นค้างใน output |

## Line ending กับ checksum

- ระบบตั้ง `core.autocrlf=true` · `.gitattributes` เดิมบังคับ LF เฉพาะ `*.py`, `*.json`, `*.md`, requirements
- **แก้แล้ว (X1):** เพิ่ม `*.csv text eol=lf`, `*.npz binary`, `*.png binary` ก่อนหน้านี้ `outputs/reports/spike_event_audit.csv`
  ถูก checkout เป็น CRLF บน Windows clone ใหม่ ทำให้ SHA ไม่ตรงกับที่เก็บใน `spike_analysis.json:509`
- **ยังเป็นข้อจำกัด (X2):** CSV ของ Phase 1 เขียนด้วย `to_csv()` ที่ไม่ระบุ `lineterminator` จึงเป็น CRLF บน Windows
  hash ใน report และใน `config.json` ของโมเดล 38 ชุดจึง **ใช้ได้บน Windows เท่านั้น** การแก้ต้อง regenerate
  ข้อมูลและ artifact ทั้งหมด จึงไม่ทำ
- Raw snapshot อยู่ใน git แล้วตั้งแต่ `7690862` ด้วย attribute `-text` (CRLF, 124,154 bytes, ตรง manifest) git จึงไม่แปลง line ending ของมัน

## Requirements

`requirements.txt` ชี้ไป `requirements-lock.txt` (69 pin ตรงกับ `.venv` ทุกตัว, `pip check` สะอาด, Python 3.14.8)
ข้อสังเกต: `pypdf` ติดตั้งแต่ไม่อยู่ใน lock · `scipy` ถูก import ใน `tests/ml` แต่ไม่ประกาศใน `requirements.in`
