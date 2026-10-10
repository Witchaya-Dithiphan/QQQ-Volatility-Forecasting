# TASKS — Core Framework + Linear / Probabilistic / Clustering Family (11 โมเดล)

> **เอกสารนี้สำหรับผู้พัฒนาที่ใช้ Claude Code** · branch `feat/ml-core` → `feat/models-linear-family`
>
> ฝั่งนี้รับผิดชอบ **core framework ด้วย** ซึ่งเป็นคอขวดของทั้งทีม **Phase A สำคัญกว่าโมเดลตัวใดก็ตาม**
>
> **A0 ถูกแยกออกมาทำก่อนเป็นอย่างแรกแล้ว** — ตรวจ dependency จริงแล้วพบว่าเพื่อนต้องการแค่
> `core/base.py` + `core/compare.py` (~110 บรรทัด) ไม่ใช่ core ทั้งก้อน ส่ง 2 ไฟล์นี้ก่อน
> ทำให้เขาเริ่ม T1 ได้ทันทีแทนที่จะรอถึงวันที่ 12 — ได้เวลาขนานเพิ่ม 2 วัน
>
> อ่านก่อน: [`AGENTS.md`](../../AGENTS.md) · [`docs/plan/MODEL_CARDS.md`](../plan/MODEL_CARDS.md) ·
> แผนละเอียดพร้อมโค้ดจริง: [`docs/superpowers/plans/2026-10-10-ml-core-framework.md`](../superpowers/plans/2026-10-10-ml-core-framework.md)

---

## ภาพรวม 3 phase

| Phase | งาน | ทำไมเรียงแบบนี้ |
| --- | --- | --- |
| **A** | core framework + pilot `multiple_linear` | ทั้งทีมรอ · ต้อง merge ภายใน **12 ต.ค.** |
| **B** | 10 โมเดลที่เหลือของฝั่งนี้ | ขนานกับเพื่อนได้เต็มที่ |
| **C** | `report` / `finalize` CLI + variant B + figures | ต้องรอโมเดลครบทั้งสองฝั่ง |

---

## PHASE A — core framework (คอขวด ห้ามเพิ่ม scope)

แผนรายขั้นพร้อมโค้ดเต็มอยู่ใน
[`docs/superpowers/plans/2026-10-10-ml-core-framework.md`](../superpowers/plans/2026-10-10-ml-core-framework.md)
ที่นี่สรุปเฉพาะลำดับและเกณฑ์ผ่าน

```bat
git checkout -b feat/ml-core
```

| # | Task | ไฟล์ | เกณฑ์ผ่าน |
| --- | --- | --- | --- |
| **A0** | **unblock set — ทำก่อนทุกอย่าง** | `core/base.py`, `core/compare.py`, `__init__.py` ของ package | 21 tests ผ่าน · merge + push + แจ้งเพื่อนทันที |
| A1 | ย้ายโค้ดคณิตที่ test ผ่านแล้ว | `core/preprocess.py`, `core/metrics.py`, `core/contracts.py` | `pytest tests/ml/core` เขียว (20 tests ที่ย้ายมา) |
| A2 | save / load / verify | `core/persist.py` | 5 tests ผ่าน · save ทับไฟล์เดิมได้ |
| A3 | loader 2 variant + Test แบบ default-deny | `core/data.py` | 8 tests ผ่าน · `load_test()` raise `PermissionError` |
| A4 | `BaseModel` interface | `core/base.py` | 5 tests ผ่าน · instantiate ตรง ๆ ไม่ได้ |
| A5 | strict parity harness | `core/compare.py` | 6 tests ผ่าน · error message บอก index ที่ต่างมากสุด |
| A6 | registry | `registry.py` | 5 tests ผ่าน · ชื่อเรียงตามตัวอักษร |
| A7 | **pilot `multiple_linear`** | `regression/multiple_linear.py` | parity กับ `LinearRegression` ผ่าน |
| A8 | trainer + artifacts | `core/trainer.py` | 5 tests ผ่าน · ผลิต artifact ครบ 7 ไฟล์ |
| A9 | figures + leaderboard | `core/figures.py` | 4 tests ผ่าน · ไฟล์ PNG มีขนาด > 0 |
| A10 | CLI | `run.py` | `train` ใช้ได้ · `report`/`finalize` คืน exit 1 พร้อมเหตุผล |
| A11 | ลบ `src/modeling/` + ตั้ง `pytest.ini` basetemp | — | `pytest -q` เขียว · ไม่มี `src.modeling` หลงเหลือ |
| A12 | อัปเดตเอกสารด้วยผลรันจริง + PR | `PROJECT_STATUS.md`, `RUNBOOK.md` | merge เข้า `main` |

**Gate ของ Phase A — ห้าม merge ถ้าข้อใดข้อหนึ่งไม่ผ่าน**

```bat
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple_linear --variant with_spike
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple_linear --variant non_spike
type outputs\modeling\with_spike\regression\multiple_linear\load_verification.json
```

ต้องเห็น: `pytest` ไม่มี failed · ทั้งสอง variant ขึ้น `reload=ok` · ไฟล์ JSON มี `"passed": true`

> **merge แล้วแจ้งเพื่อนทันที** ให้ `git pull --rebase origin main` แล้วเริ่ม T1 ของเขาได้

หลัง merge: `src/ml/core/**` ถือว่า **freeze** — ถ้าจำเป็นต้องแก้ ให้แก้บน `main` และแจ้งอีกฝั่งก่อน
เพราะเขาเขียนโค้ดทับ interface นั้นอยู่

---

## PHASE B — 10 โมเดลของฝั่งนี้

```bat
git checkout main && git pull && git checkout -b feat/models-linear-family
```

ลำดับเรียงจาก **ง่าย → ยาก** เพื่อให้ได้โมเดลครบเร็วที่สุดในช่วงแรก และเหลือเวลาให้ตัวยากท้าย ๆ

| # | Task | card | ชม. | จุดที่ต้องระวังที่สุด |
| --- | --- | --- | ---: | --- |
| B1 | `simple_linear` | #1 | 1 | ตัวหารเป็น `Σ(x-x̄)²` ไม่ใช่ `var(ddof=1)` · ฟีเจอร์คงที่ต้องคืน `measure(reason=...)` |
| B2 | `polynomial` | #3 | 1 | **standardize ก่อนยกกำลัง** · ปิด bias ของ `PolynomialFeatures` |
| B3 | `gaussian_nb` | #6 | 1 | `var_smoothing` คูณกับ **variance สูงสุดของทั้งชุด** ไม่ใช่ของแต่ละคลาส |
| B4 | `knn` | #7 | 1.5 | `argsort(kind="stable")` เพื่อ tie-break ให้ตรง · `algorithm="brute"` ใน reference |
| B5 | `perceptron` | #8 | 1.5 | **`tol=None`** ใน reference ไม่งั้น sklearn early-stop คนละรอบ |
| B6 | `logistic` | #5 | 2.5 | `C = 1/(λ·Σw)` · ใช้ `l1_ratio=0` ไม่ใช่ `penalty="l2"` (sklearn 1.9) |
| B7 | `elastic_net` ★ | #4 | 3 | soft-threshold · หาร `1/(2n)` · `selection="cyclic"` |
| B8 | `kmeans` | #18 | 2 | k-means++ ของ sklearn สุ่ม `2+log(k)` candidate ต่อรอบ ไม่ใช่ตัวเดียว |
| B9 | `agglomerative` | #19 | 2.5 | Lance-Williams update · OOS ใช้ frozen nearest-centroid |
| B10 | ทบทวน + เก็บตก | — | 1 | ตรวจ card ทุกใบว่าใส่ตัวเลขจริงครบ |

**วงจรของแต่ละตัว** ใช้เหมือนฝั่งเพื่อนทุกประการ — ดู
[`docs/tasks/codex-tree-family.md`](codex-tree-family.md) §4 (TDD 7 ขั้น) และ §5 (เมื่อ parity ไม่ผ่าน)

### หมายเหตุเฉพาะฝั่งนี้

**B7 `elastic_net` ★** เป็นโมเดล extracurricular ที่โจทย์บังคับให้ทำ **ทั้ง scratch และ ready-to-use**
และอธิบายกลไกภายในอย่างละเอียดในรายงาน → ตอนเขียน card ให้เขียนลึกพอที่จะยกไปใส่รายงาน §6.1 ได้ตรง ๆ
จุดขายคือ multicollinearity ที่วัดได้จริง (condition number ของ covariance matrix 35.6, `hist_vol_5d ~ 20d` |r| = 0.804)

**B8–B9 clustering** ใช้ `task = "clustering"` และ `fit(X, y=None)` · การประเมินใช้ silhouette เป็นหลัก ·
`agglomerative` ไม่มี `predict` สำหรับข้อมูลใหม่โดยธรรมชาติ ให้ใช้ frozen nearest-centroid
และ**ประกาศข้อจำกัดนี้ในรายงาน** (เป็นเนื้อหาที่ได้คะแนน discussion)

---

## PHASE C — ปิดงาน (ต้องรอโมเดลครบทั้งสองฝั่ง)

| # | Task | รายละเอียด | เกณฑ์ผ่าน |
| --- | --- | --- | --- |
| C1 | `run.py report` | รวม `validation_metrics.json` ทุกโมเดลเป็น leaderboard · ROC ซ้อน · confusion · performance curve | `leaderboard.csv` มี 19 แถวต่อ variant |
| C2 | รัน variant B ทั้งชุด | `--variant non_spike` ทุกโมเดล | 38 runs ครบ |
| C3 | paired comparison A↔B | ตารางเดลตาต่อโมเดล + กราฟ | `comparison_a_vs_b.csv` |
| C4 | `run.py finalize` | **แตะ Test ครั้งเดียว** หลังเลือกโมเดลครบแล้ว | `test_metrics.json` ครบ |
| C5 | notebooks 03 / 04 / 05 | regression · classification · results comparison | รันจบตั้งแต่ต้นจนจบไม่ error |
| C6 | report + slides | ตาม [`REPORT_AND_SLIDES_OUTLINE.md`](../plan/REPORT_AND_SLIDES_OUTLINE.md) | ทุกช่องมีตัวเลขจริง |

**C4 คือจุดที่ห้ามพลาด** — Test set ต้องถูกอ่านครั้งเดียวในชีวิตโปรเจกต์ หลังจาก
hyperparameter ของทุกโมเดลถูกล็อกจาก Validation แล้วเท่านั้น

---

## ตารางเวลา

| วัน | Phase A | Phase B | Phase C |
| --- | --- | --- | --- |
| 10 ศ. | A1–A5 | | |
| 11 ส. | A6–A10 (pilot ครบลูป) | | |
| 12 อา. | A11–A12 **merge + แจ้งเพื่อน** | B1–B2 | |
| 13 จ. | | B3–B5 | |
| 14 อ. | | B6–B7 | |
| 15 พ. | | B8–B10 | |
| 16 พฤ. | | | C1–C3 |
| 17 ศ. | | | C4–C6 (ร่างรายงาน) |
| 18 ส. | | | slides + ซ้อม + ส่ง |

ถ้าช้ากว่ากำหนด ให้ตัดตามลำดับใน [`ROADMAP.md`](../plan/ROADMAP.md) §4 — **ห้ามตัด C4, C6**

---

## Definition of Done ของ branch นี้

- [ ] core framework merge เข้า `main` แล้วและเพื่อนเริ่มงานได้
- [ ] 11 โมเดลผ่าน DoD ครบ 6 ข้อตาม `AGENTS.md` §5
- [ ] `pytest -q` เขียวทั้งชุด
- [ ] 38 runs ครบ ทุกตัวมี `load_verification.json` ที่ `"passed": true`
- [ ] leaderboard + paired comparison A↔B เสร็จ
- [ ] Test ถูกแตะครั้งเดียวในขั้น finalize
- [ ] notebooks 03/04/05 รันจบ · report + slides เสร็จ
