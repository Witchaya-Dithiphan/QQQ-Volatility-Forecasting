# REPORT & SLIDES OUTLINE — ผูกกับเกณฑ์ให้คะแนนทุกบรรทัด

> โครงที่เขียนตามได้เลยตอนเวลากดดัน · ภาษาไทย ศัพท์เทคนิคอังกฤษ
> คะแนนเต็ม **100 (+10)** และเกือบทั้งหมดถูกวัดจากเอกสารชุดนี้ ไม่ใช่จากโค้ด
> ส่งวันที่ **18 ต.ค. 2026** — นำเสนอ on-site

---

## 0. ตาราง traceability — เช็คลิสต์หลักของโปรเจกต์

ทุกบรรทัดคะแนนจาก PDF หน้า 5 → อยู่ตรงไหนในรายงาน → สไลด์ไหน → ไฟล์อะไรพิสูจน์

| เกณฑ์ (PDF) | คะแนน | หัวข้อรายงาน | สไลด์ | artifact ที่พิสูจน์ | ✓ |
| --- | ---: | --- | --- | --- | --- |
| 1. Finalize the problem setting | ส่วนของ 30 | §1 | S2–S3 | `PROJECT_STATUS.md` §1 | ☐ |
| 2. Finalize feature selection and insight — **why must use ML/DL?** | ส่วนของ 30 | §2, §2.3 | S4–S6 | `outputs/reports/feature_report.json`, notebook 02 | ☐ |
| 3. Case analysis & preparation of structured data | ส่วนของ 30 | §3 | S7–S9 | `cleaning_report.json`, `data_split_report.json`, `spike_analysis.json` | ☐ |
| Remark: ใช้ feature ≥ 5 | (เงื่อนไข) | §2.1 | S4 | 8 features ใน `configs/modeling.json` ✅ ผ่านแล้ว | ☐ |
| Task: Regression — ทุกตัว from scratch | ส่วนของ 40 | §4.1, §5.1 | S10 | `src/ml/regression/*.py` + parity tests | ☐ |
| Task: Classification — ทุกตัว from scratch | ส่วนของ 40 | §4.2, §5.2 | S11 | `src/ml/classification/*.py` + parity tests | ☐ |
| A better **regression** model (บังคับ) + อธิบายกลไกละเอียด | ส่วนของ 40 | **§6.1** | **S12** | `elastic_net.py` + `MODEL_CARDS.md` #4 | ☐ |
| A better **classification** model (บังคับ) + อธิบายกลไกละเอียด | ส่วนของ 40 | **§6.2** | **S13** | `adaboost.py` + `MODEL_CARDS.md` #15 | ☐ |
| Remark: ทุกโมเดล from scratch, มี ready-to-use ประกอบเพื่อยืนยันผล | ส่วนของ 40 | §5.3 | S14 | `compare.py` + ตาราง parity 19 แถว | ☐ |
| Build model, training-testing iterations, batch runs | ส่วนของ 40 | §5.4 | S14 | `trainer.py`, `search_results.json` ทุกโมเดล | ☐ |
| **save-load model แล้ว test ซ้ำ** | ส่วนของ 40 | §5.5 | S14 | `load_verification.json` 38 ไฟล์ `"passed": true` | ☐ |
| Evaluation: Loss | ส่วนของ 30 | §7.1 | S15 | `validation_metrics.json` | ☐ |
| Evaluation: Confusion matrix | ส่วนของ 30 | §7.2 | S16 | `outputs/figures/confusion_*.png` | ☐ |
| Evaluation: Accuracy | ส่วนของ 30 | §7.2 | S16 | leaderboard | ☐ |
| Evaluation: Precision, Sensitivity, Specificity, TNR, F1 | ส่วนของ 30 | §7.2 | S16 | leaderboard (5 คอลัมน์แยก) | ☐ |
| Evaluation: ROC & AUC curve | ส่วนของ 30 | §7.3 | S17 | `outputs/figures/roc_*.png` | ☐ |
| Evaluation: Performance curve | ส่วนของ 30 | §7.4 | S18 | `history.json` → `perf_curve_*.png` | ☐ |
| Evaluation: R-Square (regression) | ส่วนของ 30 | §7.1 | S15 | `validation_metrics.json` → `r2` | ☐ |
| **Benchmarking, analysis, and discussion** | ส่วนของ 30 | **§8, §9** | S19–S21 | `leaderboard.csv` + `comparison_a_vs_b.csv` | ☐ |
| Data analysis (if any) | optional | §3.4 | S9 | notebook 02 | ☐ |
| +5 creative problem/storytelling/data prep | +5 | §1, §3.3, §10.1 | S8, S22 | spike analysis + A/B design | ☐ |
| +5 model ดีกว่าโมเดลที่สอนในห้องมาก | +5 | §8.3, §10.2 | S20 | ตารางเทียบ + ablation | ☐ |
| ส่ง Report | deliverable | ทั้งฉบับ | — | `report.pdf` | ☐ |
| ส่ง Slide & Presentation | deliverable | — | ทั้งชุด | `slides.pdf` | ☐ |
| ส่ง .ipynb และ/หรือ .py | deliverable | — | — | notebooks 01–05 + `src/` | ☐ |
| Other related documents (if any) | optional | — | — | `MODEL_CARDS.md`, `ARCHITECTURE.md` | ☐ |
| ลงทะเบียนกลุ่ม+dataset ใน Google Sheet (หน้า 4) | เงื่อนไข | — | — | **ต้องเช็คภายใน 11 ต.ค.** (R10) | ☐ |

---

## Part A — โครงรายงาน

เป้าหมายความยาว ~25–35 หน้ารวมรูป

### §1 บทนำและการตั้งปัญหา (2–3 หน้า · ส่วนของ 30 คะแนน + creative)

- **ปัญหา:** พยากรณ์ความผันผวน (realized volatility) ของ QQQ ล่วงหน้า 5 วันทำการ ในสองรูปแบบ
  — regression ทำนายค่าต่อเนื่อง `target_volatility_5d` และ classification ทำนายว่าจะเข้าสู่ภาวะ
  ผันผวนสูงหรือไม่ `target_high_volatility` (เกณฑ์ Q75 = 0.2530580184684854 จาก train เท่านั้น)
- **ทำไมเรื่องนี้สำคัญ:** volatility เป็นอินพุตโดยตรงของการตั้งราคา option, การกำหนดขนาดสถานะ
  และการบริหารความเสี่ยง — การรู้ล่วงหน้า 5 วันมีมูลค่าใช้งานจริง ไม่ใช่แค่แบบฝึกหัด
- **คำถามวิจัยที่ทำให้โปรเจกต์นี้ต่างจากกลุ่มอื่น:** *เหตุการณ์ spike ที่รุนแรงช่วยหรือทำร้าย
  การเรียนรู้ของโมเดล?* → ออกแบบเป็นการทดลองแบบจับคู่ 2 ชุดข้อมูล
- **ขอบเขต:** daily OHLCV 2,512 แถว 2016-08-31 → 2026-08-28

### §2 ฟีเจอร์ storytelling และเหตุผลที่ต้องใช้ ML (3–4 หน้า · **คะแนนตรง**)

โจทย์ถามตรง ๆ ว่า *"why must use ML/DL?"* ต้องมีหัวข้อที่ตอบคำถามนี้โดยเฉพาะ

- **§2.1 ฟีเจอร์ 8 ตัวและเหตุผลรายตัว** — ตาราง: ชื่อ · นิยาม · สมมติฐานทางการเงินที่อยู่เบื้องหลัง ·
  คาดว่าสัมพันธ์กับ target อย่างไร (ผ่านเงื่อนไข ≥ 5 features ของโจทย์)
- **§2.2 Feature insight จากข้อมูลจริง** — correlation matrix, การกระจายตัว, ความสัมพันธ์กับ target
  จาก notebook 02 · **ชี้ประเด็น multicollinearity ตรงนี้:** condition number ของ covariance matrix = 35.6 (cond(X) = 6.0),
  `historical_volatility_5d ~ 20d` |r| = 0.804 ซึ่งจะกลายเป็นเหตุผลของการเลือก Elastic Net ใน §6.1
- **§2.3 ทำไมต้องใช้ ML/DL** — ตอบด้วยหลักฐาน ไม่ใช่ความเห็น:
  ความสัมพันธ์ไม่เชิงเส้นและมี regime change · target มี skew 2.80 / kurtosis 16.26 ซึ่งผิดจากสมมติฐาน
  ของวิธีเชิงสถิติแบบดั้งเดิม · ฟีเจอร์มีปฏิสัมพันธ์กันที่กฎ if-then เขียนมือไม่ไหว ·
  แสดงว่า baseline เชิงเส้นทำได้แค่ไหน แล้วโมเดลที่ซับซ้อนขึ้นดีกว่าเท่าไร

### §3 การเตรียมข้อมูลและการวิเคราะห์กรณี (4–5 หน้า · ส่วนของ 30 คะแนน + creative)

- **§3.1 แหล่งข้อมูลและการ freeze** — snapshot 2,512 แถว SHA-256 `649e1db1…b6b13`
  อธิบายว่าทำไมถึง pin snapshot แทนการ download สด (แหล่งข้อมูลแบบ rolling ให้ผลต่างกันทุกวัน
  ทำให้ reproduce ไม่ได้) — ประเด็นนี้แสดงวุฒิภาวะด้านวิธีวิจัย
- **§3.2 Cleaning → features → targets → split** — chronological split 1,733 / 371 / 373
  **อธิบายว่าทำไมห้ามสุ่มแบ่ง** (จะเกิด look-ahead leakage) และ Q75 คำนวณจาก train เท่านั้น
- **§3.3 Spike analysis — จุดขายของโปรเจกต์** (creative +5)
  นิยาม spike = `|return_1d| > Q3 + 3·IQR` fit จาก train เท่านั้น → threshold 0.0465436445413787 ·
  หน้าต่างผลกระทบ `[s-5, s+19]` · ได้ `non_spike` train 1,520 แถว (ตัดออก 213)
  **ย้ำว่า validation/test เป็น full original ทั้งสอง variant** จึงเป็น paired comparison ที่เปลี่ยนตัวแปรเดียว
- **§3.4 Data analysis** — สถิติเชิงพรรณนา, class balance 1,300/433, การกระจายของ target

### §4 ระเบียบวิธี (2–3 หน้า · ส่วนของ 40 คะแนน)

- **§4.1 Inventory 19 โมเดล** — ตารางจาก `MODEL_CARDS.md` §1
- **§4.2 ทำไมถึงเขียนเองทั้งหมด** — ข้อกำหนดของโจทย์ + สิ่งที่ได้เรียนรู้จากการเขียนเอง
- **§4.3 Protocol** — tune บน validation เท่านั้น · **แตะ test ครั้งเดียวตอนจบ** · seed 42 ·
  preprocessing fit จาก train ของ variant นั้นเท่านั้น

### §5 การ implement (4–6 หน้า · **40 คะแนน**)

- **§5.1 Regression 4 ตัว** — สรุปกลไกตัวละย่อหน้า
- **§5.2 Classification 13 + Clustering 2** — เช่นเดียวกัน
- **§5.3 การยืนยันผลด้วย reference** — **ตารางสำคัญ 19 แถว:** โมเดล · reference class · เกณฑ์ ·
  ผลต่างสูงสุดที่วัดได้ · ผ่าน/ไม่ผ่าน · เหตุผลถ้าต่าง
  อธิบายด้วยว่า sklearn ใช้ legacy MT19937 จึงต้องใช้ `RandomState` ไม่ใช่ `default_rng`
  (เป็นรายละเอียดที่แสดงว่าเข้าใจจริง)
- **§5.4 Training iterations และ batch runs** — grid search, จำนวน candidate ต่อโมเดล, 38 runs
- **§5.5 Save/load และ re-test** — **ข้อบังคับตรงตัวของโจทย์** แสดงว่าทุก run มี
  `load_verification.json` ที่ `passed: true` พร้อมอธิบายว่าพิสูจน์อย่างไร

### §6 โมเดลนอกห้องเรียน 2 ตัว (4–5 หน้า · **คะแนนเน้น ให้พื้นที่เยอะที่สุดต่อหน้า**)

โจทย์สั่ง *"describe the internal mechanism of the model in detail"* ทั้งสองตัว
และบังคับว่าต้องมี **ทั้ง scratch และ ready-to-use**

- **§6.1 Elastic Net (regression)** — ยกเนื้อหาจาก `MODEL_CARDS.md` #4 มาขยาย:
  objective, coordinate descent, soft-threshold และ**ทำไมถึงเหมาะกับชุดนี้** (multicollinearity ที่วัดได้จริง)
  พร้อมเทียบ coefficient path กับ Multiple Regression ให้เห็นว่า L1 ตัดฟีเจอร์ซ้ำซ้อนทิ้งจริง
- **§6.2 AdaBoost (classification)** — ยกจาก `MODEL_CARDS.md` #15: SAMME, การอัปเดตน้ำหนัก, α
  และ**สมมติฐานที่ทดสอบได้**: exponential loss ไวต่อ outlier ที่สุด + target kurtosis 16.26
  → ทำนายว่าจะฟื้นตัวมากที่สุดใน variant B (ไปพิสูจน์ใน §8.2)

### §7 ผลการทดลอง (5–7 หน้า · **30 คะแนน**)

- **§7.1 Regression** — ตาราง RMSE / MAE / **R²** ทุกโมเดล × 2 variants + loss
- **§7.2 Classification** — ตาราง Accuracy / Precision / **Sensitivity** / **Specificity** / **TNR** /
  F1 / AUC — **ต้องมีครบทุกคอลัมน์ตามที่โจทย์ลิสต์ไว้** (Specificity กับ TNR พิมพ์แยกสองคอลัมน์
  เพราะ PDF ลิสต์แยกกัน แม้จะเป็นค่าเดียวกัน — ต้นทุนเป็นศูนย์ ไม่ต้องเสี่ยง)
- **§7.3 ROC + AUC** — ROC 13 เส้นซ้อนกันต่อ variant
- **§7.4 Performance curve** — loss ต่อ iteration ของโมเดลที่เทรนแบบวนรอบ
- **§7.5 Clustering** — silhouette, inertia, cluster size และข้อจำกัดของ Agglomerative เรื่อง OOS
- **§7.6 ผลบน Test** — ประกาศชัดว่าแตะครั้งเดียวหลังเลือกโมเดลเสร็จหมดแล้ว

### §8 Benchmarking และการอภิปราย (4–5 หน้า · **โจทย์ลิสต์ไว้ตรง ๆ ไม่ใช่แค่พิมพ์ตัวเลข**)

- **§8.1 Leaderboard รวม** — จัดอันดับทุกโมเดล พร้อมอธิบายว่าทำไมตัวชนะถึงชนะ
- **§8.2 Paired comparison A ↔ B** — **หัวใจของรายงาน** ตารางเดลตาต่อโมเดล
  (metric ของ `non_spike` ลบ `with_spike`) แล้ววิเคราะห์ว่า
  โมเดลกลุ่มไหนได้ประโยชน์จากการตัด spike และกลุ่มไหนเสีย · เทียบกับที่ทฤษฎีทำนายไว้ใน §6.2
- **§8.3 เทียบกับโมเดลที่สอนในห้อง** (+5) — baseline เชิงเส้น vs ตัวที่ดีที่สุดของเรา
  วัดเป็น % improvement ให้ชัด
- **§8.4 Error analysis** — ทำนายพลาดตรงไหน ช่วงเวลาไหน เกี่ยวกับ spike หรือไม่

### §9 ข้อจำกัดและงานต่อ (1–2 หน้า)

ข้อมูลชุดเดียวช่วงเวลาเดียว · Agglomerative ไม่มี OOS predict ตามธรรมชาติ ·
โมเดลที่ parity ไม่ผ่านและเหตุผล (ถ้ามี) · ไม่ได้ทำ walk-forward retraining

### §10 สรุป (1 หน้า)

ตอบคำถามวิจัยใน §1 · ข้อค้นพบหลัก 3 ข้อ · สิ่งที่ได้เรียนรู้จากการเขียนทุกโมเดลเอง

### ภาคผนวก

ตาราง parity 19 แถวเต็ม · hyperparameter grid ทั้งหมด · วิธี reproduce (ชี้ไป `RUNBOOK.md`)

---

## Part B — โครงสไลด์

เสนอ **22 สไลด์** สำหรับช่วงเวลาปกติ ~15–20 นาที · คอลัมน์ "ตัดได้" คือลำดับที่ตัดเมื่อเวลาน้อย

| # | สไลด์ | สารที่ต้องสื่อ | ภาพที่พา | ตัดได้ |
| ---: | --- | --- | --- | --- |
| S1 | ปก | ชื่อโปรเจกต์ · สมาชิก · วันที่ | — | |
| S2 | ปัญหา | พยากรณ์ volatility QQQ ล่วงหน้า 5 วัน 2 รูปแบบ | กราฟ volatility ตามเวลา | |
| S3 | ทำไมสำคัญ | ใช้ตั้งราคา option / คุมความเสี่ยง | — | ✂3 |
| S4 | ข้อมูลและฟีเจอร์ | 2,512 แถว 10 ปี · 8 features | ตารางฟีเจอร์ | |
| S5 | Feature insight | multicollinearity ชัด cond(X'X) = 35.6 | correlation heatmap | |
| S6 | **ทำไมต้องใช้ ML** | ไม่เชิงเส้น + fat tail (kurtosis 16.26) | scatter + distribution | |
| S7 | การเตรียมข้อมูล | chronological split ห้ามสุ่ม | ไทม์ไลน์ split | |
| S8 | **Spike analysis** | จุดขาย — นิยาม spike และหน้าต่างผลกระทบ | `daily_return_spikes.png` | |
| S9 | 2 ชุดข้อมูล | A 1,733 vs B 1,520 · val/test เหมือนกัน | `dataset_comparison.png` | |
| S10 | Regression 4 ตัว | ทุกตัวเขียนเอง | ตาราง | |
| S11 | Classification 13 + Clustering 2 | ทุกตัวเขียนเอง | ตาราง | |
| S12 | **Elastic Net** | กลไก + ทำไมเหมาะกับ multicollinearity | สมการ + coefficient path | |
| S13 | **AdaBoost** | กลไก + ทำไมไวต่อ spike | สมการ + weight evolution | |
| S14 | พิสูจน์ความถูกต้อง | scratch ตรงกับ sklearn + save/load ผ่าน | ตาราง parity | |
| S15 | ผล regression | RMSE / R² | ตาราง | |
| S16 | ผล classification | 7 metric ครบ | ตาราง + confusion matrix | |
| S17 | ROC + AUC | เส้นไหนเหนือกว่า | ROC ซ้อน | |
| S18 | Performance curve | loss ลู่เข้าจริง | loss curve | |
| S19 | Leaderboard | ตัวไหนชนะ เพราะอะไร | ตารางจัดอันดับ | |
| S20 | **A ↔ B** | ตัดสินคำถามวิจัย | bar chart ของเดลตา | |
| S21 | ดีกว่า baseline เท่าไร | +5 | bar chart | ✂2 |
| S22 | สรุป + ข้อจำกัด | 3 ข้อค้นพบ · ข้อจำกัดที่ยอมรับ | — | |

ลำดับตัด: ✂1 = S18 (ถ้ามีใน §7 แล้ว) · ✂2 = S21 (ยุบเข้า S19) · ✂3 = S3 (ยุบเข้า S2)
**ห้ามตัด:** S8, S9, S12, S13, S14, S20 — เป็นสไลด์ที่แบกคะแนนเน้นทั้งหมด

---

## Part C — แผนเก็บคะแนนพิเศษ +10

### +5 ข้อ 1 — problem/storytelling/data preparation ที่สร้างสรรค์

ของที่เรามีอยู่จริงแล้วและกลุ่มอื่นไม่น่ามี:

1. **ตั้งคำถามวิจัยที่ทดสอบได้ แทนที่จะแค่ "เอาข้อมูลมาเทรนโมเดล"** — *spike ช่วยหรือทำร้ายการเรียนรู้?*
   แล้วออกแบบการทดลองจับคู่มาตอบมันโดยเฉพาะ
2. **Spike detection ที่ทำอย่างมีวินัย** — fit จาก train เท่านั้น, หน้าต่างผลกระทบ `[s-5, s+19]`,
   ตัดเฉพาะ train ส่วน validation/test คงเดิม เพื่อให้เปรียบเทียบได้จริง
3. **Reproducibility ระดับ snapshot pinning + SHA-256** ทุกไฟล์ และปฏิเสธ live download
   โดยอธิบายเหตุผล — เป็นวุฒิภาวะด้านวิธีวิจัยที่เห็นได้ชัดในรายงาน

### +5 ข้อ 2 — โมเดลที่ดีกว่าโมเดลที่สอนในห้องมาก

PDF เขียนว่า *"Your model has much better performance than existing-taught models in on the above list"*
— **ไม่ได้บังคับว่าต้องเป็นโมเดล extracurricular** ดังนั้นเล่นสองทางพร้อมกัน:

- **ทางหลัก:** เทียบตัวที่ดีที่สุดของเรา (น่าจะเป็น GB / XGBoost / Stacking) กับ baseline เชิงเส้น
  (Simple/Multiple Linear, Logistic) แล้วรายงาน % improvement บน AUC และ F1 ให้ชัดเป็นตัวเลข
- **ทางเสริมที่เป็นงานวิจัยจริง:** แสดงว่า **การเลือกชุดข้อมูล (A vs B) ให้ผลต่างมากกว่าการเลือกโมเดล**
  ถ้าผลออกมาเช่นนั้นจริง — เป็นข้อค้นพบที่หนักแน่นกว่าการบอกว่า "โมเดล X ชนะ" และโยงกลับไป
  สมมติฐาน AdaBoost ใน §6.2 ได้พอดี

**ข้อควรระวัง:** ถ้า B ไม่ได้ดีกว่า A ก็**รายงานตามจริงพร้อมวิเคราะห์ว่าทำไม** — ผลลบที่อธิบายได้
ยังเป็นงานที่ดี ส่วนการบิดตัวเลขให้สวยคือสิ่งที่กฎข้อ 1 ของโปรเจกต์ห้ามไว้

---

## Part D — ข้อกำหนดที่เกือบตกหล่น

ตรวจจาก PDF ทั้ง 5 หน้าแล้ว รายการนี้คือสิ่งที่ไม่อยู่ใน master prompt เดิมและต้องไม่ลืม

| # | ข้อกำหนด | อยู่ที่ | จัดการที่ |
| --- | --- | --- | --- |
| 1 | **ต้องตอบคำถาม "why must use ML/DL?"** เป็นเกณฑ์ให้คะแนนโดยตรง | หน้า 5 | §2.3 + S6 |
| 2 | **storytelling และ feature insight** เป็นงานที่ถูกให้คะแนน ไม่ใช่แค่ EDA | หน้า 1 | §2 + S5 |
| 3 | **benchmarking + analysis + discussion** ลิสต์แยกจาก metric | หน้า 3 | §8 |
| 4 | ใช้ feature ≥ 5 ไม่งั้นต้องขออนุญาตอาจารย์ก่อน | หน้า 1 | เรามี 8 ✅ |
| 5 | โมเดลใหม่ 2 ตัวต้องมี **ทั้ง scratch และ ready-to-use** | หน้า 3 | §6, `MODEL_CARDS.md` #4, #15 |
| 6 | Specificity กับ True Negative Rate ถูกลิสต์แยกกัน | หน้า 3 | §7.2 พิมพ์สองคอลัมน์ |
| 7 | R-Square ระบุว่า "For regression analysis" | หน้า 3 | §7.1 เท่านั้น |
| 8 | Feature importance เป็น hint พร้อมลิงก์อ่าน | หน้า 1–2 | §2.2 (ทางเก็บ creative) |
| 9 | **ลงทะเบียนกลุ่ม + dataset ใน Google Sheet** | หน้า 4 | งานธุรการ — เช็คภายใน 11 ต.ค. (R10) |
| 10 | นำเสนอ **on-site ในคาบเรียน** | หน้า 1 | ต้องซ้อมพูด ไม่ใช่แค่ส่งไฟล์ |
| 11 | Demo เป็น optional | หน้า 5 | ทำถ้าเหลือเวลาเท่านั้น |

---

## ลำดับการเขียนเมื่อเวลาเหลือน้อย

เขียนตามลำดับนี้ เพราะเรียงตามคะแนนต่อชั่วโมงที่ลงแรง:

1. **§7 + §8** (30 คะแนน และตัวเลขมาจาก artifact อยู่แล้ว เขียนเร็วที่สุด)
2. **§6** (คะแนนเน้น และเนื้อหายกจาก `MODEL_CARDS.md` ได้เกือบทั้งหมด)
3. **§2 + §3** (30 คะแนน และงานข้อมูลเสร็จหมดแล้ว แค่เรียบเรียง)
4. **§5** (ยกจาก `MODEL_CARDS.md` + `ARCHITECTURE.md`)
5. §1, §4, §9, §10 (สั้นและเขียนทีหลังได้)
6. สไลด์ — ทำหลังรายงานนิ่งแล้ว จะได้ไม่ต้องแก้สองรอบ
