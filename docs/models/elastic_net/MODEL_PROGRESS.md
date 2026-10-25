# elastic_net ★ — regression · ✅ เสร็จ · **extracurricular**

**ผู้รับผิดชอบ:** พิทวัส · **สเปก:** [card #4](../../design/MODEL_CARDS.md) · **วันที่:** 2026-10-11

> โมเดลนอกห้องเรียนฝั่ง regression ที่โจทย์บังคับให้ทำ **ทั้ง scratch และ ready-to-use**
> และ **อธิบายกลไกภายในอย่างละเอียด** → เนื้อหาในไฟล์นี้ยกไปใส่รายงาน §6.1 ได้โดยตรง

## ผล (validation) — ชนะทุกโมเดล regression

| variant | alpha | l1_ratio | RMSE | R² |
| --- | --- | --- | --- | --- |
| `with_spike` | 0.1 | 0.25 | **0.066806** | **+0.016942** |
| `non_spike` | 0.1 | 0.25 | **0.064256** | **+0.090577** |

reload ✅ · preprocess `standardize` · 12 candidates

### เทียบกับโมเดล regression ที่สอนในห้อง — หลักฐานสำหรับ +5 "much better"

| model | RMSE A | R² A | RMSE B | R² B |
| --- | --- | --- | --- | --- |
| `simple_linear` | 0.069038 | −0.049847 | 0.066479 | +0.026542 |
| `multiple_linear` | 0.068864 | −0.044554 | 0.066820 | +0.016536 |
| `polynomial` | 0.069065 | −0.050652 | 0.066056 | +0.038890 |
| **`elastic_net` ★** | **0.066806** | **+0.016942** | **0.064256** | **+0.090577** |

- **เป็นโมเดลเดียวที่ได้ R² บวกบน `with_spike`** อีกสามตัวติดลบหมด แปลว่า
  **มันเป็นโมเดล regression ตัวเดียวที่ทำนายได้ดีกว่าการเดาค่าเฉลี่ย เมื่อข้อมูลยังมี spike อยู่**
- เทียบ `multiple_linear` โดยตรง: RMSE ลดลง **2.99%** (A) และ **3.84%** (B) ·
  R² จาก **−0.0446 → +0.0169** (A) และ **+0.0165 → +0.0906** (B)

## ทำไมมันชนะ — เรื่องที่ใช้เขียนรายงานได้

**การเลือกฟีเจอร์เกิดขึ้นจริง ไม่ใช่แค่ทฤษฎี** — L1 ตัดฟีเจอร์ทิ้งเป็นศูนย์พอดี:

| variant | ฟีเจอร์ที่รอด | น้ำหนัก |
| --- | --- | --- |
| `with_spike` | 5/8 | `historical_volatility_5d` 0.0207 · `historical_volatility_20d` 0.0235 · `intraday_range` 0.0289 · `sma_ratio_5_20` −0.0034 · `rsi_14` −0.0036 |
| `non_spike` | 4/8 | `historical_volatility_5d` 0.0070 · `historical_volatility_20d` 0.0144 · `intraday_range` 0.0122 · `rsi_14` −0.0052 |

ตัด `return_1d`, `return_5d`, `volume_zscore_20` ทิ้งทั้งสอง variant และ**เก็บกลุ่มที่วัดความผันผวน
ไว้ทั้งหมด** ซึ่งตรงกับสัญชาตญาณทางการเงิน: volatility มี persistence (วันนี้ผันผวน พรุ่งนี้มักผันผวนต่อ)
ส่วน return รายวันเป็น noise ที่แทบไม่มีข้อมูลทิศทางความผันผวน

**ชนะเพราะ multicollinearity พอดี** — `historical_volatility_5d` กับ `20d` สัมพันธ์กัน |r| = 0.804
และ condition number ของ covariance matrix = 35.6 ทำให้ OLS กระจายน้ำหนักระหว่างสองตัวนี้
แบบไม่เสถียร Elastic Net เก็บทั้งคู่ไว้ด้วยน้ำหนักที่สมดุล (0.0207 / 0.0235) แทนที่จะให้
ตัวใดตัวหนึ่งพองแล้วอีกตัวติดลบชดเชย

## กลไกโดยละเอียด (สำหรับรายงานหัวข้อ 2.1.3)

```text
J(b) = ‖y − Xb‖² / (2n)  +  α·ρ·‖b‖₁  +  (α(1−ρ)/2)·‖b‖²        ρ = l1_ratio
```

พจน์ L1 ไม่ differentiable ที่ 0 จึงใช้ gradient descent ไม่ได้ → ใช้ **coordinate descent**
ซึ่ง fix ทุกสัมประสิทธิ์ไว้แล้วแก้ทีละตัว ซึ่งมีคำตอบปิดรูป:

```text
r_j   = y − Xb + X[:,j]·b_j        residual ที่ยังไม่รวมผลของ feature j
ρ_j   = X[:,j]ᵀ·r_j / n            correlation ของ j กับ residual
z_j   = X[:,j]ᵀ·X[:,j] / n
b_j   = S(ρ_j, α·ρ) / (z_j + α(1−ρ))

soft-threshold:  S(v, γ) = sign(v)·max(|v| − γ, 0)
```

- **`S` คือหัวใจของ L1** — ถ้า correlation ของฟีเจอร์กับ residual ต่ำกว่าเกณฑ์ γ
  สัมประสิทธิ์จะถูกผลักเป็น **0 พอดี** ไม่ใช่แค่เล็กลงอย่าง Ridge → นี่คือที่มาของการเลือกฟีเจอร์
- **ตัวหาร `(z_j + α(1−ρ))` คือที่ที่ L2 ทำงาน** — หดทุกสัมประสิทธิ์ที่รอดแบบ proportional
  และเป็นเหตุผลที่ฟีเจอร์ที่สัมพันธ์กันได้น้ำหนักใกล้เคียงกันแทนที่จะถูกเลือกมาตัวเดียว
- **intercept ไม่ถูก penalize** — fit บนข้อมูลที่ center แล้วคำนวณกลับเป็น `b₀ = ȳ − x̄ᵀb`

## Parity

exact ครบ **12/12 candidates** (alpha 4 × l1_ratio 3) · coefficient ตรงที่ **1e-6**
ทดสอบเพิ่มบนข้อมูล collinear |r| = 0.8 ที่จำลองความสัมพันธ์ของฟีเจอร์จริง

**scratch + ready-to-use ครบตามที่โจทย์บังคับ** — `reference()` คืน `sklearn.linear_model.ElasticNet`
ที่ตั้งค่าตรงกันทุกตัว

## การตัดสินใจ

- objective หาร `1/(2n)` ตรงกับ sklearn — ถ้าใช้ `1/2` เฉย ๆ ค่า α เดียวกันจะแรงกว่า n เท่า
- `selection="cyclic"` วน j ตามลำดับ 0..p−1 ไม่สุ่ม
- อัปเดต residual แบบ incremental (`residual -= X[:,j]·Δb_j`) แทนการคำนวณใหม่ทั้งก้อนทุกครั้ง
- center ข้อมูลก่อน fit เพื่อให้ intercept อยู่นอก penalty โดยไม่ต้องใส่ column ของ 1

## ปัญหาที่เจอและแก้แล้ว

**test ที่ผมเขียนเองผิด** — `test_intercept_is_not_penalised` ตั้งสมมติฐานว่า α=1.0 กับ
l1_ratio=1.0 จะดันทุกสัมประสิทธิ์เป็นศูนย์ แต่สัญญาณที่แรงจริง (coef 2.0) ยังรอดมาที่ 1.14
ซึ่งเป็นพฤติกรรม Lasso ที่ถูกต้อง

**แก้:** เขียน test ใหม่ให้วัดคุณสมบัติโดยตรง — เลื่อน `y` ไป 1000 แล้วตรวจว่า intercept
ดูดส่วนต่างไปทั้งหมดโดยสัมประสิทธิ์ไม่ขยับ (ซึ่งคือนิยามของ "ไม่ถูก penalize") และแยก test
กรณี α ใหญ่พอ (10.0) ที่ทุกตัวเป็นศูนย์จริงออกมาต่างหาก

## ไฟล์

- `src/ml/regression/elastic_net.py`
- `tests/ml/regression/test_elastic_net.py` (28 tests)

## ค้างอยู่

- card #4 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/elastic_net.md -->
### ผลจริงของ `elastic_net`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.0668 | 0.0539 | 0.0169 |
| with_spike | test (one-shot) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.1174 | 0.0731 | 0.2758 |
| non_spike | validation (selection score) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.0643 | 0.0510 | 0.0906 |
| non_spike | test (one-shot) | {"alpha": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-06} | 0.1230 | 0.0736 | 0.2059 |
<!-- END GENERATED -->
