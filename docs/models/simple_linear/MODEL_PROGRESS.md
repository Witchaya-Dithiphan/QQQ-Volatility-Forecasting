# simple_linear — regression · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #1](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | ฟีเจอร์ที่ชนะ | RMSE | MAE | R² | reload |
| --- | --- | --- | --- | --- | --- |
| `with_spike` | `historical_volatility_20d` | 0.069038 | — | **−0.049847** | ✅ |
| `non_spike` | `return_1d` | 0.066479 | — | **+0.026542** | ✅ |

**8 candidates เรียงตาม RMSE (`with_spike`)**

| ฟีเจอร์ | RMSE | R² |
| --- | --- | --- |
| `historical_volatility_20d` | 0.069038 | −0.049847 |
| `historical_volatility_5d` | 0.071352 | −0.121390 |
| `return_1d` | 0.073906 | −0.203116 |
| `volume_zscore_20` | 0.074294 | −0.215782 |
| `return_5d` | 0.075566 | −0.257767 |
| `intraday_range` | 0.075910 | −0.269220 |
| `sma_ratio_5_20` | 0.076167 | −0.277855 |
| `rsi_14` | 0.081931 | −0.478559 |

## ข้อสังเกตที่ใช้ในรายงานได้

1. **ทุกฟีเจอร์เดี่ยวให้ R² ติดลบใน `with_spike`** — ไม่มีตัวไหนทำนายได้ดีกว่าเดาค่าเฉลี่ย
   ยืนยันซ้ำกับ `multiple_linear` ว่าโมเดลเชิงเส้นไม่พอสำหรับปัญหานี้
2. **variant ต่างกันเลือกคนละฟีเจอร์** — `with_spike` เลือก `historical_volatility_20d`
   (ตัวที่เรียบที่สุด ทนต่อ outlier) ส่วน `non_spike` เลือก `return_1d` ได้
   เพราะเมื่อตัด spike ออก `return_1d` ไม่ถูก outlier ครอบงำอีกต่อไป
3. `rsi_14` แย่ที่สุด (R² −0.48) — oscillator ที่ bounded 0–100 ไม่มีข้อมูลเรื่องขนาดความผันผวน

## Parity

exact · เทียบกับ `LinearRegression(fit_intercept=True)` **ครบทั้ง 8 ฟีเจอร์** ผ่าน `allclose(rtol=1e-5)`

## การตัดสินใจ

- `feature` รับได้ทั้ง index (int) และชื่อ (str) เพราะ `configs/modeling.json` เขียน grid เป็นชื่อ
- ฟีเจอร์คงที่ (variance = 0) → slope = 0 และ intercept = mean แทนที่จะปล่อย `nan`
  (เป็นคำตอบ least-squares ที่ถูกต้องของกรณีนั้น และทำให้ metric ยังคำนวณได้)
- ตัวหารใช้ `Σ(x−x̄)²` ดิบ ไม่ใช่ `var(ddof=1)` — ผสมสองแบบจะคลาด n/(n−1)

## ไฟล์

- `src/ml/regression/simple_linear.py`
- `tests/ml/regression/test_simple_linear.py` (13 tests)

## ปัญหาที่เจอและแก้แล้ว

**trainer ไม่เคารพ `preprocess` ใน config** — พบตอนจะเริ่มตัวนี้ว่า trainer standardize ทุกโมเดลเสมอ
ทั้งที่ config ระบุ `preprocess: none` ให้ 7 โมเดล (รวม `simple_linear` และ 5 ตัวของอีก agent)

parity ไม่พัง (ทั้งสองฝั่งได้ X เดียวกัน) แต่รายงานจะบรรยาย preprocessing ผิด และ
`gaussian_nb` จะได้ `var_smoothing` คนละสเกลกับที่ตั้งใจ (เพราะมันนิยามเทียบกับ variance สูงสุด)

**แก้:** `trainer.config_preprocess()` อ่านจาก config · `preprocess="none"` ข้าม standardize ·
PCA บังคับ standardize เสมอเพราะต้องการ input ที่ centered · บันทึก `preprocess` ลง
`config.json` และ metadata ของ `preprocessor.npz` · เพิ่ม 4 tests

## ค้างอยู่

- card #1 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
