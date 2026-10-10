# polynomial — regression · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #3](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | feature | degree | RMSE | R² |
| --- | --- | ---: | --- | --- |
| `with_spike` | `historical_volatility_20d` | 3 | 0.069065 | **−0.050652** |
| `non_spike` | `return_5d` | 2 | 0.066056 | **+0.038890** |

reload ✅ · preprocess `standardize` · 16 candidates (degree {2,3} × feature 8)

**4 อันดับแรก (`with_spike`)**

| | feature | RMSE | R² |
| ---: | --- | --- | --- |
| deg3 | `historical_volatility_20d` | 0.069065 | −0.050652 |
| deg3 | `historical_volatility_5d` | 0.069201 | −0.054796 |
| deg2 | `return_5d` | 0.070082 | −0.081816 |
| deg3 | `return_5d` | 0.070369 | −0.090712 |

## ข้อสังเกต

- **เพิ่ม degree แล้วไม่ช่วยเลยใน `with_spike`** — polynomial deg3 (R² −0.0507) แย่กว่า
  `simple_linear` deg1 บนฟีเจอร์เดียวกัน (R² −0.0498) เล็กน้อย → ความไม่เชิงเส้นแบบ univariate
  ไม่ใช่สิ่งที่ขาด ปัญหาอยู่ที่ spike ครอบงำ
- **`non_spike` ให้ R² ดีที่สุดในบรรดา 3 regression ที่ทำแล้ว** (+0.0389 vs multiple +0.0165,
  simple +0.0265) และเลือก `return_5d` deg2 — พอตัด spike ออก ความโค้งเริ่มมีประโยชน์จริง
- ยืนยัน pattern regression ครบ 3/3: `with_spike` ติดลบ · `non_spike` บวก

## Parity

exact ครบ **16/16 candidates** (ทดสอบทุก degree × ทุก feature แยกกัน)
เทียบกับ `make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())`

## การตัดสินใจ

- **standardize ก่อนยกกำลัง** ตาม `standardize_before_powers: true` ใน config —
  มี test ยืนยันว่า `(standardised x)²` ≠ `standardise(x²)` และลำดับสลับกันให้ผลคนละเรื่อง
- `include_bias=False` ใน `PolynomialFeatures` ไม่งั้นจะได้ column ซ้ำกับ intercept
  ทำให้ design matrix rank-deficient
- ไม่มี interaction term — เป็น univariate powers ล้วนตาม `scope` ใน config
- `lstsq` เพราะ design ของ degree 3 บนฟีเจอร์ที่มี fat tail conditioning แย่
- เก็บ mean/scale ของฟีเจอร์ไว้ใน `state_dict` ด้วย ไม่งั้น reload แล้วแปลงค่าคนละแบบ

## ไฟล์

- `src/ml/regression/polynomial.py`
- `tests/ml/regression/test_polynomial.py` (27 tests)

## ปัญหาที่เจอ

ไม่มี — ผ่าน parity รอบแรกทั้ง 16 candidates

## ค้างอยู่

- card #3 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
