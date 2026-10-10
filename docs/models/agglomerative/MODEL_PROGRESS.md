# agglomerative — clustering · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #19](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | k | linkage | silhouette | cluster sizes |
| --- | ---: | --- | --- | --- |
| `with_spike` | 2 | ward | 0.3602 | 293 / 78 |
| `non_spike` | 2 | average | **0.4642** | **359 / 12** ⚠️ |

reload ✅ · preprocess `standardize` · 10 candidates (k 2–6 × linkage {ward, average})

## ⚠️ silhouette ของ `non_spike` สูงแบบหลอกตา — ต้องเขียนเตือนในรายงาน

0.4642 ดูดีที่สุดในบรรดา clustering ทั้งหมด แต่มาจากการแบ่ง **359 / 12** คือแยก outlier
12 จุดออกมาเป็นกลุ่มเดียว ไม่ใช่การค้นพบโครงสร้างที่มีความหมาย

silhouette ให้รางวัลกับการแบ่งแบบนี้โดยธรรมชาติ (กลุ่มเล็กที่อยู่ห่างมาก ๆ ได้คะแนนเต็ม)
เทียบกับ `with_spike` ที่ใช้ ward แล้วได้ 293/78 ซึ่งสมดุลกว่าและสะท้อนโครงสร้างจริงมากกว่า

**ข้อเสนอสำหรับรายงาน:** รายงาน silhouette คู่กับ cluster size เสมอ และอภิปรายว่า
average linkage มีแนวโน้มสร้าง chain/outlier cluster ขณะที่ ward บังคับให้กลุ่มสมดุลกว่า
เพราะมันลด within-cluster variance

## ⚠️ ไม่มี `predict` สำหรับข้อมูลใหม่โดยธรรมชาติ

Agglomerative ให้ **labelling ของ train set** ไม่ใช่กฎสำหรับจุดใหม่ — config แก้ด้วย
`oos_extension: frozen_nearest_centroid` คือ freeze centroid ของแต่ละกลุ่มหลัง fit
แล้วส่งจุดใหม่ไป centroid ที่ใกล้ที่สุด

**ส่วนขยายนี้เราเพิ่มเอง ไม่ใช่ส่วนหนึ่งของอัลกอริทึม → ต้องประกาศเป็นข้อจำกัดในรายงาน**
(ตรงกับ R9 ใน `RISKS_AND_DOD.md`) · `predict` บน train set คืน label เดิมที่ fit ไว้ ไม่ใช่
ผลของ nearest-centroid เพื่อไม่ให้ผลที่รายงานเพี้ยนจากการ approximate ตัวเอง

## Parity

**partition ตรง sklearn ครบ 10/10 candidates** (k 2–6 × ward/average)
ทดสอบเพิ่มบนข้อมูลสุ่ม 4 มิติที่กลุ่มทับกัน ซึ่งเป็นที่ที่ tie-breaking มีผลจริง

## การตัดสินใจ

- ใช้ **Lance-Williams recurrence** อัปเดต cost หลัง merge แทนการคำนวณใหม่ทั้งก้อน —
  เป็นวิธีที่ scipy/sklearn ใช้ และเป็นทางเดียวที่ตัวเลขจะตรงกัน

  ```text
  d(A∪B, C) = αₐ·d(A,C) + α_b·d(B,C) + β·d(A,B) + γ·|d(A,C) − d(B,C)|
  ```

- `ward`: cost ของ singleton สองจุด = `d²/2` (ส่วนเพิ่มของ within-cluster variance)
  · `average`: ระยะเฉลี่ยของทุกคู่ข้ามกลุ่ม (**ไม่ใช่** ระยะระหว่าง centroid ซึ่งเป็น centroid linkage คนละตัว)
- tie เมื่อ cost เท่ากัน → `np.argmin` สแกนแบบ row-major จึงได้คู่ที่ index น้อยกว่าก่อน
  ตรงกับพฤติกรรม sklearn

## ไฟล์

- `src/ml/clustering/agglomerative.py`
- `tests/ml/clustering/test_agglomerative.py` (25 tests)

## ปัญหาที่เจอ

ไม่มี parity failure — ผ่าน 10/10 รอบแรก การใช้ Lance-Williams ตั้งแต่ต้นคือเหตุผลหลัก

## ค้างอยู่

- card #19 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
