# kmeans — clustering · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #18](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | k | silhouette | inertia | cluster sizes |
| --- | ---: | --- | --- | --- |
| `with_spike` | 2 | **0.3666** | 1075.14 | 73 / 298 |
| `non_spike` | 2 | 0.3298 | 1448.93 | 267 / 104 |

reload ✅ · preprocess `standardize` · เกณฑ์เลือก = train silhouette ตาม config

**5 candidates (`with_spike`)**

| k | silhouette | inertia |
| ---: | --- | --- |
| **2** | **0.3666** | 1075.14 |
| 3 | 0.2805 | 968.72 |
| 4 | 0.2671 | 858.95 |
| 5 | 0.2437 | 851.70 |
| 6 | 0.2008 | 790.55 |

## ข้อสังเกต

- **k=2 ชนะชัดเจน** และ silhouette ลดลงแบบ monotonic เมื่อ k เพิ่ม → โครงสร้างของข้อมูล
  แบ่งเป็นสองกลุ่มตามธรรมชาติ สอดคล้องกับ target แบบ binary (ผันผวนสูง / ไม่สูง)
- สัดส่วนกลุ่มของ `with_spike` คือ **73 / 298 ≈ 20% / 80%** ซึ่งใกล้เคียงกับสัดส่วนคลาสจริง
  ในชุด train (433/1733 ≈ 25%) → การแบ่งกลุ่มแบบไม่มี label จับโครงสร้างเดียวกับ target ได้บ้าง
  **เป็นประเด็นที่ควรขยายในรายงาน** (unsupervised structure สอดคล้องกับ supervised label)
- inertia ลดลงทุกครั้งที่เพิ่ม k ตามที่คณิตศาสตร์กำหนด (มี test ยืนยัน) จึงใช้เลือก k ไม่ได้
  ต้องใช้ silhouette

## Parity

**partition ตรง sklearn ครบทั้ง 5 ค่า k** และ `inertia ≤ sklearn` ทุกค่า

### ปัญหาที่เจอและแก้แล้ว — RNG call sequence ของ k-means++

รอบแรก k=2,3 ผ่าน แต่ **k=4,5,6 ได้ inertia แย่กว่า sklearn** (gap 0.01% / 0.03% / 0.09%)

วินิจฉัยก่อนแก้: เพิ่ม `n_init` จาก 10 → 200 แล้ว gap ปิดเป็น **0.00%** พอดี
→ พิสูจน์ว่า **Lloyd loop ถูกต้อง** ปัญหาอยู่ที่ init พา restart ไปเจอ local optimum คนละจุด

จากนั้นอ่าน source ของ `sklearn.cluster._kmeans._kmeans_plusplus` แล้วพบว่าต่างกัน 3 จุด:

| จุด | ของเดิม (ผิด) | sklearn |
| --- | --- | --- |
| center แรก | `rng.randint(n)` | **`rng.choice(n, p=w/w.sum())`** — consume RNG คนละแบบ |
| การสุ่มแต่ละรอบ | `rng.random_sample(...)` | **`rng.uniform(size=...)`** |
| ก่อน cluster | ใช้ X ดิบ | **ลบ column mean ออกก่อน** แล้วบวกกลับตอนจบ |

ข้อที่ 3 ไม่เปลี่ยน partition ในเชิงคณิตศาสตร์ แต่เปลี่ยน floating point และเปลี่ยน
ค่าที่ k-means++ สุ่มได้ **แก้ครบทั้ง 3 จุดแล้ว init ตรงกันเป๊ะทุก k (max diff 0.00e+00)**

> บทเรียนทั่วไป: เมื่อโมเดลที่มี RNG ไม่ตรง sklearn ให้ทดสอบก่อนว่า **เพิ่ม budget แล้ว gap ปิดไหม**
> ถ้าปิด แปลว่าอัลกอริทึมถูกและปัญหาอยู่ที่ init/RNG ไม่ใช่ที่ตัว optimizer

## การตัดสินใจ

- k-means++ สุ่ม `2 + int(log(k))` candidate ต่อรอบแล้วเลือกตัวที่ลด inertia ได้มากสุด
  (ไม่ใช่สุ่มมาตัวเดียว) ตามที่ sklearn ทำ
- cluster id ไม่มีความหมายเชิงลำดับ → เทียบ parity ด้วย `align_clusters()` จาก `core/compare.py`
- `predict` สำหรับจุดใหม่ = centroid ที่ใกล้ที่สุด (k-Means มี OOS predict ตามธรรมชาติ
  ต่างจาก `agglomerative`)

## ไฟล์

- `src/ml/clustering/kmeans.py`
- `tests/ml/clustering/test_kmeans.py` (22 tests)

## ค้างอยู่

- card #18 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
