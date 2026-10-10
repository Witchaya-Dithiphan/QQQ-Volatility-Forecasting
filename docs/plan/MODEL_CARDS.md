# MODEL CARDS — สเปก 19 โมเดล

> สเปกรายตัวที่ใช้ลงมือเขียนโค้ดได้เลย โดยไม่ต้องอ่านบทสนทนาหรือถามใคร
> กฎกลางอยู่ใน [`AGENTS.md`](../../AGENTS.md) · เหตุผลเชิงสถาปัตยกรรมอยู่ใน [`ARCHITECTURE.md`](ARCHITECTURE.md)
> **grid ทุกตัวมาจาก `configs/modeling.json` ห้ามแต่งเอง** — ชื่อโมเดล = key ใน config = ชื่อไฟล์ = `BaseModel.name`

---

## 0. เรื่องที่ต้องอ่านก่อนเขียนโมเดลตัวแรก

### 0.1 RNG — กับดักที่ทำให้ parity พังโดยที่อัลกอริทึมถูกต้อง

sklearn ทั้งไลบรารีสร้างบน **legacy MT19937** (`np.random.RandomState`) ไม่ใช่ `np.random.default_rng` (PCG64)
พิสูจน์แล้วบนเครื่องนี้:

```text
sklearn._forest._generate_sample_indices(42, 200, 200, None) -> [102 179  92  14 106  71 188  20]
np.random.RandomState(42).randint(0, 200, 200)               -> [102 179  92  14 106  71 188  20]   ตรง
np.random.default_rng(42).integers(0, 200, 200)              -> [ 17 154 130  87  86 171  17 139]   ไม่ตรง
```

**ใช้ `np.random.RandomState(seed)` เสมอในโมเดลที่ต้องเทียบ RNG กับ sklearn** (`random_forest`, `mlp`, `slp`, `kmeans`)

### 0.1.1 API ของ scikit-learn 1.9 ที่เปลี่ยนไป

`penalty=` ของ `LogisticRegression` **deprecated ตั้งแต่ 1.8 และจะถูกถอดใน 1.10** เวอร์ชันที่ lock ไว้คือ 1.9.1
จึงต้องเขียนแบบใหม่ ไม่งั้นจะได้ `FutureWarning` เต็มไปหมดและพังเมื่ออัปเกรด:

| เดิม | ใช้แทนด้วย |
| --- | --- |
| `penalty="l2"` | `l1_ratio=0` |
| `penalty="l1"` | `l1_ratio=1` |
| `penalty="elasticnet"` | `l1_ratio=<float>` |
| `penalty=None` | `C=np.inf` |

`AdaBoostClassifier` ก็ถอด `algorithm="SAMME.R"` ออกแล้วและ `SAMME` เป็นค่า default — ดู card #15

### 0.1.2 สิ่งที่ card นี้ override จาก `configs/modeling.json`

`references` ใน config เป็นของเดิมจากสถาปัตยกรรมที่เลิกใช้แล้ว (ADR-001) และบางตัวเลือก reference
ที่ทำให้ parity ไม่มีทางผ่าน card นี้จึงแก้ไว้ 2 จุด โดยมีหลักฐานการทดลองกำกับ:

| โมเดล | config เดิม | card นี้ใช้ | เหตุผล |
| --- | --- | --- | --- |
| `logistic` | `SGDClassifier` | **`LogisticRegression`** | SGD อัปเดตทีละตัวอย่าง ส่วน scratch เป็น full-batch → path ต่างกันถาวร แต่ objective เป็น convex จึงมี optimum เดียว เทียบกับ LBFGS แล้ว**ตรงกันที่ 1e-8** (พิสูจน์แล้ว ดู card #5) |
| `svm` | `LinearSVC` (เทียบน้ำหนัก) | `LinearSVC` (**เทียบ objective**) | hinge ไม่เรียบ ทั้งสอง solver หยุดคนละจุดใกล้ optimum — วัดจริงที่ C=1.0 พบว่า **scratch ได้ objective ต่ำกว่า sklearn** (34.8258 < 34.9859) คือเราดีกว่า ไม่ใช่ผิด ดู card #17 |

### 0.2 นิยาม "ตรงกับ reference" รายโมเดล

มติของโปรเจกต์คือ strict เท่ากันทุกตัว (`AGENTS.md` §6) แต่ต้นทุนการไล่ให้ตรงต่างกันมาก
คอลัมน์ **parity** ในตารางข้างล่างบอกว่าแต่ละตัวอยู่กลุ่มไหน:

| กลุ่ม | ความหมาย | ทำอย่างไร |
| --- | --- | --- |
| **exact** | ไม่มี RNG ไม่มี iteration — ต้องตรงระดับ 1e-10 หรือดีกว่า | เทียบ `predict` ตรง ๆ |
| **seeded** | มี RNG แต่ไล่ให้ตรงได้ถ้าก๊อป call sequence ของ sklearn | ใช้ `RandomState` + ลำดับการเรียกเดียวกัน |
| **iterative** | ผลขึ้นกับลำดับ update และ learning rate | ตั้ง hyperparameter ให้ตรงเป๊ะแล้วเทียบ ถ้ายังต่างให้เทียบที่ objective value และบันทึกเหตุผล |

ถ้าไล่ไม่ตรงจริง ๆ **ห้ามลดเกณฑ์เงียบ ๆ และห้ามเอาผล reference มาสวมแทน scratch** —
บันทึกใน card ของตัวเองว่าต่างตรงไหนเพราะอะไร แล้วแจ้งเจ้าของโปรเจกต์ (`AGENTS.md` §6)

---

## 1. ตารางรวม 19 โมเดล

| # | ชื่อ (= key ใน config) | task | เจ้าของ | preprocess | candidates | parity | ยาก | ชม. |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | `simple_linear` | regression | Claude | none | 8 | exact | ง่าย | 1 |
| 2 | `multiple_linear` | regression | Claude | standardize | 1 | exact | ง่าย | 0.5 |
| 3 | `polynomial` | regression | Claude | standardize | 16 | exact | ง่าย | 1 |
| 4 | `elastic_net` ★ | regression | Claude | standardize | 12 | iterative | กลาง | 3 |
| 5 | `logistic` | classification | Claude | standardize | 6 | iterative | กลาง | 2.5 |
| 6 | `gaussian_nb` | classification | Claude | none | 3 | exact | ง่าย | 1 |
| 7 | `knn` | classification | Claude | standardize | 8 | exact | ง่าย | 1.5 |
| 8 | `perceptron` | classification | Claude | standardize | 3 | iterative | ง่าย | 1.5 |
| 9 | `slp` | classification | Claude | standardize | 8 | iterative | กลาง | 2.5 |
| 10 | `kmeans` | clustering | Claude | standardize | 5 | seeded | กลาง | 2 |
| 11 | `agglomerative` | clustering | Claude | standardize | 10 | exact | กลาง | 2.5 |
| 12 | `decision_tree` | classification | Codex | none | 6 | exact | กลาง | 3 |
| 13 | `random_forest` | classification | Codex | none + PCA | 7 | seeded | กลาง | 2.5 |
| 14 | `gradient_boosting` | classification | Codex | none | 8 | exact | ยาก | 3.5 |
| 15 | `xgboost` | classification | Codex | none | 8 | exact | ยาก | 4 |
| 16 | `adaboost` ★ | classification | Codex | none | 6 | exact | กลาง | 3 |
| 17 | `stacking` | classification | Codex | base-specific | 1 | iterative | ยาก | 3.5 |
| 18 | `svm` | classification | Codex | standardize + PCA | 12 | iterative | ยาก | 3.5 |
| 19 | `mlp` | classification | Codex | standardize | 12 | iterative | ยาก | 4 |

★ = extracurricular ที่โจทย์บังคับให้ทำ **ทั้ง scratch และ ready-to-use** และอธิบายกลไกภายในอย่างละเอียด

รวมฝั่ง Claude ≈ 19 ชม. · ฝั่ง Codex ≈ 27 ชม. — ฝั่ง Codex หนักกว่าโดยตั้งใจ เพราะ tree family
reuse กันเองได้สูง (เขียน `_best_split` + stump ครั้งเดียว ใช้ต่อใน DT → RF → GB → XGB → AdaBoost)

**PCA ไม่ใช่โมเดล** แต่เป็น config ของ `random_forest` และ `svm` ตามที่โจทย์ระบุ
*"Dimensionality reduction for random forest/support vector machine"* — components `[2, 4, 6]` บวกกรณีไม่ใช้ PCA

---

## 2. Regression

### 1. `simple_linear` — Simple Linear Regression

**ไฟล์:** `src/ml/regression/simple_linear.py` · Claude · parity **exact**

**กลไก** OLS บนฟีเจอร์เดียว ปิดรูปได้ ไม่ต้องวนซ้ำ

```text
b1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
b0 = ȳ - b1·x̄
ŷ  = b0 + b1·x
```

**Grid** `feature` ∈ 8 ฟีเจอร์ → 8 candidates (เลือกฟีเจอร์ที่ดีที่สุดบน Validation)

**Preprocess** `none` — ไม่ต้อง standardize เพราะ univariate OLS ไม่ไวต่อสเกล

**Reference** `LinearRegression(fit_intercept=True)` fit บน `X[:, [i]]`

**จุดที่ parity พลาด** ตัวหารต้องใช้ `Σ(xi - x̄)²` ไม่ใช่ `var` ที่มี `ddof=1` — ถ้าใช้ `np.var(x, ddof=1)`
ต้องหารด้วย `n-1` ทั้งเศษและส่วนให้สอดคล้องกัน · ถ้าฟีเจอร์คงที่ ตัวส่วนเป็น 0 ต้องคืน
`measure(reason=...)` ไม่ใช่ปล่อย `nan`

**Unit test** `x=[1,2,3], y=[2,4,6]` → `b1=2, b0=0` เป๊ะ

**ความเสี่ยง** ไม่มี เป็นโมเดลที่ควรเขียนเป็นตัวแรกเพื่อทดสอบ framework

---

### 2. `multiple_linear` — Multiple Linear Regression (pilot)

**ไฟล์:** `src/ml/regression/multiple_linear.py` · Claude · parity **exact**

**กลไก** ลด `||y - Xb||²` แก้ด้วย pseudo-inverse / lstsq (SVD) ไม่ใช่ `inv(XᵀX)`

```text
b = pinv(X_design) @ y        โดย X_design = [1, X]
```

**Grid** ว่าง → 1 candidate · config ระบุ `solver: numpy.linalg.pinv`

**Preprocess** `standardize`

**Reference** `LinearRegression(fit_intercept=True)`

**จุดที่ parity พลาด** ใช้ normal equation `inv(XᵀX) @ Xᵀy` จะยกกำลังสองของ condition number
(ชุดนี้ cond = 35.6 → 1267) ทำให้ต่างจาก sklearn ที่ระดับ 1e-6 แทนที่จะเป็น 1e-15 ·
`np.linalg.lstsq` กับ `pinv` ให้ค่าเท่ากันทั้งคู่และตรงกับ sklearn ซึ่งใช้ `scipy.linalg.lstsq`

**Unit test** ระบบที่มีคำตอบเดียว `X=[[1,0],[0,1],[1,1]], y=[2,3,5]`, `fit_intercept=False` → `coef=[2,3]`

**ความเสี่ยง** ไม่มี — ใช้เป็น pilot พิสูจน์ framework ทั้งลูป

---

### 3. `polynomial` — Polynomial Regression

**ไฟล์:** `src/ml/regression/polynomial.py` · Claude · parity **exact**

**กลไก** ยกกำลังฟีเจอร์เดียว **ไม่มี interaction term** แล้วทำ OLS

```text
design = [1, z, z², ..., z^d]   โดย z = standardize(x) ก่อนยกกำลัง
b = lstsq(design, y)
```

**Grid** `degree` ∈ {2, 3} × `feature` ∈ 8 ตัว = **16 candidates** (ตรงกับ `polynomial_max_candidates: 16`)

**Preprocess** `standardize` และ config ระบุ `standardize_before_powers: true` — **ต้อง standardize ก่อนยกกำลัง**

**Reference** `make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())`
บน `z[:, [i]]` ที่ standardize แล้ว

**จุดที่ parity พลาด** ลำดับ standardize↔ยกกำลังสลับกันให้ผลคนละเรื่อง (`(x²)` ที่ scale แล้ว ≠ `(scaled x)²`) ·
`PolynomialFeatures` ใส่ bias column มาด้วยถ้าไม่ปิด จะได้ column ซ้ำกับ intercept ·
degree 3 บนฟีเจอร์ที่มี outlier ทำให้ design matrix ill-conditioned — ใช้ lstsq ไม่ใช่ normal equation

**Unit test** `y = 1 + 2z + 3z²` พอดี ๆ → ค่าสัมประสิทธิ์กลับมา `[1, 2, 3]` ภายใน 1e-10

**ความเสี่ยง** ต่ำ

---

### 4. `elastic_net` ★ — Elastic Net (extracurricular, regression)

**ไฟล์:** `src/ml/regression/elastic_net.py` · Claude · parity **iterative**

> โจทย์บังคับให้โมเดลนี้มี **ทั้ง scratch และ ready-to-use** และต้องอธิบายกลไกภายในอย่างละเอียดในรายงาน
> card นี้จึงเขียนลึกกว่าตัวอื่น และเนื้อหาส่วน "กลไก" ให้ยกไปใช้ในรายงานได้โดยตรง

**ทำไมโมเดลนี้เหมาะกับชุดข้อมูลนี้** วัดจาก train set จริง: condition number = **35.6**,
`historical_volatility_5d ~ historical_volatility_20d` ให้ |r| = **0.804** ซึ่งแปลว่า Multiple Regression
จะได้สัมประสิทธิ์ที่ไม่เสถียร (สลับเครื่องหมาย/พองผิดปกติ) Elastic Net แก้ตรงจุดนี้ด้วยการรวม
penalty สองแบบ: L1 ตัดฟีเจอร์ที่ซ้ำซ้อนทิ้ง ส่วน L2 กระจายน้ำหนักให้กลุ่มฟีเจอร์ที่สัมพันธ์กันสูง
แทนที่จะเลือกมาตัวเดียวแบบสุ่มอย่างที่ Lasso ล้วนทำ

**กลไกโดยละเอียด**

Objective (ตรงกับ sklearn และตรงกับ `objective` ใน config):

```text
J(b) = (1 / (2n)) · ||y - Xb||²  +  α·ρ·||b||₁  +  (α·(1-ρ)/2)·||b||²
         เมื่อ ρ = l1_ratio
```

แก้ด้วย **coordinate descent** เพราะพจน์ L1 ไม่ differentiable ที่ 0 จึงใช้ gradient descent ตรง ๆ ไม่ได้
หลักการคือ fix ทุกสัมประสิทธิ์ไว้ แล้วหา optimum ของทีละตัวซึ่งมีคำตอบปิดรูป:

```text
ทำซ้ำจนลู่เข้า (cyclic: j = 0, 1, ..., p-1):
    r_j   = y - X·b + X[:,j]·b_j          # residual ที่ยังไม่รวมผลของ feature j
    ρ_j   = (1/n) · X[:,j]ᵀ · r_j         # correlation ของ feature j กับ residual
    z_j   = (1/n) · X[:,j]ᵀ · X[:,j]      # = 1 พอดีถ้า standardize แล้ว
    b_j   = S(ρ_j, α·ρ) / (z_j + α·(1-ρ))

soft-threshold:  S(v, γ) = sign(v)·max(|v| - γ, 0)
```

`S` คือหัวใจของ L1: ถ้า correlation ของฟีเจอร์กับ residual น้อยกว่าเกณฑ์ γ สัมประสิทธิ์จะถูกผลักเป็น **0 พอดี**
(ไม่ใช่แค่เล็กลงอย่าง Ridge) นี่คือเหตุผลที่ Elastic Net เลือกฟีเจอร์ได้ในตัว ส่วนตัวหาร `(z_j + α(1-ρ))`
คือผลของ L2 ที่หดสัมประสิทธิ์ทุกตัวแบบ proportional

**intercept** ไม่ถูก penalize — คำนวณแยกเป็น `b0 = ȳ - x̄ᵀb` หลังจบแต่ละรอบ

**เงื่อนไขหยุด** `max|Δb| < tol` (1e-6) หรือครบ `max_iter` (5000)

**Grid** `alpha` ∈ {0.0001, 0.001, 0.01, 0.1} × `l1_ratio` ∈ {0.25, 0.5, 0.75} = **12 candidates**

**Preprocess** `standardize` — จำเป็นจริง ไม่ใช่ทางเลือก เพราะ penalty เดียวกันถูกใช้กับทุกฟีเจอร์
ถ้าสเกลต่างกันจะเท่ากับลงโทษฟีเจอร์ที่สเกลเล็กหนักกว่าโดยไม่ตั้งใจ

**Reference** `ElasticNet(alpha=α, l1_ratio=ρ, fit_intercept=True, max_iter=5000, tol=1e-6, selection="cyclic")`

**จุดที่ parity พลาด — อ่านให้ครบก่อนเขียน**

1. **การหารด้วย n** sklearn ใช้ `1/(2n)` ในพจน์ squared error ถ้า scratch ใช้ `1/2` เฉย ๆ
   ค่า α ที่ได้ผลเท่ากันจะต่างกัน n เท่า → ผลไม่มีวันตรง
2. **`selection="cyclic"` ไม่ใช่ `"random"`** — ต้องวน j ตามลำดับ 0..p-1 เท่านั้น
3. **เกณฑ์หยุดของ sklearn** ใช้ dual gap ไม่ใช่ `max|Δb|` จึงอาจหยุดคนละรอบ
   ตั้ง `tol` ให้เล็ก (1e-6) และ `max_iter` ให้ใหญ่ (5000) เพื่อให้ทั้งคู่ลู่เข้าจุดเดียวกัน
   แล้วเทียบที่ **ค่า objective** ด้วย ไม่ใช่เทียบเฉพาะ `predict`
4. **`z_j` = 1 เมื่อ standardize ด้วย ddof=0** ซึ่งเป็นค่าที่ `Standardizer` ของเราใช้ — อย่า hard-code 1
   ให้คำนวณจริงเผื่อฟีเจอร์คงที่
5. `alpha=0` ทำให้กลายเป็น OLS และ sklearn จะเตือน — ไม่มีใน grid นี้จึงไม่ต้องรองรับ

**Unit test กลไกภายใน**

- soft-threshold: `S(0.5, 0.2) = 0.3`, `S(-0.5, 0.2) = -0.3`, `S(0.1, 0.2) = 0` เป๊ะ
- `l1_ratio=1` (Lasso ล้วน) + α ใหญ่ → สัมประสิทธิ์เป็น 0 ทั้งหมด
- ข้อมูลที่ฟีเจอร์ 2 ตัวเหมือนกันเป๊ะ → Elastic Net ต้องแบ่งน้ำหนักให้ทั้งคู่ใกล้เคียงกัน
  (ต่างจาก Lasso ที่เลือกตัวเดียว) — เป็น test ที่พิสูจน์ว่า L2 ทำงานจริง และใช้ในรายงานได้ด้วย
- objective ต้องลดลงทุก iteration (monotone) — จับด้วย `history_`

**ความเสี่ยง** กลาง — ถ้าไล่ไม่ตรง sklearn ภายใน 3 ชม. ให้เทียบที่ objective value แทน predict
แล้วบันทึกว่า "ลู่เข้าจุดเดียวกัน แต่หยุดคนละ iteration เพราะเกณฑ์ dual gap" ซึ่งยังเป็นหลักฐานที่ยอมรับได้

---

## 3. Classification

### 5. `logistic` — Logistic Regression

**ไฟล์:** `src/ml/classification/logistic.py` · Claude · parity **iterative**

**กลไก** full-batch gradient descent บน weighted BCE + L2 (ตาม config ไม่ใช่ Newton/LBFGS)

```text
p     = σ(Xb + b0),  σ(t) = 1/(1 + e^(-t))
J     = Σ wi·BCE(yi, pi) / Σ wi  +  (λ/2)·||b||²      (ไม่ penalize b0)
∇b    = Xᵀ(w·(p - y)) / Σw + λ·b
b    ← b - lr·∇b        lr = 0.05, max_iter = 5000, tol = 1e-6
```

`class_weight="balanced"` → `wi = n_samples / (2 · count(classของ i))` ตาม `balanced_weight` ใน config

**Grid** `class_weight` ∈ {null, "balanced"} × `l2` ∈ {0, 0.001, 0.01} = **6 candidates**

**Preprocess** `standardize`

**Reference** **`LogisticRegression`** (override จาก config ที่ระบุ `SGDClassifier` — ดู §0.1.2)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

sample_weight_sum = len(y) if class_weight is None else len(y)   # 'balanced' ก็ได้ n เท่ากัน
if l2 == 0:
    ref = LogisticRegression(C=np.inf, fit_intercept=True, tol=1e-12, max_iter=200000)
else:
    ref = LogisticRegression(C=1.0 / (l2 * sample_weight_sum), l1_ratio=0,
                             fit_intercept=True, tol=1e-12, max_iter=200000)
# class_weight="balanced" ส่งต่อให้ sklearn ตรง ๆ ได้
```

**เหตุผลที่เปลี่ยน reference** `SGDClassifier` อัปเดตทีละตัวอย่าง ส่วน scratch เป็น full-batch
→ เดินคนละ path ตลอดกาล ไม่มีทางตรง แต่ objective นี้เป็น **convex** จึงมี optimum เดียว
LBFGS กับ full-batch GD จึงลู่เข้าจุดเดียวกัน **ทดสอบจริงแล้วตรงกันที่ 1e-8**:

```text
lam=0.01    C=0.25   max abs coef diff = 1.05e-08   max abs prob diff = 7.43e-09
lam=0.001   C=2.50   max abs coef diff = 2.45e-09   max abs prob diff = 1.78e-09
balanced    C=0.25   max abs coef diff = 2.51e-08
lam=0       C=inf    max abs coef diff = 7.37e-10
```

**การแปลง C — ที่มาของสูตร**

```text
ของเรา   : J   = (Σ w_i·BCE_i) / (Σ w_i)  +  (λ/2)·||b||²
sklearn  : J_sk = C · Σ w_i·logloss_i     +  0.5·||b||²
หาร J ด้วย λ แล้วจับคู่สัมประสิทธิ์ →  C = 1 / (λ · Σ w_i)
กรณีไม่ถ่วงน้ำหนัก Σw = n  →  C = 1/(λ·n)
'balanced' ก็ยังได้ Σw = n พอดี เพราะ w_i = n/(2·n_c)
```

**จุดที่ parity พลาด**

- **`C` คูณกับพจน์ loss ไม่ใช่ penalty** — ใช้ `C = 1/λ` เฉย ๆ จะผิดไป n เท่า
- **`penalty=` deprecated ใน sklearn 1.9** ใช้ `l1_ratio=0` และ `C=np.inf` แทน (§0.1.1)
- `tol` ต้องเล็กจริง (1e-12) และ `max_iter` ใหญ่ ไม่งั้น LBFGS หยุดก่อนถึง optimum
- `np.log(σ(t))` overflow เมื่อ `|t|` ใหญ่ — ใช้ `np.logaddexp(0, -margin)` แทน
- intercept ต้องไม่ถูก penalize ทั้งสองฝั่ง (`penalize_intercept: false`)
- scratch ต้องรัน iteration ให้พอ (lr 0.05 ตาม config อาจต้องมากกว่า 5000 รอบถึงจะถึง 1e-8)
  ถ้า `max_iter` ของ config ไม่พอ ให้เทียบที่ **objective value** แทน แล้วบันทึกจำนวนรอบที่ใช้

**Unit test** ข้อมูลแยกกันชัด 2 กลุ่ม → accuracy = 1.0 · gradient check เทียบกับ numerical gradient ที่ 1e-6 ·
objective ลดลงทุก iteration

**ความเสี่ยง** กลาง — เพราะ reference เป็น SGD ที่ path ต่างกัน ดูแผนสำรองข้างบน

---

### 6. `gaussian_nb` — Gaussian Naive Bayes

**ไฟล์:** `src/ml/classification/gaussian_nb.py` · Claude · parity **exact**

**กลไก** ปิดรูปทั้งหมด ไม่มี iteration ไม่มี RNG

```text
สำหรับแต่ละคลาส c:  μ_c, σ²_c  = mean/var ของฟีเจอร์ในคลาสนั้น (ddof=0)
                     prior_c    = n_c / n
σ²_c ← σ²_c + var_smoothing · max(var ของทุกฟีเจอร์ทั้งชุด)
log p(x|c) = -0.5·Σ[ log(2π σ²_cj) + (xj - μ_cj)² / σ²_cj ]
ทำนายคลาสที่ log prior + log likelihood สูงสุด
```

**Grid** `var_smoothing` ∈ {1e-9, 1e-8, 1e-7} = 3 candidates

**Preprocess** `none`

**Reference** `GaussianNB(var_smoothing=v)` — priors ใช้ค่า empirical จาก train (`priors: train_empirical`)

**จุดที่ parity พลาด**

- `var_smoothing` ของ sklearn คูณกับ **variance สูงสุดของทั้งชุดข้อมูล** (`np.var(X, axis=0).max()`)
  ไม่ใช่ variance ของแต่ละคลาส — พลาดข้อนี้แล้วค่าจะต่างเล็กน้อยแต่สม่ำเสมอ
- ใช้ `ddof=0` ตรงกับ sklearn
- ต้องคำนวณใน log space ตลอด (`gaussian_log_space` ใน config) การคูณ likelihood ตรง ๆ จะ underflow

**Unit test** คลาสละ 2 จุดที่คำนวณ μ, σ² มือได้ → เทียบ `theta_` และ `var_` กับ sklearn ที่ 1e-12

**ความเสี่ยง** ต่ำมาก — ควรผ่าน parity ในรอบแรก

---

### 7. `knn` — k-Nearest Neighbors

**ไฟล์:** `src/ml/classification/knn.py` · Claude · parity **exact**

**กลไก** ไม่มีการเทรน เก็บ train ไว้แล้วตอนทำนายหาค่าเพื่อนบ้านที่ใกล้ที่สุด k ตัว

```text
d(x, xi) = ||x - xi||₂
uniform  : vote = นับคลาสของ k ตัวที่ใกล้สุด
distance : vote = Σ (1/di) ของแต่ละคลาส   (ถ้า di = 0 ให้จุดนั้นน้ำหนักเต็มและตัดที่เหลือ)
probability = vote_class1 / vote_total
```

**Grid** `k` ∈ {3, 5, 9, 15} × `weights` ∈ {uniform, distance} = **8 candidates**

**Preprocess** `standardize` — จำเป็น เพราะระยะทางยุคลิดไวต่อสเกลโดยตรง

**Reference** `KNeighborsClassifier(n_neighbors=k, weights=w, algorithm="brute", metric="euclidean")`

**จุดที่ parity พลาด**

- **tie-breaking เมื่อระยะเท่ากัน** sklearn เลือกตามลำดับ index ที่เจอก่อน ให้ใช้
  `np.argsort(d, kind="stable")` เพื่อให้ลำดับเหมือนกัน (config ระบุ `tie: deterministic`)
- `weights="distance"` เมื่อมีจุดที่ระยะ 0 sklearn ให้น้ำหนักจุดนั้นเป็น 1 และตัวอื่นเป็น 0 ไม่ใช่ `inf`
- ใช้ `algorithm="brute"` ใน reference เพื่อตัดความต่างจาก KD-tree ออกไป

**Unit test** 3 จุดที่คำนวณระยะมือได้ · กรณี tie ที่ระยะเท่ากันเป๊ะ → ต้องได้คำตอบเดียวกับ sklearn

**ความเสี่ยง** ต่ำ

---

### 8. `perceptron` — Perceptron

**ไฟล์:** `src/ml/classification/perceptron.py` · Claude · parity **iterative**

**กลไก** online mistake-driven update (ไม่มี loss surface ให้ลด ไม่ใช่ gradient descent)

```text
สำหรับแต่ละ epoch, แต่ละตัวอย่างตามลำดับเดิม (shuffle = false):
    ŷ = sign(w·x + b)         โดย label แปลงเป็น {-1, +1}
    ถ้า ŷ ≠ y:  w ← w + lr·y·x ,  b ← b + lr·y
หยุดเมื่อไม่มี mistake ใน epoch หรือครบ max_epochs (1000)
```

output เป็น **decision score** ไม่ใช่ probability (`output: decision_score`) และต้องรายงาน `mistake_rate`

**Grid** `learning_rate` ∈ {0.001, 0.01, 0.1} = 3 candidates

**Preprocess** `standardize`

**Reference** `Perceptron(eta0=lr, shuffle=False, random_state=42, max_iter=1000, tol=None)`

**จุดที่ parity พลาด**

- `tol=None` **สำคัญมาก** — ค่า default ของ sklearn คือ `tol=1e-3` ซึ่งจะ early-stop คนละรอบกับเรา
- `shuffle=False` ต้องตั้งทั้งสองฝั่ง ไม่งั้นลำดับการ update ต่างกันทันที
- sklearn ไม่คูณ `eta0` กับ update ของ bias ต่างจาก weight — ตรวจกับโค้ดจริงถ้าไม่ตรง
- loss ที่บันทึกใน `history_` ใช้ `perceptron_criterion` = `mean(max(0, -margin))` ตาม config

**Unit test** ข้อมูล 2 มิติที่แยกเชิงเส้นได้ → ต้อง converge และ mistake_rate = 0 ·
ตัวอย่างเดียวที่ทายผิด → ตรวจว่า w เปลี่ยนไปเท่ากับ `lr·y·x` พอดี

**ความเสี่ยง** ต่ำ ถ้าตั้ง `tol=None`

---

### 9. `slp` — Single-Layer Perceptron

**ไฟล์:** `src/ml/classification/slp.py` · Claude · parity **iterative**

**กลไก** 1 ชั้น sigmoid เทรนด้วย mini-batch gradient descent บน BCE — ต่างจาก `perceptron`
ตรงที่มี activation ต่อเนื่องและมี loss ที่หาอนุพันธ์ได้

```text
p = σ(Xw + b)
J = mean BCE(y, p) + (λ/2)||w||²
∇w = Xᵀ(p - y)/batch + λw          (รูปนี้เกิดจาก sigmoid + BCE หักล้างกันพอดี)
batch: ตัดตามลำดับเวลา ไม่สลับ (contiguous_chronological, last_short_batch = true)
early stopping: ดู BCE บน internal tail 15% ของ train, patience 50, min_delta 1e-6
แล้ว refit ด้วยจำนวน epoch ที่ดีที่สุดบน train เต็ม (refit_full_train, refit_early_stopping = false)
```

**Grid** `batch` ∈ {32, "full"} × `l2` ∈ {0, 0.0001} × `learning_rate` ∈ {0.001, 0.01} = **8 candidates**

**Preprocess** `standardize`

**Reference** `MLPClassifier(hidden_layer_sizes=(), ...)` ใช้ไม่ได้ (sklearn ไม่รับชั้นซ่อนว่าง) →
ใช้ `SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, shuffle=False, random_state=42)`
เป็น behavioral reference และเทียบที่ objective

**จุดที่ parity พลาด**

- weight init ต้องใช้ `np.random.RandomState(42)` และบันทึกสูตร init ไว้ใน card ตัวเอง
- batch สุดท้ายที่ไม่เต็มต้องนับด้วย (`last_short_batch: true`) และหารด้วยขนาดจริงของ batch นั้น
- `patience` นับจาก epoch ที่ดีที่สุด ไม่ใช่นับต่อเนื่องจาก epoch ล่าสุด · tie เลือก epoch แรกสุด
- scaler ของ early-stopping fit จาก subtrain เท่านั้น (`scaler_fit: subtrain`) ไม่ใช่ train เต็ม

**Unit test** gradient check เทียบ numerical gradient ที่ 1e-6 · ข้อมูลแยกได้ → BCE ลดลง monotone ·
batch=full ต้องให้ผลเท่ากับ full-batch gradient descent ที่คำนวณมือ

**ความเสี่ยง** กลาง — ส่วนที่พลาดบ่อยคือ early stopping ไม่ใช่ตัว gradient

---

### 10. `mlp` — Multi-Layer Perceptron

**ไฟล์:** `src/ml/classification/mlp.py` · Codex · parity **iterative**

**กลไก** ชั้นซ่อน ReLU + output sigmoid เทรนด้วย backpropagation

```text
forward :  a0 = X ;  zl = a(l-1)·Wl + bl ;  al = relu(zl)        สำหรับชั้นซ่อน
           output  p = σ(z_last)
backward:  δ_last = p - y
           δ_l    = (δ_(l+1) · W_(l+1)ᵀ) ⊙ 1[zl > 0]             อนุพันธ์ ReLU
           ∇Wl    = a(l-1)ᵀ · δ_l / batch + λ·Wl
```

**Grid** `hidden_layers` ∈ {[16], [32], [16,8]} × `l2` ∈ {0, 0.0001} × `learning_rate` ∈ {0.001, 0.01} = **12 candidates**

**Preprocess** `standardize`

**Reference** `MLPClassifier(activation="relu", solver="sgd", batch_size=32, shuffle=False,
early_stopping=False, tol=0, random_state=42, hidden_layer_sizes=..., learning_rate_init=...,
alpha=..., max_iter=<epoch ที่เลือกได้>, n_iter_no_change=<epoch+1>)` ตาม `references.mlp` ใน config

**จุดที่ parity พลาด**

- **weight init** sklearn ใช้ `RandomState` แล้ว uniform ในช่วง `±sqrt(6/(fan_in+fan_out))` สำหรับ ReLU
  ต้องก๊อปสูตรและลำดับการ init ทีละชั้นให้ตรง — นี่คือจุดที่พังบ่อยที่สุด
- sklearn หาร L2 ด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch
- `tol=0` + `n_iter_no_change` ใหญ่กว่า max_iter เพื่อปิด early stopping ภายในของ sklearn
- อนุพันธ์ ReLU ที่ `z = 0` — sklearn ใช้ 0 ให้ใช้ `z > 0` ไม่ใช่ `z >= 0`

**Unit test** gradient check ทุกชั้นเทียบ numerical gradient · XOR ที่มีชั้นซ่อนต้องแก้ได้ (accuracy 1.0)
ซึ่งพิสูจน์ว่า backprop ทำงานจริง · ชั้นซ่อนเดียว + linear activation ต้องเท่ากับ logistic regression

**ความเสี่ยง** สูง — เป็นตัวที่ยากที่สุดคู่กับ XGBoost แผนสำรอง: เทียบที่ learning curve และ final metric
แล้วบันทึกว่า weight init ต่างกัน

---

### 11. `decision_tree` — Decision Tree

**ไฟล์:** `src/ml/classification/decision_tree.py` · Codex · parity **exact**

**เขียนตัวนี้ก่อนเพื่อน** เพราะ `_best_split` และ stump จะถูก reuse ใน RF, GB, XGBoost และ AdaBoost

**กลไก** CART แบบ greedy ใช้ Gini

```text
Gini(node) = 1 - Σ p_c²
Gain = Gini(parent) - [ (n_L/n)·Gini(L) + (n_R/n)·Gini(R) ]
เลือก (feature, threshold) ที่ gain สูงสุด; threshold = จุดกึ่งกลางระหว่างค่าที่ติดกัน
หยุดเมื่อ: ถึง max_depth / ใบมีน้อยกว่า min_samples_leaf / node บริสุทธิ์ / ไม่มี split ที่ gain > 0
```

**Grid** `max_depth` ∈ {2, 3, 5} × `min_samples_leaf` ∈ {5, 10} = **6 candidates**

**Preprocess** `none` — tree ไม่ไวต่อสเกล

**Reference** `DecisionTreeClassifier(criterion="gini", max_depth=d, min_samples_leaf=m,
random_state=42, max_features=None, splitter="best")`

**จุดที่ parity พลาด**

- **tie-breaking** config ระบุ `tie: [feature_index, threshold]` → เมื่อ gain เท่ากัน เลือก feature index
  น้อยกว่าก่อน แล้วจึง threshold น้อยกว่า ใช้ `>` ไม่ใช่ `>=` ตอนเทียบ gain เพื่อให้ตัวแรกชนะ
- `max_features=None` **ต้องตั้งให้ชัด** ไม่งั้น sklearn จะสุ่มฟีเจอร์และ parity พังทันที
- threshold ของ sklearn คือ midpoint `(a+b)/2` ของค่าที่เรียงติดกัน ไม่ใช่ค่าตัวอย่างเอง
- `min_samples_leaf` ต้องเช็ค **ทั้งสองฝั่ง** ของ split ก่อนยอมรับ ไม่ใช่เช็คหลังแบ่ง

**Unit test** 4 ตัวอย่างที่คำนวณ Gini มือได้ → ตรวจ `_best_split` คืน (feature, threshold) ที่ถูก ·
ข้อมูลบริสุทธิ์ → ต้นไม้มีแค่ root · เทียบ `tree_.feature` และ `tree_.threshold` กับ sklearn ทีละ node

**ความเสี่ยง** กลาง แต่ **ผลกระทบสูง** เพราะอีก 4 โมเดลพึ่งโค้ดนี้ — ให้เวลากับมันมากกว่าที่ประมาณไว้

---

### 12. `random_forest` — Random Forest

**ไฟล์:** `src/ml/classification/random_forest.py` · Codex · parity **seeded**

**กลไก** bagging ของ decision tree + สุ่มฟีเจอร์ที่แต่ละ split

```text
สำหรับ tree ที่ i:
    indices = RandomState(seed_i).randint(0, n, n)        # bootstrap, มีซ้ำได้
    ที่แต่ละ node สุ่ม max_features = sqrt(p) ฟีเจอร์มาพิจารณา
ทำนาย: เฉลี่ย predict_proba ของทุกต้น (ไม่ใช่ majority vote)
```

**Grid** `max_depth` ∈ {3, 5} × `n_estimators` ∈ {50, 100} = 4 · บวกกรณี PCA components {2,4,6}
ที่ใช้ร่วมกับค่าที่ดีที่สุดแบบไม่มี PCA → รวม **7 candidates** (ตรงกับ `final_candidates: 7`)

**Preprocess** `none` สำหรับตัวหลัก และ `PCA(k)` สำหรับ 3 candidates ที่เหลือ
(`pca_uses_best_no_pca: true` — ใช้ hyperparameter ที่ชนะตอนไม่มี PCA แล้วเปลี่ยนแค่ PCA)

**Reference** `RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_leaf=5,
max_features="sqrt", bootstrap=True, random_state=42)`

**จุดที่ parity พลาด — อ่านข้อแรกให้ดี**

1. **`np.random.RandomState` เท่านั้น** sklearn สร้าง seed ของแต่ละต้นจาก
   `random_state.randint(np.iinfo(np.int32).max, size=n_estimators)` แล้วแต่ละต้นใช้ seed นั้นเรียก
   `RandomState(seed_i).randint(0, n_samples, n_samples)` เพื่อ bootstrap — พิสูจน์แล้วว่าตรงกับ
   `sklearn.ensemble._forest._generate_sample_indices` เป๊ะ **ถ้าใช้ `default_rng` จะไม่มีทางตรง**
2. sklearn เฉลี่ย **probability** ไม่ใช่ vote — ใช้ `np.mean([t.predict_proba(X) for t in trees], axis=0)`
3. `max_features="sqrt"` = `int(sqrt(p))` = `int(sqrt(8))` = 2 สำหรับชุดนี้ (ไม่ใช่ปัดขึ้น)
4. การสุ่มฟีเจอร์เกิดใหม่ **ที่ทุก node** ไม่ใช่ครั้งเดียวต่อต้น

**Unit test** `n_estimators=1, bootstrap=False, max_features=None` → ต้องเท่ากับ decision tree เดี่ยวเป๊ะ
(เป็น test ที่แยกบั๊กของ forest ออกจากบั๊กของ tree ได้) · ตรวจว่า bootstrap indices ตรงกับ
`_generate_sample_indices` ของ sklearn

**ความเสี่ยง** กลาง — ถ้า RNG ตรง ที่เหลือจะตรงตามไปเอง

---

### 13. `gradient_boosting` — Gradient Boosting

**ไฟล์:** `src/ml/classification/gradient_boosting.py` · Codex · parity **exact**

**กลไก** สร้าง regression tree ทีละต้นเพื่อ fit **negative gradient** ของ logistic loss

```text
F0 = log(p̄ / (1 - p̄))                        # log-odds ของสัดส่วนคลาสบวก
วน m = 1..M:
    p        = σ(F)
    residual = y - p                          # = -∂L/∂F ของ binary logistic loss
    fit regression tree (criterion = squared error) กับ residual
    ค่าในใบ: γ = Σ residual / Σ p(1-p)        # Newton step ไม่ใช่ค่าเฉลี่ย residual
    F ← F + learning_rate · γ
```

**Grid** `learning_rate` ∈ {0.03, 0.1} × `max_depth` ∈ {1, 2} × `n_estimators` ∈ {50, 100} = **8 candidates**

**Preprocess** `none`

**Reference** `GradientBoostingClassifier(loss="log_loss", learning_rate=lr, max_depth=d,
n_estimators=n, min_samples_leaf=5, max_features=None, random_state=42, subsample=1.0)`

**จุดที่ parity พลาด**

- **ค่าในใบใช้ Newton step** `Σg / Σh` ไม่ใช่ mean ของ residual — พลาดข้อนี้คือสาเหตุอันดับหนึ่งที่ไม่ตรง
- `F0` ต้องเป็น log-odds ของ prior ไม่ใช่ 0
- tree ภายในใช้ **squared error** กับ residual ไม่ใช่ Gini (ต่างจาก `decision_tree` card)
- `subsample=1.0` เพื่อปิด stochastic GB ไม่ให้มี RNG เข้ามาเกี่ยว
- `max_features=None` เช่นเดียวกับ decision tree

**Unit test** ข้อมูล 4 ตัวอย่าง คำนวณ `F0`, residual รอบแรก และ γ ของใบด้วยมือ → เทียบ ·
`n_estimators=1, learning_rate=1` → `F = F0 + γ` ตรวจได้ตรง ๆ

**ความเสี่ยง** สูง — ให้เวลา 3.5 ชม. และเทียบทีละ stage ด้วย `staged_decision_function` ของ sklearn
ซึ่งทำให้รู้ได้ทันทีว่าเริ่มต่างที่ต้นไหน

---

### 14. `xgboost` — Extreme Gradient Boosting

**ไฟล์:** `src/ml/classification/xgboost.py` · Codex · parity **exact** · **เริ่มแต่เช้า**

**กลไก** boosting แบบใช้ second-order Taylor ของ loss พร้อม regularization ในสูตร gain โดยตรง

```text
g_i = p_i - y_i                      # first-order gradient
h_i = p_i(1 - p_i)                   # second-order (hessian)
ค่าที่ดีที่สุดของใบ:   w* = -G / (H + λ)         โดย G = Σg, H = Σh ในใบนั้น
คะแนนของ node:        score = G² / (H + λ)
Gain ของ split = 0.5·[ G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) ] - γ
แบ่งต่อเมื่อ Gain > 0 ; F ← F + learning_rate · w*
```

ต่างจาก `gradient_boosting` ตรงที่ **regularization อยู่ในเกณฑ์เลือก split เลย** ไม่ใช่แค่ตัดกิ่งทีหลัง
และ `γ` ทำหน้าที่เป็นค่าผ่านขั้นต่ำของการแตกกิ่ง

**Grid** `tuples` 8 ชุด ตาม `tuple_fields: [n_estimators, max_depth, learning_rate, reg_lambda, gamma]`

| # | n_est | depth | lr | λ | γ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 1 | 0.1 | 1 | 0 |
| 2 | 50 | 2 | 0.1 | 1 | 0 |
| 3 | 100 | 1 | 0.03 | 1 | 0 |
| 4 | 100 | 2 | 0.03 | 1 | 0 |
| 5 | 100 | 2 | 0.1 | 1 | 0 |
| 6 | 100 | 2 | 0.1 | 10 | 0 |
| 7 | 100 | 2 | 0.1 | 1 | 0.1 |
| 8 | 100 | 2 | 0.03 | 10 | 0.1 |

**Preprocess** `none`

**Reference** `xgboost.XGBClassifier(tree_method="exact", n_estimators=…, max_depth=…,
learning_rate=…, reg_lambda=…, gamma=…, reg_alpha=0, subsample=1, colsample_bytree=1,
colsample_bylevel=1, colsample_bynode=1, min_child_weight=0, base_score=0.5, random_state=42)`

**จุดที่ parity พลาด**

- **`tree_method="exact"`** บังคับ ไม่งั้น XGBoost ใช้ histogram approximate แล้วไม่มีทางตรง
- ปิด subsample/colsample ทั้ง 4 ตัวให้เป็น 1 เพื่อตัด RNG ออกทั้งหมด
- `min_child_weight=0` ไม่งั้น default 1 จะตัดกิ่งที่เราไม่ได้ตัด
- `base_score=0.5` → `F0 = 0` (XGBoost ใช้ค่านี้เป็น probability เริ่มต้น ไม่ใช่ log-odds ของ prior
  ต่างจาก GradientBoosting — **จุดนี้ต่างกันระหว่างสองโมเดล อย่าก๊อปข้ามกัน**)
- `reg_alpha=0` เพื่อไม่ให้มี L1 บนค่าใบ

**Unit test** คำนวณ `w*` และ gain ของ split เดียวบน 4 ตัวอย่างด้วยมือ → เทียบ ·
`λ=0, γ=0, depth=1, n=1` → ต้องเท่ากับ Newton step ของ GB ที่ depth 1 ·
เทียบ `predict(output_margin=True)` ทีละ iteration ด้วย `iteration_range`

**ความเสี่ยง** สูงที่สุดในโปรเจกต์ (R1 ใน `RISKS_AND_DOD.md`)
**แผนสำรอง:** ถ้าเกินครึ่งวันแล้วยังต่าง > 1e-3 ให้ลดเหลือกลไกแกน (exact greedy + g/h + leaf shrinkage)
แล้วอธิบายขอบเขตในรายงานอย่างตรงไปตรงมา — ยังนับเป็น XGBoost from scratch ได้

---

### 15. `adaboost` ★ — AdaBoost (extracurricular, classification)

**ไฟล์:** `src/ml/classification/adaboost.py` · Codex · parity **exact**

> โจทย์บังคับให้โมเดลนี้มี **ทั้ง scratch และ ready-to-use** และอธิบายกลไกภายในอย่างละเอียด

**ทำไมโมเดลนี้น่าสนใจกับชุดข้อมูลนี้เป็นพิเศษ** AdaBoost ลด **exponential loss** `Σ exp(-y·F(x))`
ซึ่งเพิ่มน้ำหนักตัวอย่างที่ทายผิดแบบ**เอ็กซ์โพเนนเชียล** จึงไวต่อ outlier มากที่สุดในบรรดา ensemble ทั้งหมด
ขณะที่ target ของเรามี **kurtosis = 16.26** (fat tail สุดขั้วจาก spike) ทฤษฎีจึงทำนายล่วงหน้าได้เลยว่า
**AdaBoost คือโมเดลที่จะเสียหายหนักที่สุดใน variant A และฟื้นตัวชัดที่สุดใน variant B**
→ เป็นสมมติฐานที่ทดสอบได้จริงด้วย paired comparison ที่เราทำอยู่แล้ว และเป็นเนื้อหา discussion ชั้นดี
ให้บันทึกผลข้อนี้ลงรายงานไม่ว่าจะออกหัวหรือก้อย

**กลไกโดยละเอียด (SAMME, discrete)**

```text
w_i = 1/n  สำหรับทุกตัวอย่าง
วน m = 1..M:
    1) fit decision stump (depth 1) ด้วย sample_weight = w
    2) err_m = Σ w_i · 1[y_i ≠ h_m(x_i)] / Σ w_i
    3) α_m   = learning_rate · log((1 - err_m) / err_m)        # SAMME สองคลาส
    4) w_i  ← w_i · exp(α_m · 1[y_i ≠ h_m(x_i)])               # ทายผิดโดนเพิ่มน้ำหนัก
    5) normalize w ให้ผลรวมเป็น 1
ทำนาย: sign( Σ α_m · h_m(x) )   โดย h ∈ {-1, +1}
```

**สัญชาตญาณ** แต่ละรอบ stump ตัวใหม่ถูกบังคับให้สนใจตัวอย่างที่ตัวก่อนหน้าทำพลาด เพราะน้ำหนักของมันพองขึ้น
`α_m` คือ "ความน่าเชื่อถือ" ของ stump ตัวนั้น — ยิ่ง error ต่ำยิ่งได้เสียงโหวตมาก และถ้า `err_m = 0.5`
(เดาสุ่ม) จะได้ `α_m = 0` คือไม่มีเสียงเลย

**เงื่อนไขขอบที่ config กำหนดไว้ชัด** (`adaboost` section): `error ≤ epsilon` (1e-12) → `completed_perfect` ·
`error ≥ 0.5` ตั้งแต่ตัวแรก → `failed` · `error ≥ 0.5` ตัวหลัง ๆ → `completed_early` ·
คำนวณน้ำหนักใน log space ด้วย `logsumexp` · `alpha_factor = 0.5` · zero-vote → คลาส 1

**Grid** `learning_rate` ∈ {0.5, 1} × `n_estimators` ∈ {25, 50, 100} = **6 candidates**

**Preprocess** `none`

**Reference** `AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion="gini"),
n_estimators=n, learning_rate=lr, algorithm="SAMME", random_state=42)`

> ตรวจเวอร์ชันที่ติดตั้ง: sklearn 1.9 ถอด `algorithm="SAMME.R"` ออกแล้วและ `SAMME` เป็น default
> ถ้า constructor ไม่รับพารามิเตอร์นี้ ให้ตัดออกแล้วบันทึกไว้ใน manifest ตามที่
> `references.adaboost.api_difference` สั่ง

**จุดที่ parity พลาด**

- **stump ต้องรองรับ `sample_weight` จริง** — weighted Gini ไม่ใช่ Gini ธรรมดา
  (`Σ w` แทน `count` ในทุกสูตร) นี่คือจุดที่พังบ่อยที่สุดและเป็นเหตุผลที่ต้องเขียน `decision_tree` ก่อน
- α ใช้ natural log ไม่ใช่ log2
- sklearn คูณ `learning_rate` เข้าไปใน α แล้ว อย่าคูณซ้ำตอนทำนาย
- normalize น้ำหนักทุกรอบ ไม่ใช่รอบสุดท้าย
- label ต้องแปลงเป็น {-1, +1} ตอนคำนวณ margin แล้วแปลงกลับเป็น {0, 1} ตอนคืนค่า (`labels: [-1, 1]` ใน config)

**Unit test กลไกภายใน**

- stump ที่ error = 0 → `α = inf` ต้องถูกจัดการตาม `completed_perfect` ไม่ใช่ปล่อยให้เป็น `inf`
- 2 รอบบนข้อมูล 4 ตัวอย่าง คำนวณ `err`, `α`, `w` ใหม่ด้วยมือ → เทียบทุกค่า
- น้ำหนักรวมหลัง normalize = 1 เสมอ
- `n_estimators=1` → ต้องเท่ากับ stump เดี่ยว
- เทียบ `estimator_weights_` และ `estimator_errors_` กับ sklearn ทีละตัว ไม่ใช่เทียบแค่ `predict`

**ความเสี่ยง** กลาง — ขึ้นกับว่า weighted stump ถูกหรือไม่ ซึ่งเป็นของที่ใช้ร่วมกับ `decision_tree`

---

### 16. `stacking` — Stacking Classifier

**ไฟล์:** `src/ml/classification/stacking.py` · Codex · parity **iterative** · **ต้องทำหลัง logistic/DT/kNN เสร็จ**

**กลไก** เทรน base learner 3 ตัว เอา out-of-sample probability มาเป็นฟีเจอร์ให้ meta learner

```text
bases = [logistic, decision_tree, knn]        meta = logistic
1) แบ่ง train ตามเวลาเป็น 4 บล็อกด้วย np.array_split (expanding window)
   เริ่มจาก 40% แรกเป็น training ก้อนแรก (initial_original_fraction)
2) แต่ละ fold: fit bases บนอดีต → predict_proba บนบล็อกถัดไป (out-of-sample)
   เว้น purge 5 วันทำการตาม original timeline ก่อนบล็อกถัดไป (purge_original_trading_days)
3) meta fit บน OOS probability เท่านั้น (meta_fit: train_oos_only)
4) refit bases บน train เต็มของ variant นั้น (refit_bases: full_variant_train)
```

**Grid** ว่าง → 1 candidate

**Preprocess** `base_specific` — แต่ละ base ใช้ preprocessing ของตัวเองตาม card ของมัน

**Reference** `StackingClassifier(estimators=[...], final_estimator=LogisticRegression(),
cv=TimeSeriesSplit(4), passthrough=False)` เป็น behavioral reference เท่านั้น

**จุดที่ parity พลาด**

- **purge ต้องนับตาม original trading day ไม่ใช่ row gap** เพราะ variant `non_spike` มีแถวหายไป
  การนับ 5 แถวจะข้ามเวลาจริงมากกว่า 5 วัน → เกิด leakage (ความเสี่ยงที่ `MODEL_TRAINING_PLAN.md` เตือนไว้)
- meta ห้ามเห็น in-sample probability เด็ดขาด
- join ระหว่าง base กับ meta ต้องเป็น exact one-to-one ตาม Date (`position_join`)
- `StackingClassifier` ของ sklearn ไม่มี purge → ตัวเลขจะต่าง และนั่นถูกต้องแล้ว ให้บันทึกเหตุผล

**Unit test** ตรวจว่าไม่มี index ของ validation โผล่ใน training ของ fold ใด ๆ ·
ตรวจว่า purge ตัดแถวในช่วง 5 วันทำการจริง · meta features มี 3 คอลัมน์ตามชื่อใน config

**ความเสี่ยง** สูง — ความซับซ้อนอยู่ที่ fold/purge ไม่ใช่ที่อัลกอริทึม

---

### 17. `svm` — Support Vector Machine

**ไฟล์:** `src/ml/classification/svm.py` · Codex · parity **iterative**

**กลไก** linear soft-margin แก้ด้วย subgradient descent (ตาม `scope` ใน config ไม่ใช่ SMO)

```text
J(w, b) = 0.5·||w||² + C · Σ max(0, 1 - y_i(w·x_i + b))        y ∈ {-1, +1}
subgradient:
    ถ้า y_i(w·x_i + b) < 1 :  ∇w += -C·y_i·x_i ,  ∇b += -C·y_i
    เสมอ                   :  ∇w += w
decision_function = w·x + b          (ไม่ใช่ probability)
```

**Grid** `C` ∈ {0.1, 1, 10} = 3 · × (ไม่มี PCA + PCA 3 ขนาด) = **12 candidates** (`final_candidates: 12`)

**Preprocess** `standardize` + PCA components {2, 4, 6}

**Reference** `LinearSVC(C=C, loss="hinge", fit_intercept=True, intercept_scaling=1000, tol=1e-12, max_iter=2000000, random_state=42)`

**เกณฑ์ parity ของโมเดลนี้คือ objective ไม่ใช่น้ำหนัก — มีหลักฐาน**

hinge loss **ไม่เรียบ** (ไม่ differentiable ที่ margin = 1) ทำให้ solver สองตัวหยุดคนละจุดใกล้ optimum ได้
โดยที่ไม่มีใครผิด วัดจริงบนปัญหาสังเคราะห์ `fit_intercept=False`:

```text
C=0.1   obj_scratch=5.615042   obj_sklearn=5.615042    rel_gap=1.2e-08   max|dw|=1.4e-06   label agree=1.0000
C=1.0   obj_scratch=34.825849  obj_sklearn=34.985936   rel_gap=4.6e-03   max|dw|=4.7e-02   label agree=0.9900
```

ที่ C=1.0 **scratch ได้ objective ต่ำกว่า sklearn** (34.8258 < 34.9859) แปลว่า subgradient descent
ของเราเข้าใกล้ optimum ได้ดีกว่า liblinear ในรอบนั้น — ถ้าตั้งเกณฑ์เป็น "น้ำหนักต้องเท่ากัน"
เราจะเสียเวลาไล่แก้สิ่งที่ไม่ได้พัง ดังนั้นเกณฑ์ของโมเดลนี้คือ:

1. `rel_gap = |J_scratch - J_sklearn| / J_sklearn < 1e-3` **และ** `J_scratch <= J_sklearn * (1 + 1e-3)`
2. label agreement ≥ 0.99
3. ผ่าน KKT check (ดู unit test)

**จุดที่ parity พลาด**

- `loss="hinge"` ไม่ใช่ `"squared_hinge"` ซึ่งเป็น **default ของ sklearn**
- liblinear **penalize intercept** ส่วนสูตรเราไม่ → ตั้ง `intercept_scaling=1000` เพื่อลดผลกระทบให้เล็กลง
  และควรทำ unit test หลักด้วย `fit_intercept=False` ซึ่งตัดปัญหานี้ทิ้งทั้งหมด
- `C` คูณกับพจน์ hinge ไม่ใช่พจน์ norm (ตรงกับ config `half_l2_plus_C_sum_hinge`)
- step size ของ subgradient ต้องเป็น `1/t` (เหมาะกับ strongly convex) ไม่ใช่ค่าคงที่
  ไม่งั้นจะแกว่งรอบ optimum ไม่ลู่เข้า

**Unit test** ข้อมูลแยกได้ margin กว้าง → support vector ต้องเป็นจุดที่ margin = 1 ·
**KKT check:** จุดที่ `y(w·x) > 1` ต้องไม่มีส่วนร่วมใน subgradient เลย · objective ลดลง monotone ·
`fit_intercept=False, C=0.1` → น้ำหนักต้องตรง sklearn ที่ 1e-5 (กรณีนี้ตรงจริงตามตารางข้างบน)

**ความเสี่ยง** สูง (R3) **แผนสำรอง:** คง linear SVM ไว้แล้วอธิบาย kernel trick ในรายงานโดยไม่ implement

---

## 4. Clustering

### 18. `kmeans` — k-Means

**ไฟล์:** `src/ml/clustering/kmeans.py` · Claude · parity **seeded**

**กลไก** Lloyd's algorithm + k-means++ init

```text
init k-means++:
    เลือก centroid แรกแบบสุ่มสม่ำเสมอ
    centroid ถัดไป: สุ่มโดยมีความน่าจะเป็น ∝ D(x)²  (D = ระยะถึง centroid ที่ใกล้ที่สุดที่มีอยู่)
วนจน converge:
    assign : แต่ละจุดไป centroid ที่ใกล้ที่สุด
    update : centroid = ค่าเฉลี่ยของสมาชิก
หยุดเมื่อ shift < tol (1e-6) หรือครบ max_iter (300)
เลือกผลที่ inertia ต่ำสุดจาก n_init = 10 ครั้ง
```

**Grid** `k` ∈ {2, 3, 4, 5, 6} = 5 candidates

**Preprocess** `standardize`

**Reference** `KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=1e-6, random_state=42)`

**จุดที่ parity พลาด**

- `RandomState` เท่านั้น และ k-means++ ของ sklearn สุ่ม **n_local_trials = 2 + log(k)** candidate
  ต่อรอบแล้วเลือกตัวที่ลด inertia ได้มากสุด — ไม่ใช่สุ่มมาตัวเดียว ข้อนี้พลาดแล้ว init ต่างทันที
- `tol` ของ sklearn เทียบกับ **Frobenius norm ของการขยับ centroid หารด้วย variance เฉลี่ย** ไม่ใช่ระยะดิบ
- cluster label ไม่มีความหมายเชิงลำดับ — ใช้ `align_clusters()` ใน `compare.py` ก่อนเทียบเสมอ
- การประเมินใช้ `train_silhouette` เป็นเกณฑ์หลัก (`selection.clustering.primary`)
  และ map cluster → label ด้วย majority บน **train เท่านั้น** (tie → คลาส 1)

**Unit test** 3 กลุ่มที่แยกกันชัดเจน → ต้องได้ 3 cluster ที่ถูกต้องไม่ว่า init อย่างไร ·
inertia ลดลง monotone ทุก iteration · k=1 → centroid = ค่าเฉลี่ยทั้งชุด

**ความเสี่ยง** กลาง — ถ้า k-means++ ไม่ตรง ให้ fix init ด้วย centroid ที่กำหนดเองทั้งสองฝั่งแล้วเทียบ
เฉพาะส่วน Lloyd ซึ่งพิสูจน์ความถูกต้องของอัลกอริทึมได้เหมือนกัน

---

### 19. `agglomerative` — Agglomerative Clustering

**ไฟล์:** `src/ml/clustering/agglomerative.py` · Claude · parity **exact**

**กลไก** bottom-up merge ทีละคู่จนเหลือ k กลุ่ม

```text
เริ่ม: ทุกจุดเป็น cluster ของตัวเอง
วนจนเหลือ k cluster:
    หาคู่ (A, B) ที่ระยะน้อยที่สุดแล้ว merge
    ward    : Δ = (|A|·|B| / (|A|+|B|)) · ||centroid_A - centroid_B||²     # เพิ่มของ within-variance
    average : ระยะเฉลี่ยของทุกคู่ข้ามกลุ่ม
```

**Grid** `k` ∈ {2,3,4,5,6} × `linkage` ∈ {ward, average} = **10 candidates**

**Preprocess** `standardize` · metric euclidean

**Reference** `AgglomerativeClustering(n_clusters=k, linkage=l, metric="euclidean")`

**จุดที่ parity พลาด**

- **ไม่มี `predict` สำหรับข้อมูลใหม่โดยธรรมชาติ** (`standard_predict: false`) — config กำหนดทางออกไว้แล้วคือ
  `oos_extension: frozen_nearest_centroid` คือคำนวณ centroid ของแต่ละ cluster จาก train แล้วตอน
  validation ให้จุดใหม่ไปอยู่ centroid ที่ใกล้ที่สุด **ต้องประกาศข้อจำกัดนี้ในรายงานให้ชัด** (R9)
- ward ต้องใช้สูตร Lance-Williams update ไม่ใช่คำนวณ centroid ใหม่ทุกครั้ง ถึงจะตรงกับ scipy/sklearn
- tie เมื่อระยะเท่ากัน → merge คู่ที่ index น้อยกว่าก่อน
- `average` linkage ของ sklearn ใช้ระยะเฉลี่ย ไม่ใช่ระยะระหว่าง centroid (นั่นคือ `centroid` linkage คนละตัว)

**Unit test** 4 จุดบนเส้นตรงที่รู้ลำดับการ merge → เทียบ dendrogram ทีละขั้น ·
k = n → ทุกจุดเป็น cluster เดี่ยว · ward บนสองกลุ่มที่สมมาตร → merge คู่ที่ใกล้ที่สุดก่อน

**ความเสี่ยง** กลาง — ความยากอยู่ที่ Lance-Williams และ OOS extension ไม่ใช่ตัว merge

---

## 5. Checklist ก่อนบอกว่าโมเดลเสร็จ

คัดจาก `AGENTS.md` §5 — ติ๊กครบ 6 ข้อเท่านั้น

- [ ] ไฟล์ `src/ml/<task>/<name>.py` ทำตาม `BaseModel` ครบทุก method และ hyperparameter มี default หมด
- [ ] `tests/ml/<task>/test_<name>.py` มีครบ 3 แบบ: unit test กลไกภายใน · strict parity · save/load round-trip
- [ ] รันผ่าน trainer ได้ทั้ง `with_spike` และ `non_spike`
- [ ] artifact ครบใน `outputs/modeling/<variant>/<task>/<name>/` และ `load_verification.json` เป็น `"passed": true`
- [ ] มีแถวใน leaderboard
- [ ] กลับมาแก้ card ของตัวเองในไฟล์นี้ด้วย **ตัวเลขจริง** และบันทึกความต่างจาก reference ถ้ามี
