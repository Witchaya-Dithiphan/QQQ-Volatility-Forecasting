# QQQ 5-Day Annualized Volatility Forecasting — Model Training Plan

> **สถานะ:** เอกสารแผนหลัก (single source of truth) สำหรับ Modeling Phase  
> **วันที่จัดทำ:** 2026-10-08  
> **Deadline ตาม PDF:** 2026-10-12 — Final presentation และส่งเอกสารทั้งหมด  
> **ขอบเขตรอบนี้:** ตรวจ requirement/repository และ freeze implementation configuration; ยังไม่มี model implementation, training หรือ performance result

เอกสารนี้เป็นแหล่งอ้างอิงหลักสำหรับ model inventory, scratch/reference contract, training protocol, evaluation, artifacts และ milestones เอกสารอื่นควรสรุปสถานะและ link มาที่ไฟล์นี้แทนการคัดลอกรายละเอียดทั้งหมด

## 1. ข้อสรุปจาก Repository Audit

### 1.1 สถานะที่ตรวจจาก checkout นี้

| รายการ | สถานะ | หลักฐาน |
| --- | --- | --- |
| Branch/commit | ตรวจใน audit นี้ | `main` ที่ `03dfada` |
| Course PDF | มีใน checkout แต่ยัง untracked | `_69-1-01076641_Project-description.pdf` จำนวน 5 หน้า |
| Phase 1 data pipeline | Implemented; มีหลักฐานจาก source, tests และ artifacts เดิม | `src/run_data_pipeline.py`, Phase 1 modules, reports และ labeled splits |
| Phase 2 spike pipeline | Implemented; มีหลักฐานจาก source, tests และ artifacts เดิม | M1–M8 modules/tests, experiment datasets, reports และ figures |
| With-Spike Train | ตรวจไฟล์จริงใน audit นี้ | `data/processed/experiments/with_spikes/train.csv`, 1,733 rows; byte-identical กับ Original Train ตาม SHA-256 ที่บันทึก |
| Non-Spike Train | ตรวจไฟล์จริงใน audit นี้ | `data/processed/experiments/non_spike/train.csv`, 1,520 rows |
| Full Validation/Test | ตรวจไฟล์จริงใน audit นี้ | Original `validation_labeled.csv` 371 rows และ `test_labeled.csv` 373 rows |
| Model source | Missing / Planned | `src/models/regression.py` และ `src/models/classification.py` มีเพียง module docstring |
| Modeling notebooks | Missing / Planned | `notebooks/03_regression.ipynb` และ `04_classification.ipynb` มี 0 cells |
| Model dependencies | Missing / M2 dependency gate | `requirements.txt` ยังไม่มี `scikit-learn` หรือ `xgboost`; policyเลือก Python 3.14.8แล้ว (re-frozen 2026-10-10) แต่ exact package versionsต้อง lockจาก environmentที่ทดสอบผ่านจริง |
| Models/metrics/predictions | Missing | ยังไม่มีหลักฐาน model artifacts หรือ quantitative model results |
| Python environment | Frozen for modeling | ใช้ `.venv` Python 3.14.8 (re-frozen 2026-10-10 จาก 3.12.3); หลักฐานตรวจรับเดิม Python 3.11.9 เก็บเป็น historical evidenceและต้องไม่เขียนเหมือนเป็น modeling environmentปัจจุบัน |

### 1.2 การแบ่งสถานะหลักฐาน

- **Implemented:** พบ source implementation จริง
- **Verified in this audit:** ตรวจไฟล์/metadata/checksum/count ใน checkout รอบนี้
- **Previously verified according to recorded evidence:** อ้างผลคำสั่งเดิมจากเอกสารอย่างชัดเจน ไม่เขียนเหมือนเพิ่งรัน
- **Planned:** เป็นแบบออกแบบหรือขั้นตอนอนาคต ยังไม่มี executable implementation
- **Missing / Blocked:** ยังไม่มี implementation, dependency, artifact หรือยังต้องตัดสินใจ

ผล `334 passed, 2 skipped`, spike-specific `142 passed, 1 skipped`, Ruff/mypy และ M8 gate เป็น **previously verified according to recorded evidence dated 2026-10-07**; audit เอกสารรอบนี้ไม่ได้รัน full test suite ใหม่

## 2. Requirement Matrix

เลขหน้าอ้างจาก PDF 5 หน้าใน checkout นี้

| ID | แหล่ง | หน้า/หัวข้อ | Requirement/ข้อสังเกต | ผลต่อแผน | สถานะ |
| --- | --- | --- | --- | --- | --- |
| PDF-01 | Explicit PDF requirement | หน้า 1, `Document submitted` | ส่ง report, slides/presentation, notebook และ/หรือ Python scripts รวม related documents | ต้องมี submission package ไม่ใช่เฉพาะโค้ด | Mandatory |
| PDF-02 | Explicit PDF requirement | หน้า 1, `Date` | Final presentation และส่งเอกสารทั้งหมดวันที่ **12 October 2026** | วาง critical path ตาม deadline จริง | Mandatory |
| PDF-03 | Explicit PDF requirement | หน้า 1, task description | อธิบาย problem setting, current state, storytelling และ feature insight | Report/slides ต้องมีเหตุผลของ features และ why ML | Mandatory |
| PDF-04 | Explicit PDF requirement | หน้า 1, remark | พิจารณาใช้ feature อย่างน้อย 5 ตัว | Canonical feature set มี 8 ตัว จึงครอบคลุมเชิงจำนวน | Mandatory unless instructor exception |
| PDF-05 | Explicit PDF requirement | หน้า 2, model implementation | Build model, training/testing iterations, batch runs และ save/load model; ใช้ library ได้แต่ต้องมี from-scratch functions | Framework ต้องรองรับ run, persist, reload และ reload test | Mandatory |
| PDF-06 | Explicit PDF requirement | หน้า 2, Regression list | Linear, Multiple, Polynomial และ better/extracurricular model | Inventory ใช้ Simple Linear, Multiple Linear, Polynomial และ Elastic Net | Mandatory scope interpretation |
| PDF-07 | Explicit PDF requirement | หน้า 2, Classification list | Logistic, Decision Tree, Random Forest, Stacking, GB, XGBoost, Naive Bayes (if any), SVM, dimensionality reduction, k-NN, k-Means, Agglomerative, Perceptron, SLP, MLP และ better model | ตาม course interpretation ที่ผู้ใช้ยืนยันเมื่อ 2026-10-08 รายการหลายชื่อใน bullet เดียวต้องทำทุกตัว จึงแยก predictive classification 13 รายการและ clustering 2 รายการ; dimensionality reduction เป็น required configuration | Mandatory scope interpretation; counting ambiguity resolved |
| PDF-08 | Explicit PDF requirement | หน้า 3, new models | อย่างน้อย 1 new regression model และ 1 new classification modelที่ไม่ใช่ตัวเดียวกัน พร้อมอธิบายกลไก, scratch และ ready-to-use | เลือก Elastic Net และ AdaBoost ตาม user instruction; ยังควรยืนยันว่าอาจารย์ยอมรับว่า extracurricular | Mandatory; approval ambiguity |
| PDF-09 | Explicit PDF requirement | หน้า 3, results | Quantitative results, benchmarking, analysis, discussion, Loss, Confusion Matrix, Accuracy, Precision, Sensitivity, Specificity/TNR, F1, ROC/AUC, performance curve และ R² | กำหนด task-specific metrics และ plots; ไม่ใช้ metric ที่ไม่เหมาะกับ task | Mandatory |
| PDF-10 | Explicit PDF requirement | หน้า 5, scoring | Regression และ Classification ทุกโมเดล built from scratch; ready-to-use Scikit-learn/PyTorch ใช้ประกอบเพื่อยืนยันได้ | ห้าม wrapper `.fit()` หรือเขียน metrics เองแล้วเรียก model ว่า scratch | Mandatory; 40 points |
| PDF-11 | Explicit PDF requirement | หน้า 5, scoring | Evaluation มี Accuracy, loss, F1, performance curve ฯลฯ | ต้องมี persisted metrics/figures ที่ trace กลับ config และ predictions ได้ | Mandatory; 30 points |
| PDF-12 | Explicit PDF / administrative | หน้า 4, group/dataset channel | ช่องทางลงข้อมูลสมาชิกและ dataset อยู่ใน Google Classroom/Google Sheet; หน้า 3–4 ยังเตือนห้ามใช้ dataset ซ้ำกับกลุ่มที่เสนอแล้ว | ไม่เปลี่ยน modeling implementation; ทีมต้องยืนยันการลงทะเบียน QQQ นอก repository | Administrative; verify with instructor/classroom |
| USER-01 | User instruction | รอบนี้ | เพิ่ม Elastic Net และ AdaBoost และยืนยันว่า model หลายชื่อใน requirement bullet เดียวต้องทำทุกตัว | อยู่ใน 19-item inventory; Perceptron และ SLP เป็นคนละ implementation/deliverable | Mandatory for project scope |
| USER-02 | User instruction | รอบนี้ | ส่ง With-Spike ให้ครบก่อน แล้ว reuse กับ Non-Spike | Milestones ใช้ With-Spike-first; อนุญาต M3 Non-Spike Train/Validation-only pilotหนึ่งโมเดลเป็น explicit plumbing exception แล้วหยุดจน M9; ห้ามอ้าง paired comparisonจน Non-Spikeครบ | Priority decision |
| USER-03 | User instruction | รอบนี้ | รักษา Phase 1/2 data contract; ห้าม recompute features/targets/Q75 หลังกรอง | Modeling loader ต้องเลือก explicit features และใช้ accepted CSVs แบบ read-only | Mandatory guardrail |
| USER-04 | User instruction | รอบนี้ | สร้าง requirement matrix, reusable framework, scratch/reference table, milestones, artifact/resume design | อยู่ในเอกสารนี้ | Mandatory documentation |
| DEC-01 | Frozen implementation decision | Modeling | Dataset variant เป็น config `with_spike`/`non_spike`; model codeไม่ hard-code variant | ทำให้ reuse ได้และลด divergence | Frozen 2026-10-08 |
| DEC-02 | Frozen implementation decision | Modeling | ใช้ NumPy/Pandas สำหรับ scratch numerical operations และ Scikit-learn/XGBoost เป็น reference เมื่อเหมาะสม | Core learning logicต้องเขียนเอง; exact dependency lockสร้างหลัง compatibility smoke test | Frozen policy; versions gated in M2 |
| DEC-03 | Frozen implementation decision | Evaluation | Train สำหรับ fit, Validation สำหรับ tuning/threshold, Test หลัง freeze เท่านั้น | ลด leakage และทำ paired protocol ได้ | Frozen 2026-10-08 |
| DEC-04 | Frozen implementation decision | PDF หน้า 2 | `Perceptron & Single-layer perceptron (SLP)` ต้องทำทุกชื่อใน requirement bullet | แยก Perceptron และ SLP เป็นคนละ scratch/reference deliverable | Frozen 2026-10-08 |
| AMB-02 | Ambiguity | PDF หน้า 2 | Naive Bayes มีคำว่า `if any` | รวมไว้ก่อนตาม user instruction; ไม่ตัดเป็น optional | Scope assumption |
| DEC-05 | Frozen implementation decision | PDF หน้า 2 | Dimensionality reduction สำหรับ RF/SVM | ทำ scratch PCA configurationกับทั้ง Random Forest และ SVM พร้อม no-PCA baseline; ไม่เพิ่ม model count | Frozen 2026-10-08 |
| AMB-04 | Ambiguity | PDF หน้า 2 | Regression bullet เขียน `better and extracurricular classification model` | ใช้หน้า 3 และหน้า 5 เป็นหลักว่า ต้องมี better regression หนึ่งตัวและ better classification หนึ่งตัว | Document inconsistency |
| DEC-06 | Frozen implementation decision | Algorithm fidelity | `XGBoost` ต้องมีองค์ประกอบเฉพาะ ไม่ใช่เรียก generic gradient boosting ว่า XGBoost | ใช้ faithful minimal regularized second-order boosting scopeตาม Section 14 | Frozen; high-risk implementation |
| DEC-07 | Frozen implementation decision | Clustering | PDF จัด k-Means/Agglomerative ใต้ classification แต่เป็น unsupervised | รายงานแยกเป็น clustering taskและใช้ internal metrics/post-hoc Train mappingตาม Section 14 | Frozen 2026-10-08 |

## 3. Model Inventory และสมมติฐานการนับ

### 3.1 รายการหลักที่ยืนยันแล้ว: 19 รายการ

**Regression — 4**

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Elastic Net — extracurricular candidate

**Classification — 13**

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Stacking Classifier
5. Gradient Boosting
6. XGBoost
7. Naive Bayes
8. Support Vector Machine
9. k-Nearest Neighbors
10. Perceptron
11. Single-layer Perceptron (SLP)
12. Multi-layer Perceptron
13. AdaBoost — extracurricular candidate

**Clustering — 2**

1. k-Means
2. Agglomerative Clustering

### 3.2 สิ่งที่เลข 19 ไม่ได้หมายถึง

- PDF ไม่ได้พิมพ์ยอดรวมว่า “ต้องมี 19 โมเดล” โดยตรง; 19 เป็นผลรวมหลังแยกทุก model name ในแต่ละ requirement bullet ตาม course interpretation ที่ผู้ใช้ยืนยัน
- 19 ไม่รวม reference implementation เป็นโมเดลหลักใหม่
- 19 ไม่รวม With-Spike/Non-Spike เป็นคนละ algorithm
- 19 ไม่รวม PCA/no-PCA, raw/clipped predictions, threshold choices หรือ hyperparameter trials เป็นโมเดลใหม่
- 19 algorithms × 2 dataset variants ให้ **38 finalized scratch deliverables** เป็น planning lower bound และมี reference deliverablesได้อีก 38; ตัวเลข 38/76 ไม่ใช้เป็น operational completeness check เพราะ candidate fits, PCA variants, stacking folds, clustering runs, retries และ save/load verificationมีจำนวนต่างกัน ระบบต้องสร้าง expected-run manifestจาก executable config
- Stacking มี base learners และ meta-learnerหลายตัว แต่ใน inventory นับเป็นหนึ่ง ensemble item

## 4. Data Contract ที่ Modeling ต้องรักษา

### 4.1 Inputs หลัก

- With-Spike Train: `data/processed/experiments/with_spikes/train.csv` — 1,733 rows
- Non-Spike Train: `data/processed/experiments/non_spike/train.csv` — 1,520 rows
- Validation: `data/processed/validation_labeled.csv` — 371 rows
- Test: `data/processed/test_labeled.csv` — 373 rows
- Diagnostic flags: `data/processed/experiments/diagnostics/validation_flagged.csv` และ `test_flagged.csv`; ใช้ segmentation เท่านั้น

### 4.2 Explicit model columns

Canonical `FEATURE_COLUMNS`:

1. `return_1d`
2. `return_5d`
3. `historical_volatility_5d`
4. `historical_volatility_20d`
5. `intraday_range`
6. `sma_ratio_5_20`
7. `rsi_14`
8. `volume_zscore_20`

Targets:

- Regression: `target_volatility_5d`
- Classification: `target_high_volatility`

`Date`, OHLCV raw columns, targets, spike flags, diagnostic segment และ auxiliary numeric columnsต้องไม่เข้า feature matrix โดยอัตโนมัติ

### 4.3 Guardrails

- ห้ามเปลี่ยน Raw/Snapshot/Manifest หรือ Phase 1/2 accepted artifacts
- ห้าม recompute returns, features, targets หรือ Q75 หลังกรอง Non-Spike
- ใช้ Original Train-only Q75 และ labels เดิมทั้งสอง variants
- ใช้ chronological splits และ purge gaps เดิม
- Full Validation/Test เป็นผลหลัก; flagged segments เป็น diagnostics
- Non-Spike หมายถึง operationally non-spike-affected ตาม `[s-5,s+19]`; ไม่อ้างว่าปราศจาก spike influence ทุก feature เพราะ Wilder RSI มี recursive decay

## 5. Reusable Modeling Architecture — Planned

ออกแบบให้เล็กพอทำทันและแยก responsibility ชัดเจน

```text
src/modeling/
  contracts.py          # schema, feature/target constants, variant paths
  datasets.py           # read-only loader/validator
  preprocessing.py      # standardization, polynomial terms, PCA
  registry.py           # task/model -> constructors + capabilities
  metrics.py            # regression/classification/clustering metrics
  selection.py          # chronological validation/search protocol
  persistence.py        # save/load + compatibility manifest
  artifacts.py          # strict JSON, predictions, figures, checksums
  runner.py             # CLI orchestration/resume
src/models/
  regression.py         # scratch regression algorithms
  classification.py     # scratch classifiers
  clustering.py         # scratch clustering algorithms (planned new file)
tests/modeling/
notebooks/03_regression.ipynb
notebooks/04_classification.ipynb
notebooks/05_clustering.ipynb   # planned presentation notebook for mandatory clustering work
```

Common API ใช้เท่าที่มีความหมาย:

- `fit(X, y)` หรือ `fit(X)` สำหรับ clustering
- `predict(X)`
- `decision_function(X)` เมื่อมี score แต่ไม่มี probability
- `predict_proba(X)` เฉพาะ algorithm ที่นิยาม probability จริง
- `get_training_history()` เฉพาะ iterative algorithm ที่มี history จริง
- `save(path)` / `load(path)` หรือ persistence adapter พร้อม reload verification

Notebook ต้องเป็น presentation layer ที่เรียก source APIs ไม่คัดลอก training logicจำนวนมาก

## 6. Scratch / Reference Contract ต่อโมเดล

NumPy ใช้สำหรับ matrix operations, random generation และ numerical primitives ได้ แต่ core fitting logic, update rules, splitting criteria, tree growth, ensemble aggregation และ stopping logicต้องอยู่ใน project code

| Model | Scratch objective/core logic | Preprocessing/HP หลัก | Reference | Verification และความเสี่ยง |
| --- | --- | --- | --- | --- |
| Simple Linear Regression | OLS สำหรับ feature เดียว; closed-form หรือ gradient descent | เลือก featureด้วย Validation เท่านั้น; intercept; scalingถ้าใช้ GD | `sklearn.linear_model.LinearRegression` | synthetic line; compare predictions/tolerance; risk ต่ำ |
| Multiple Linear Regression | OLS หลาย features; normal equation/pseudoinverse หรือ GD | scaling, intercept, solver/tolerance | `LinearRegression` | synthetic full-rank/collinear fixtures; ไม่บังคับ coefficient exact เมื่อ solutionไม่ unique |
| Polynomial Regression | Standardize candidate featureหนึ่งตัว, สร้าง basis `[x, x²]` หรือ `[x, x², x³]` แล้ว OLSด้วย pseudoinverse | ทดลอง 8 features × degree `[2,3]`; ไม่มี interactions; เลือกด้วย Validation RMSE | `PolynomialFeatures(include_bias=False)` + `LinearRegression` | ตรวจ basis/term orderและ predictions; fixed univariate scopeลด term explosion |
| Elastic Net | MSE + L1/L2; coordinate descentพร้อม convergence | standardization; `alpha=[1e-4,1e-3,1e-2,1e-1]`, `l1_ratio=[0.25,0.5,0.75]`, `max_iter=5000`, `tol=1e-6` | `sklearn.linear_model.ElasticNet` | objective decreases/KKT-style checks; compare predictions/objectiveภายใต้ scalingเดียวกัน |
| Logistic Regression | Binary cross-entropy + gradient update | standardization, learning rate, regularization, class weight, tolerance | canonical reference=`sklearn.linear_model.SGDClassifier(loss="log_loss")`; `LogisticRegression` เป็น secondary behavioral benchmarkเท่านั้น | gradient sanity, probability range, separable/non-separable fixtures |
| Decision Tree | Recursive splitเลือก impurity reduction; stopping/pruning policy | max depth, min samples, criterion, feature tie-break | `DecisionTreeClassifier` | hand-checkable splits; deterministic tie policy; risk medium |
| Random Forest | Bootstrap trees + random feature subsets + vote/probability aggregation | trees, depth, max features, bootstrap, seed | `RandomForestClassifier` | bootstrap/feature-subset tests; deterministic seed; runtime risk |
| Stacking Classifier | Train Logistic Regression, Decision Tree และ k-NN base learnersบน 4 expanding chronological OOS folds; fit Logistic Regression meta-learnerบน OOS scores | purge 5 trading daysบน original timeline; no shuffle; fixed base/meta set | reference stackใช้ canonical `SGDClassifier(loss="log_loss")`, `DecisionTreeClassifier`, `KNeighborsClassifier`; `StackingClassifier` เป็น conceptual referenceเท่านั้น | leakage riskสูง; assertทุก meta-featureมาจากอดีต-only fold |
| Gradient Boosting | Sequential additive trees fit negative gradients/residuals | estimators, depth, learning rate, loss | `GradientBoostingClassifier` | loss/history decreasesบน fixtures; distinguishจาก AdaBoost/XGBoost |
| XGBoost | Binary-logistic regularized second-order boosting; gradients/Hessians, exact split gain, regularized leaf weights, shrinkage | curated budget≤12 จาก estimators `[50,100]`, depth `[1,2]`, learning rate `[0.03,0.1]`, lambda `[1,10]`, gamma `[0,0.1]` | `xgboost.XGBClassifier` หลัง dependency gateผ่าน | ความเสี่ยงสูงมาก; ไม่ implement GPU/distributed/quantile sketch แต่ห้ามใช้ generic GB เปลี่ยนชื่อ |
| Naive Bayes | Gaussian class priors/means/variances และ log-posterior | variance smoothing | `GaussianNB` | closed-form fixture/log-space checks; one-class/zero variance policy |
| Support Vector Machine | Linear soft-margin hinge lossด้วย deterministic subgradient optimization | standardization; `C=[0.1,1,10]`; no-PCA/PCA `[2,4,6]`; scratch kernel=`linear` | `sklearn.svm.LinearSVC`; RBF referenceเป็น optional enhancementเท่านั้น | decision scoreไม่เรียก probability; kernel SVMไม่อยู่ primary scratch scope |
| k-Nearest Neighbors | Distance computation, neighbor selection, voting/tie rule | scaling, `k`, metric, weights | `KNeighborsClassifier` | exact toy distances/ties; runtime/memory riskต่ำ-กลาง |
| Perceptron | Linear score, hard step/sign activation และ online mistake-driven update | scaling, learning rate, epochs, shuffle policy/seed | `sklearn.linear_model.Perceptron` | linearly separable fixture; scoreเป็น decision scoreไม่ใช่ probability |
| Single-layer Perceptron (SLP) | Neural-network style single trainable layer, sigmoid output, BCE และ batch/mini-batch gradient descent; ไม่มี hidden layer | standardization; learning rate `[0.001,0.01]`, L2 `[0,1e-4]`, batch size `[32,full]`, max epochs `1000`, patience `50` | `SGDClassifier(loss="log_loss")` เป็น reference ประกอบ | gradient/shape/probability tests; อธิบายความเทียบเท่ากับ Logistic Regression และห้ามใช้ code aliasเดียวกัน |
| Multi-layer Perceptron | ReLU hidden layers, sigmoid output, BCE, backpropagation และ mini-batch gradient descent | hidden `[(16,),(32,),(16,8)]`, learning rate `[0.001,0.01]`, L2 `[0,1e-4]`, batch 32, max epochs 1000, patience 50 | `sklearn.neural_network.MLPClassifier` | gradient checks; chronological internal Train-tail early stoppingตาม Section 14.4; convergence riskสูง |
| AdaBoost | Binary discrete AdaBoost labels `{-1,+1}`; weighted decision stumps, error, alpha, exponential weight updateและ vote | estimators `[25,50,100]`, learning rate `[0.5,1.0]`; fail/stopเมื่อ weak learner error `>=0.5` | `sklearn.ensemble.AdaBoostClassifier` | toy weight updates; noisy-label sensitivity; decision stumpต้องรองรับ sample weightsจริง |
| k-Means | k-means++ initialization, assign, recompute, inertia, convergenceและ 10 restarts | standardization; `k=[2,3,4,5,6]`, `n_init=10`, `max_iter=300`, `tol=1e-6`, seed 42 | `sklearn.cluster.KMeans` | toy clusters/inertia monotonicity; label permutation-aware comparison |
| Agglomerative Clustering | Pairwise distancesและ iterative linkage merges | standardization; `k=[2,3,4,5,6]`, linkage `[ward,average]`; Euclidean metric; 10 configs | `sklearn.cluster.AgglomerativeClustering` | verify merge sequence; O(n²); primary Train structure, secondary frozen nearest-centroid OOS extension |

### Dimensionality reduction

ใช้ `PCA` แบบ scratch (center data, covariance/SVD/eigendecomposition, explained variance, transform) เป็น preprocessing configuration สำหรับทั้ง SVM และ Random Forest ตามข้อความ PDF โดย:

- fit PCA จาก Train ของ variantนั้นเท่านั้น
- เปรียบเทียบ no-PCA กับ `n_components=[2,4,6]` และเลือกด้วย task-specific Validation metricเท่านั้น
- fit PCA ใหม่สำหรับ Non-Spike
- ไม่เพิ่ม PCA เป็น predictive model itemลำดับที่ 20

## 7. Leakage-Safe Training Protocol

1. `train`/`tune` โหลดเฉพาะ variant Train และ Original Validation แบบ read-only; Test loaderปิดโดย default
2. Validate schema, explicit feature order, targets, finite values, dates และ checksumsของ splitที่ phaseนั้นอนุญาต; pre-gate Test identityอ้าง accepted immutable manifest ไม่อ่าน Test CSV
3. สร้าง **fresh** preprocessor และ scratch/reference model ต่อ run
4. Fit scaler/PCA/polynomial mappingจาก Train ของ variantนั้นเท่านั้น
5. Hyperparameter selectionใช้ Train + chronological validation protocol; ห้าม shuffled/random CV โดยไม่คำนึง chronology/target overlap
6. Classification thresholdเลือกจาก Validation เท่านั้น; ranking metricsใช้ continuous score
7. Freeze feature policy, search space, budget, seeds, selection metric, clipping policy และ selected configก่อน Test
8. ประเมิน Test ครั้งสุดท้ายและบันทึกว่า Test ถูกเปิดดูเมื่อใด
9. ถ้าต้องส่ง With-Spike Test ก่อนทำ Non-Spike ให้ freeze Non-Spike protocol/search spacesไว้ก่อน และห้ามแก้จาก With-Spike Test feedback
10. Non-Spike ใช้สูตร/code/search space/budget/seeds/selection metricเดียวกัน แต่ fit preprocessing/modelใหม่และเลือก best hyperparametersใหม่จาก Validationได้

### Special cases

- **Stacking:** ใช้ expanding/rolling chronological OOS predictions; base learnerของแต่ละ foldเห็นเฉพาะอดีต และ purgeตาม original trading timeline ไม่ใช่นับแถว Non-Spikeหลังกรองแบบตรง ๆ
- **SLP/MLP:** กัน chronological tail 15% ของ variant Trainเป็น internal early-stop split พร้อม purge 5 trading daysบน original timeline; แต่ละ candidateหา best epochจาก internal tail แล้ว refitบน full variant Trainตาม epoch countนั้นก่อนประเมิน external Validation ห้ามใช้ external Validationควบคุม epoch
- **Class weighting/resampling:** ทำเฉพาะ Train; Validation/Test distributionคงเดิม
- **Simple Linear:** feature choiceใช้ Validation ไม่ใช้ Test
- **Polynomial:** degree/interaction scopeใช้ Validation และมี term-count guard
- **PCA:** component countและ fitห้ามแตะ Validation/Testในการคำนวณ basis
- **Random models:** seedคงที่สำหรับ reproducibility; multi-seed stabilityเป็น analysisเสริม ไม่แทน primary run

## 8. Evaluation และ Presentation Contract

### 8.1 Regression

ผลหลัก:

- MSE/Loss, MAE, RMSE, R²
- Hyperparameter/model selectionใช้ **clipped Validation RMSE** เป็น primary และ clipped Validation MAE เป็น tie-breaker; raw metricsต้องบันทึกเสมอ
- Train-mean baseline
- Historical-volatility persistence baseline เช่นใช้ `historical_volatility_5d` เป็น prediction โดยต้องประกาศสูตรล่วงหน้า
- Actual vs predicted, residual plot/distribution, prediction CSVจับคู่ `Date`
- รายงานหน่วย targetเป็น decimal annualized volatility; ถ้าแสดง percentให้ระบุว่าเป็น percentage pointsอย่างถูกต้อง
- Operational prediction ใช้ `max(raw_prediction, 0.0)`; เก็บทั้ง raw/clipped metrics, predictions และ negative count ห้ามซ่อนค่าติดลบ

### 8.2 Classification

ผลหลัก:

- Confusion Matrix, Accuracy, Precision, Recall/Sensitivity, Specificity/TNR, F1
- ROC curve และ ROC-AUC เมื่อมี continuous score
- PR curve และ Average Precision; hyperparameter selectionใช้ **Validation Average Precision** เป็น primary และ ROC-AUC เป็น tie-breaker
- หลังเลือก hyperparametersแล้ว เลือก thresholdจาก Validation F1; ถ้าเสมอเลือก Recallสูงกว่า แล้วเลือก thresholdต่ำกว่าสำหรับ deterministic final tie-break
- รายงาน default threshold `0.5` สำหรับ probability และ `0.0` สำหรับ decision scoreเป็น baselineประกอบ
- Majority-class baseline
- zero-division และ one-class subset policyเป็น `N/A` พร้อมเหตุผล ไม่ใส่ค่าปลอม
- Perceptron/SVM ใช้ decision scoreสำหรับ ranking metrics; ไม่เรียก scoreว่า probability

### 8.3 Clustering

- Internal metrics เช่น inertia (k-Means), Silhouette score และ cluster size stability
- ใช้ standardized canonical featuresทั้ง 8 ตัว; ทดลอง `k=[2,3,4,5,6]` และเลือกจาก Train Silhouette score ไม่ใช้ Test labels
- ถ้าจะเทียบ clusterกับ `target_high_volatility` ต้องสร้าง mappingจาก Trainเท่านั้นและระบุว่าเป็น post-hoc analysis
- Agglomerative ไม่มี standard out-of-sample predict: primary evaluationเป็น Train structure/stability; Validation/Testใช้ได้เฉพาะ frozen nearest-centroid extensionและต้องประกาศว่าไม่ใช่ standard algorithm ห้าม fitใหม่บน Testแล้วเรียกว่า forecast
- Cluster IDs ระหว่าง variantsไม่จำเป็นต้องมี semantic mappingตรงกัน

### 8.4 Curves และ diagnostics

- แสดง learning/loss curvesเฉพาะ algorithmที่มี training historyจริง
- Closed-form/tree/k-NN ไม่สร้าง epoch curveปลอม; ใช้ validation-by-complexity, residual, tree depth, feature importanceหรือ error comparisonที่เหมาะสม
- Full Validation/Testเป็นผลหลัก
- Spike-Affected/Non-Affected segmentsเป็น diagnostics; empty segmentแสดง `N/A`

## 9. Artifacts, Paths, Save/Load และ Resume — Planned

เสนอ path convention:

```text
outputs/modeling/<variant>/<task>/<model>/<implementation>/<run_id>/
  config.json
  manifest.json
  model/
  preprocessing/
  search_results.json
  metrics_validation.json
  metrics_test.json             # finalized stage only
  predictions_validation.csv
  predictions_test.csv         # finalized/OOS-extension stage only
  figures/
  run_status.json
```

แต่ละ runต้องเก็บตาม lifecycle stage:

- variant, task, model, scratch/reference และ run/config ID
- feature names/order, target definition, preprocessing และ selected hyperparameters
- Train/Validation paths, SHA-256, rows/date rangesตั้งแต่ training stage; Test-specific path/hash/rows/date rangeเติมเฉพาะ `finalize-test` โดย pre-gateอ้างได้เพียง accepted immutable manifest ID
- seeds, dependency versions, Python version, git revision และ dirty status
- fitted model/preprocessor หรือ state arraysที่ loadกลับได้
- predictionsจับคู่ `Date` เฉพาะ split/taskที่ lifecycleนั้นรองรับ
- metrics, convergence/failure/skipped status, timing และ warnings
- strict JSON (`allow_nan=False`) และ portable relative paths

Resume ต้องตรวจ config hash, input hashes, deterministic code snapshot hash, artifact completenessและ load test; ห้าม skipเพียงเพราะ directoryมีอยู่ รองรับรันทีละ model และสถานะ `completed`, `failed`, `skipped`, `incompatible`, `running`

CLI ตัวอย่างต่อไปนี้เป็น **planned interface ยังรันไม่ได้**:

```text
python -m src.modeling.runner train --variant with_spike --task regression --model elastic_net --implementation scratch --seed 42 --search-budget 12 --output-root outputs/modeling
python -m src.modeling.runner evaluate --run-id <id> --split validation
python -m src.modeling.runner finalize-test --run-id <frozen-id>
python -m src.modeling.runner verify-load --run-id <id>
```

## 10. Milestones

Effort เป็นช่วง person-hoursโดยสมมติว่ามีผู้พัฒนา 1–2 คน, ใช้ accepted data artifactsเดิม, ไม่เปลี่ยน data contract และลด search budgetให้เหมาะกับ deadline

### M1 — Requirement reconciliation และ documentation update

- **Goal:** freeze scope, counting assumptions, ambiguities และ source of truth
- **Dependencies:** PDF/repository audit
- **Files:** `MODEL_TRAINING_PLAN.md` และ links/statusในเอกสารหลัก
- **Checklist:** matrix, 19-item inventory, With-Spike-first, mandatory/optional, frozen implementation configurationและ external approval note
- **Verification:** link/count/path consistency; git diff review
- **DoD:** เอกสารไม่อ้างผล modelที่ยังไม่มี
- **Priority/Risk/Effort:** P0 / ต่ำ / 3–6h

### M2 — Shared modeling infrastructure

- **Goal:** loader/contracts/preprocessing/metrics/persistence/artifact writer/CLI skeletonที่ทดสอบได้
- **Dependencies:** M1 decisions
- **Files:** `src/modeling/*`, tests, dependency update
- **Checklist:** explicit features, variant config, strict JSON, checksums, save/load, status/resume
- **Verification:** synthetic/unit tests + one no-op/dummy end-to-end artifact run
- **DoD:** runnerโหลดทั้ง variantsได้โดยไม่แก้ dataและ reject contract drift
- **Priority/Risk/Effort:** P0 / กลาง / 12–20h

### M3 — Pilot reuse proof

- **Goal:** พิสูจน์ frameworkกับหนึ่งโมเดลง่ายทั้งสอง variantsก่อนเปิด Test; นี่เป็น explicit exceptionเดียวต่อ With-Spike-first และไม่ถือว่าเริ่ม Non-Spike delivery phase
- **Dependencies:** M2
- **Files:** Multiple Linear Regression scratch/reference + tests
- **Checklist:** fresh scaler/modelต่อ variant, ใช้ Train/Validation เท่านั้น, save/load reload equality; ไม่เปิด Test และไม่ใช้ pilotเลือก protocolจาก variantที่ได้คะแนนดีกว่า
- **Verification:** synthetic fixture + artifact completeness; pilotมีไว้พิสูจน์ code reuse ไม่ใช่ผลส่งหรือ paired conclusion
- **DoD:** ทั้งสอง variantsรันผ่าน plumbingด้วย configต่างเฉพาะ data provenance; Non-Spike pilot artifactsติดป้าย `pilot_only=true`, ห้าม Test/paired conclusion และหยุด Non-Spikeไว้จน M9 จากนั้นกลับไปทำ With-Spike inventory M4–M8 ให้ครบ
- **Priority/Risk/Effort:** P0 / ต่ำ-กลาง / 5–8h

### M4 — With-Spike regressionทั้งหมด

- **Goal:** Simple, Multiple, Polynomial, Elastic Net scratch/reference/evaluation
- **Dependencies:** M2–M3
- **Files:** regression models/tests/notebook sections/artifacts
- **Checklist:** feature/degree selectionจาก Validation, baselines, negative prediction policy
- **Verification:** synthetic/reference comparisons + reload tests
- **DoD:** 4 mandatory regression itemsมี convergence/status, Validationและ frozen Test artifacts
- **Priority/Risk/Effort:** P0 / กลาง / 14–24h

### M5 — With-Spike classificationพื้นฐาน

- **Goal:** Logistic, Naive Bayes, k-NN, Perceptron, SLP และ Decision Tree
- **Dependencies:** M2
- **Checklist:** scores/probabilitiesถูกประเภท, imbalance/threshold policy, confusion metrics
- **Verification:** toy fixtures/reference comparisons
- **DoD:** 6 modelsมี scratch/referenceและ required metrics; Perceptron และ SLP มี training rule/output contract แยกกัน
- **Priority/Risk/Effort:** P0 / กลาง / 20–34h

### M6 — With-Spike advanced classifiers

- **Goal:** Random Forest, Gradient Boosting, AdaBoost, SVM, MLP, Stacking, XGBoost
- **Dependencies:** M5 primitives; PCA/chronological OOS utilities
- **Checklist:** algorithm fidelity, stacking purge, MLP early stopping, PCA variant, failure reporting
- **Verification:** synthetic behavior, reference comparison, convergence/resource limits
- **DoD:** 7 modelsมี honest status; failureไม่ถูกซ่อนหรือแทนด้วย library wrapper
- **Priority/Risk/Effort:** P0 ตาม requirement / สูงมาก / 40–80h

### M7 — Clustering และ dimensionality reduction

- **Goal:** k-Means, Agglomerative และ PCA configuration
- **Dependencies:** M2
- **Checklist:** train-only scaling/PCA, cluster-count selection, no fake Test forecast
- **Verification:** toy merge/cluster fixtures, permutation-aware checks
- **DoD:** 2 clustering itemsและ PCA mechanismอธิบาย/ทดสอบครบ
- **Priority/Risk/Effort:** P0 ตาม interpreted PDF / กลาง-สูง / 10–18h

### M8 — With-Spike submission package

- **Goal:** report/slides/notebooks/scriptsที่ส่งได้สำหรับ With-Spikeครบรายการหลัก
- **Dependencies:** M4–M7
- **Checklist:** tables, plots, mechanism explanations, baselines, feature insight, why ML, save/load demo
- **Verification:** artifact-to-document trace, rerun selected demos, no invented metrics
- **DoD:** ทุก claimมี artifact; limitations/failuresเปิดเผย
- **Priority/Risk/Effort:** P0 / สูง / 16–28h

### M9 — Non-Spike reruns

- **Goal:** fitใหม่ทั้ง inventoryด้วย framework/protocolเดียวกัน
- **Dependencies:** protocol/search spaces frozenก่อน With-Spike Test feedback; M8 code complete
- **Checklist:** fresh scaler/PCA/model, same budgets/seeds/selection metrics, variant-specific artifacts
- **Verification:** provenance/config parity checksและ load tests
- **DoD:** Non-Spikeมี complete run statusทุก item; ไม่ reuse fitted stateจาก With-Spike
- **Priority/Risk/Effort:** P1 / กลาง-สูง / 18–36h + compute

### M10 — Paired comparison และ final documentation

- **Goal:** เปรียบเทียบ variantsอย่างยุติธรรมและปิดเอกสาร
- **Dependencies:** M9
- **Checklist:** Full Testหลัก, diagnostic segmentsรอง, paired deltas, limitations/sample-size trade-off
- **Verification:** programmatic joinโดย model/config IDs; countsครบ; no Test-driven redesign
- **DoD:** จึงจะใช้คำว่า paired comparison completedได้
- **Priority/Risk/Effort:** P1 / กลาง / 8–16h

## 11. Critical Path ถึง 12 October 2026

ณ 2026-10-08 เหลือประมาณ 4 วันปฏิทินก่อน deadline วันที่ 2026-10-12 แผนเต็ม 19 scratch algorithms + reference + tuning + save/load + report/slides มี effortมากกว่าช่วงเวลาที่เหลืออย่างชัดเจน โดยเฉพาะ XGBoost, SVM, MLP, Stacking และ Agglomerative ดังนั้นห้ามรับประกันว่าจะเสร็จครบโดยไม่มีทีม/implementationเดิมเพิ่มเติม

ลำดับ critical path:

1. ใช้ frozen implementation configuration ใน Section 14; Perceptron/SLP, PCA, linear SVM, faithful minimal XGBoost, clustering และ evaluation policiesปิดแล้ว
2. สร้าง M2 infrastructureและ M3 Train/Validation-only pilotก่อน เพื่อพิสูจน์ reuse โดยไม่เปิด Testหรือเริ่ม Non-Spike submission work แล้วหยุด Non-Spikeไว้จนหลัง M8
3. ทำ With-Spike scratch modelsตามลำดับ: regression → basic classification → advanced/high-risk → clustering
4. ทำ reference comparisons, save/load testsและเอกสารควบคู่ ไม่รอท้ายสุด
5. Freeze protocolก่อน Test; สร้าง With-Spike submission package
6. หลัง With-Spike frameworkนิ่งจึง rerun Non-Spikeด้วย fresh fits
7. ทำ paired comparisonเฉพาะเมื่อทั้งสอง variantsครบจริง

ถ้าเวลาขาด ให้ลด **optional enhancements** ก่อน: broad hyperparameter grids, multi-seed stability, detector sensitivity variants, interactive dashboards, extensive ablations และ cosmetic plots ห้ามย้าย modelที่ตีความว่า PDFบังคับไปเป็น optionalเพียงเพราะทำยาก

## 12. Mandatory vs Optional

### จำเป็นเพื่อส่งตาม requirementที่ตีความ

- 19-item inventory ตาม course interpretation ที่ผู้ใช้ยืนยัน โดย Perceptron และ SLP เป็นคนละ model
- Scratch core logicทุก model และ ready-to-use referenceประกอบ
- Internal mechanism, save/load/reload test
- Required quantitative metrics/plotsที่เหมาะกับ task
- With-Spike-first complete submission
- Non-Spikeอยู่ในแผนและต้อง fitใหม่ก่อนอ้าง paired comparison
- Report, slides, notebooks/scripts และ provenance

### Optional enhancements

- Multi-seed confidence/stability analysisเกิน primary seed
- Broad Bayesian/random search
- Spike detector sensitivity model runs
- SHAP/advanced explainabilityถ้า dependency/timeไม่พร้อม
- Interactive dashboard
- Additional sequence/deep modelsนอก inventory

## 13. Risk Register

| Risk | Impact | Mitigation / gate | Owner/decision deadline |
| --- | --- | --- | --- |
| เวลาไม่พอสำหรับ 19 scratch modelsก่อน 2026-10-12 | ส่งไม่ครบหรือ verificationตื้น | ทำ critical path, จำกัด search budget, ลด optional workก่อน; รายงานสถานะจริง | ทีมโครงการ / ปิด scopeทันที |
| XGBoost/SVM/MLP/Stacking fidelityไม่ตรงชื่อ | เสียคะแนน from-scratch/algorithm mechanism | freeze minimum faithful algorithm, synthetic testsและ reference comparisonก่อน full data | model implementer / ก่อนเริ่มแต่ละ model |
| `.venv` 3.14.8 ต่างจาก recorded 3.11.9 (ผ่าน 3.12.3 มาแล้ว) | reproduceไม่ได้หรือ library incompatibility | ใช้ Python 3.14.8; ยืนยันแล้วว่า reference dependencies (`scikit-learn`, `xgboost` รวมอยู่ใน `requirements-lock.txt`) ติดตั้ง/import สำเร็จและ pipeline/tests ให้ผลตรงเดิม | infrastructure owner / M2 gate |
| Elastic Net/AdaBoostไม่ถูกยอมรับเป็น extracurricular | ไม่ผ่าน requirement new-model | ขอคำยืนยันอาจารย์; เตรียมเหตุผลและ candidateสำรองโดยไม่เปลี่ยน Test protocol | team lead / ก่อน final scope freeze |
| Test feedbackจาก With-Spikeปนการออกแบบ Non-Spike | paired comparisonมี leakage/bias | freeze Non-Spike protocol/search spacesก่อนเปิด With-Spike Test; log test-access timestamp | experiment owner / ก่อน M8 Test |
| Stacking purgeบน Non-Spikeใช้ row gapแทน original timeline | future leakage | foldด้วย original dates/positionsและ explicit purge tests | stacking owner / M6 |
| Broad tuning/computeเกินเวลา | ไม่มี artifactsครบหรือ rerunไม่ได้ | fixed small budgets, per-model timeout, resume/status artifacts | experiment owner / M2 ก่อน batch runs |
| Agglomerativeไม่มี standard OOS predict | อ้าง forecastไม่ถูกต้อง | จำกัด primary evaluationเป็น Train structure/stability หรือประกาศ extensionชัดเจน | clustering owner / M7 |

## 14. Frozen Implementation Configuration

ค่าต่อไปนี้เป็น implementation contract ที่ freeze เมื่อ **2026-10-08** ให้ Codex ใช้เป็นข้อกำหนด ไม่ใช่คำแนะนำแบบเลือกได้ การเปลี่ยนค่าหลังเริ่ม Test ต้องสร้าง protocol revision ใหม่และห้ามนำผลต่างมาเปรียบเทียบเหมือนเป็น protocol เดียวกัน

> **อัปเดต 2026-10-10:** ยังไม่มี Test เริ่มและไม่มี model implementation ใด ๆ (ดูข้อ
> 1.1) จึงแก้ `environment.python` จาก `3.12.3` เป็น `3.14.8` ได้โดยไม่ต้องเปิด
> protocol revision ใหม่ ยืนยันแล้วว่า `requirements-lock.txt` ทั้งหมดติดตั้งและ
> import สำเร็จบน Python 3.14.8, `pip check` ผ่าน และ Phase 1/Phase 2 pipeline ให้
> ผลตรงกับหลักฐานเดิมทุกจุด ก่อน sync ไฟล์นี้และ `configs/modeling.json` ด้วย
> `python -m src.modeling.config_sync --write-plan`

### 14.1 Configuration source และ schema

- M2 ต้อง materialize ค่าจาก section นี้เป็น `configs/modeling.json` เพราะอ่านด้วย Python standard-library `json` ได้โดยไม่เพิ่ม YAML dependency
- Frozen revisionคือ `modeling-config-2026-10-08-v1`; JSONต้องมี `schema_version`, `plan_revision`, `plan_section_sha256`, `config_id` และ `config_sha256`
- M2 ทำให้ `configs/modeling.json` เป็น authoritative machine-readable values แล้วใช้ `python -m src.modeling.config_sync --write-plan` render blockระหว่าง `BEGIN/END GENERATED MODELING CONFIG` markersจาก JSON; `--check` ต้อง regenerateใน memoryและ byte-compare blockเพื่อพิสูจน์ semantic equality ไม่ใช่ตรวจ revision/hashอย่างเดียว
- Generated block normalizeเป็น UTF-8, LF, ไม่มี BOM และมี newlineเดียวท้าย block; rendererไม่รวม integrity metadata fieldsใน rendered summary จากนั้นคำนวณ `plan_section_sha256`จาก bytesระหว่าง markersแบบ exclusive
- `config_sha256` คำนวณจาก canonical JSON (`sort_keys=True`, compact separators, UTF-8, no NaN) หลังตัด fields `config_id` และ `config_sha256`; `config_id="cfg-" + config_sha256[:12]` แล้วจึงเขียนสอง fieldsกลับลงไฟล์
- Runner ต้อง validate unknown/missing keys และ reject invalid combinations เช่น PCA components มากกว่าจำนวน features
- Dataset pathsใช้ constantsจาก `config.py`; executable configอ้างชื่อ variant ไม่ duplicate absolute paths

### 14.2 Environment และ dependency freeze

- **Python:** `3.14.8` (re-frozen 2026-10-10; เดิม `3.12.3`)
- Environment ที่ตรวจจริงก่อน implementation มี `numpy==2.5.3`, `pandas==3.0.6`, `pytest==9.1.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `nbformat==5.11.1`, `nbconvert==7.17.1`, `ipykernel==7.4.0`, `scikit-learn==1.9.1`, `xgboost==3.4.1` ติดตั้งและตรวจผ่านบน Python 3.14.8 แล้ว
- M2 dependency gate: ติดตั้ง versionsที่รองรับ Python 3.14.8, import smoke test, รัน existing testsและ pilot reference comparison จากนั้นสร้าง `requirements-lock.txt` จาก environmentที่ผ่านจริง
- ใช้ `requirements.in` เก็บ direct dependenciesรวม `scikit-learn`, `xgboost` และ `joblib`; `requirements-lock.txt` เก็บ exact resolved versions; เปลี่ยน `requirements.txt` เป็น compatibility shimที่มี `-r requirements-lock.txt`; ถ้า dependency gateไม่ผ่านให้หยุดและบันทึก blocker ห้ามเปลี่ยน Python/packageแบบเงียบ ๆ
- PyTorchไม่ใช่ dependencyบังคับ; SLP/MLP referenceใช้ Scikit-learn เพื่อลด installation scope

<!-- BEGIN GENERATED MODELING CONFIG -->
Source: configs/modeling.json (authoritative). Model entries are configuration, not implementations.

### 14.3 Global experiment policy

```json
{
  "data": {
    "classification_target": "target_high_volatility",
    "date_column": "Date",
    "diagnostics": "segmentation_only",
    "feature_columns": [
      "return_1d",
      "return_5d",
      "historical_volatility_5d",
      "historical_volatility_20d",
      "intraday_range",
      "sma_ratio_5_20",
      "rsi_14",
      "volume_zscore_20"
    ],
    "q75_source": "original_train_only",
    "recompute_features_targets": false,
    "regression_target": "target_volatility_5d",
    "test": "full_original",
    "validation": "full_original",
    "variants": [
      "with_spike",
      "non_spike"
    ]
  },
  "environment": {
    "dependency_lock": "requirements-lock.txt",
    "python": "3.14.8",
    "reference_dependencies": [
      "scikit-learn",
      "xgboost",
      "joblib"
    ]
  },
  "experiment": {
    "failure_policy": "record_failure_no_reference_substitution",
    "max_candidates": 12,
    "non_spike_full_milestone": 9,
    "pilot": {
      "fresh_fit": true,
      "milestone": 3,
      "model": "multiple_linear",
      "pilot_only": true,
      "train_validation_only": true
    },
    "polynomial_max_candidates": 16,
    "reference_comparison": "same_features_preprocessing_seed_split",
    "resampling": false,
    "search_strategy": "explicit_deterministic",
    "seed": 42,
    "shuffle": false,
    "stability_after_mandatory": true,
    "stability_seeds": [
      7,
      42,
      2026
    ],
    "variant_order": [
      "with_spike",
      "non_spike"
    ]
  },
  "plan_revision": "modeling-config-2026-10-08-v1",
  "schema_version": 1
}
```

### 14.4 Frozen model scopes and search spaces

```json
{
  "adaboost": {
    "alpha_factor": 0.5,
    "epsilon": 1e-12,
    "error_le_epsilon": "completed_perfect",
    "first_error_ge_half": "failed",
    "labels": [
      -1,
      1
    ],
    "later_error_ge_half": "completed_early",
    "search_both_polarities": true,
    "stump_tie": [
      "feature_index",
      "threshold",
      "normal_polarity"
    ],
    "weights": "log_space_logsumexp",
    "zero_vote_label": 1
  },
  "models": {
    "adaboost": {
      "class_weight": [
        null
      ],
      "grid": {
        "learning_rate": [
          0.5,
          1
        ],
        "n_estimators": [
          25,
          50,
          100
        ]
      },
      "parameters": {
        "external_class_weight": false
      },
      "preprocess": "none",
      "scope": "discrete_weighted_stumps",
      "task": "classification"
    },
    "agglomerative": {
      "class_weight": [
        null
      ],
      "grid": {
        "k": [
          2,
          3,
          4,
          5,
          6
        ],
        "linkage": [
          "ward",
          "average"
        ]
      },
      "parameters": {
        "metric": "euclidean",
        "oos_extension": "frozen_nearest_centroid",
        "standard_predict": false
      },
      "preprocess": "standardize",
      "scope": "train_structure",
      "task": "clustering"
    },
    "decision_tree": {
      "class_weight": [
        null
      ],
      "grid": {
        "max_depth": [
          2,
          3,
          5
        ],
        "min_samples_leaf": [
          5,
          10
        ]
      },
      "parameters": {
        "criterion": "gini",
        "tie": [
          "feature_index",
          "threshold"
        ]
      },
      "preprocess": "none",
      "scope": "gini",
      "task": "classification"
    },
    "elastic_net": {
      "class_weight": [
        null
      ],
      "grid": {
        "alpha": [
          0.0001,
          0.001,
          0.01,
          0.1
        ],
        "l1_ratio": [
          0.25,
          0.5,
          0.75
        ]
      },
      "parameters": {
        "intercept": true,
        "max_iter": 5000,
        "objective": "squared_error_over_2n_plus_alpha_l1_plus_half_alpha_l2",
        "selection": "cyclic",
        "tol": 1e-06
      },
      "preprocess": "standardize",
      "scope": "coordinate_descent",
      "task": "regression"
    },
    "gaussian_nb": {
      "class_weight": "not_supported",
      "grid": {
        "var_smoothing": [
          1e-09,
          1e-08,
          1e-07
        ]
      },
      "parameters": {
        "priors": "train_empirical"
      },
      "preprocess": "none",
      "scope": "gaussian_log_space",
      "task": "classification"
    },
    "gradient_boosting": {
      "class_weight": [
        null
      ],
      "grid": {
        "learning_rate": [
          0.03,
          0.1
        ],
        "max_depth": [
          1,
          2
        ],
        "n_estimators": [
          50,
          100
        ]
      },
      "parameters": {
        "loss": "binary_logistic"
      },
      "preprocess": "none",
      "scope": "binary_logistic_regression_trees",
      "task": "classification"
    },
    "kmeans": {
      "class_weight": [
        null
      ],
      "grid": {
        "k": [
          2,
          3,
          4,
          5,
          6
        ]
      },
      "parameters": {
        "init": "k-means++",
        "max_iter": 300,
        "n_init": 10,
        "tol": 1e-06
      },
      "preprocess": "standardize",
      "scope": "kmeans_plus_plus",
      "task": "clustering"
    },
    "knn": {
      "class_weight": "not_supported",
      "grid": {
        "k": [
          3,
          5,
          9,
          15
        ],
        "weights": [
          "uniform",
          "distance"
        ]
      },
      "parameters": {
        "tie": "deterministic"
      },
      "preprocess": "standardize",
      "scope": "euclidean",
      "task": "classification"
    },
    "logistic": {
      "class_weight": [
        null,
        "balanced"
      ],
      "grid": {
        "class_weight": [
          null,
          "balanced"
        ],
        "l2": [
          0,
          0.001,
          0.01
        ]
      },
      "parameters": {
        "balanced_weight": "n_samples_over_2_class_count",
        "learning_rate": 0.05,
        "max_iter": 5000,
        "objective": "weighted_mean_bce_plus_half_lambda_l2",
        "penalize_intercept": false,
        "tol": 1e-06
      },
      "preprocess": "standardize",
      "scope": "sigmoid_bce_gradient_descent",
      "task": "classification"
    },
    "mlp": {
      "class_weight": [
        null
      ],
      "grid": {
        "hidden_layers": [
          [
            16
          ],
          [
            32
          ],
          [
            16,
            8
          ]
        ],
        "l2": [
          0,
          0.0001
        ],
        "learning_rate": [
          0.001,
          0.01
        ]
      },
      "parameters": {
        "batch": 32,
        "max_epochs": 1000,
        "patience": 50,
        "shuffle": false
      },
      "preprocess": "standardize",
      "scope": "relu_hidden_sigmoid_bce",
      "task": "classification"
    },
    "multiple_linear": {
      "class_weight": [
        null
      ],
      "grid": {},
      "parameters": {
        "intercept": true,
        "solver": "numpy.linalg.pinv"
      },
      "preprocess": "standardize",
      "scope": "pinv_ols",
      "task": "regression"
    },
    "perceptron": {
      "class_weight": [
        null
      ],
      "grid": {
        "learning_rate": [
          0.001,
          0.01,
          0.1
        ]
      },
      "parameters": {
        "max_epochs": 1000,
        "output": "decision_score",
        "report_mistake_rate": true,
        "shuffle": false
      },
      "preprocess": "standardize",
      "scope": "online_mistake_update",
      "task": "classification"
    },
    "polynomial": {
      "class_weight": [
        null
      ],
      "grid": {
        "degree": [
          2,
          3
        ],
        "feature": [
          "return_1d",
          "return_5d",
          "historical_volatility_5d",
          "historical_volatility_20d",
          "intraday_range",
          "sma_ratio_5_20",
          "rsi_14",
          "volume_zscore_20"
        ]
      },
      "parameters": {
        "intercept": true,
        "standardize_before_powers": true
      },
      "preprocess": "standardize",
      "scope": "univariate_powers_no_interactions",
      "task": "regression"
    },
    "random_forest": {
      "class_weight": [
        null
      ],
      "grid": {
        "max_depth": [
          3,
          5
        ],
        "n_estimators": [
          50,
          100
        ]
      },
      "parameters": {
        "bootstrap": true,
        "final_candidates": 7,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "pca_uses_best_no_pca": true
      },
      "preprocess": "none",
      "scope": "bootstrap_forest",
      "task": "classification"
    },
    "simple_linear": {
      "class_weight": [
        null
      ],
      "grid": {
        "feature": [
          "return_1d",
          "return_5d",
          "historical_volatility_5d",
          "historical_volatility_20d",
          "intraday_range",
          "sma_ratio_5_20",
          "rsi_14",
          "volume_zscore_20"
        ]
      },
      "parameters": {
        "intercept": true
      },
      "preprocess": "none",
      "scope": "univariate_ols_intercept",
      "task": "regression"
    },
    "slp": {
      "class_weight": [
        null
      ],
      "grid": {
        "batch": [
          32,
          "full"
        ],
        "l2": [
          0,
          0.0001
        ],
        "learning_rate": [
          0.001,
          0.01
        ]
      },
      "parameters": {
        "max_epochs": 1000,
        "patience": 50,
        "shuffle": false
      },
      "preprocess": "standardize",
      "scope": "single_sigmoid_bce",
      "task": "classification"
    },
    "stacking": {
      "class_weight": [
        null
      ],
      "grid": {},
      "parameters": {
        "protocol": "stacking"
      },
      "preprocess": "base_specific",
      "scope": "expanding_train_oos_meta",
      "task": "classification"
    },
    "svm": {
      "class_weight": [
        null
      ],
      "grid": {
        "C": [
          0.1,
          1,
          10
        ]
      },
      "parameters": {
        "final_candidates": 12,
        "kernel": "linear",
        "objective": "half_l2_plus_C_sum_hinge"
      },
      "preprocess": "standardize",
      "scope": "linear_soft_margin_subgradient",
      "task": "classification"
    },
    "xgboost": {
      "class_weight": [
        null
      ],
      "grid": {
        "tuples": [
          [
            50,
            1,
            0.1,
            1,
            0
          ],
          [
            50,
            2,
            0.1,
            1,
            0
          ],
          [
            100,
            1,
            0.03,
            1,
            0
          ],
          [
            100,
            2,
            0.03,
            1,
            0
          ],
          [
            100,
            2,
            0.1,
            1,
            0
          ],
          [
            100,
            2,
            0.1,
            10,
            0
          ],
          [
            100,
            2,
            0.1,
            1,
            0.1
          ],
          [
            100,
            2,
            0.03,
            10,
            0.1
          ]
        ]
      },
      "parameters": {
        "exact_splits": true,
        "gradient_hessian": true,
        "regularized_gain_leaf_weights": true,
        "shrinkage": true,
        "tuple_fields": [
          "n_estimators",
          "max_depth",
          "learning_rate",
          "reg_lambda",
          "gamma"
        ]
      },
      "preprocess": "none",
      "scope": "exact_second_order_regularized_trees",
      "task": "classification"
    }
  },
  "neural": {
    "batch_order": "contiguous_chronological",
    "best_epoch_tie": "earliest",
    "internal_tail_fraction": 0.15,
    "last_short_batch": true,
    "min_delta": 1e-06,
    "monitor": "binary_cross_entropy",
    "patience": 50,
    "purge_original_trading_days": 5,
    "refit_early_stopping": false,
    "refit_epochs": "selected_best_epoch",
    "refit_full_train": true,
    "refit_seed": 42,
    "scaler_fit": "subtrain"
  },
  "stacking": {
    "bases": [
      "logistic",
      "decision_tree",
      "knn"
    ],
    "blocks": "numpy.array_split",
    "folds": 4,
    "initial_original_fraction": 0.4,
    "invalid_fold": "failed",
    "meta": "logistic",
    "meta_class_weight": null,
    "meta_features": [
      "logistic_probability",
      "tree_probability",
      "knn_probability"
    ],
    "meta_fit": "train_oos_only",
    "position_join": "exact_one_to_one_date",
    "purge_original_trading_days": 5,
    "refit_bases": "full_variant_train",
    "train_boundary": "position < validation_start - purge"
  }
}
```

### 14.5 Preprocessing matrix

```json
{
  "preprocessing": {
    "dtype": "float64",
    "fit_scope": "variant_train_only",
    "pca": {
      "basis": "projector_feature_order_modified_gram_schmidt",
      "components": [
        2,
        4,
        6
      ],
      "covariance_ddof": 1,
      "degenerate_relative_tolerance": 1e-10,
      "eigensolver": "eigh",
      "include_no_pca": true,
      "models": [
        "random_forest",
        "svm"
      ],
      "negative_eigenvalue_tolerance": 1e-12,
      "rank_eigenvalue_floor": 1e-12,
      "rank_failure": "invalid_candidate",
      "sign": "largest_absolute_positive_lowest_index_tie"
    },
    "standardizer": {
      "constant_output": 0,
      "constant_scale": 1,
      "ddof": 0,
      "record_warnings": true,
      "variance_floor": 1e-12
    }
  }
}
```

### 14.6 Selection, threshold and reporting policies

```json
{
  "losses": {
    "adaboost": "exponential_loss",
    "decision_tree": "log_loss",
    "gaussian_nb": "log_loss",
    "gradient_boosting": "binary_logistic_loss",
    "knn": "log_loss",
    "logistic": "binary_cross_entropy",
    "mlp": "binary_cross_entropy",
    "perceptron": "perceptron_criterion",
    "random_forest": "log_loss",
    "slp": "binary_cross_entropy",
    "stacking": "log_loss",
    "svm": "hinge_loss",
    "xgboost": "binary_logistic_loss"
  },
  "selection": {
    "classification": [
      "average_precision",
      "roc_auc"
    ],
    "clustering": {
      "mapping": "train_majority_only",
      "mapping_tie_class": 1,
      "primary": "train_silhouette",
      "secondary": [
        "inertia",
        "cluster_sizes",
        "stability"
      ]
    },
    "defaults": {
      "decision": 0,
      "probability": 0.5
    },
    "probability_epsilon": 1e-15,
    "record_negative_count": true,
    "regression": [
      "clipped_rmse",
      "clipped_mae"
    ],
    "regression_clip_min": 0,
    "retain_raw": true,
    "score_only_probability": false,
    "threshold": [
      "f1",
      "recall",
      "lower_threshold"
    ],
    "threshold_split": "validation",
    "undefined": {
      "reason_required": true,
      "value": null
    }
  }
}
```

### 14.7 Test access gate

```json
{
  "test_gate": {
    "after_test_revision": "new_protocol_and_run_no_test_selection",
    "command": "finalize-test",
    "default_allow_test": false,
    "freeze_non_spike_before_test": true,
    "record": [
      "test_accessed",
      "utc_timestamp",
      "config_sha256",
      "test_input_sha256",
      "git_revision",
      "code_snapshot_sha256",
      "caller_command"
    ],
    "require_clean_git": true,
    "require_frozen_config": true,
    "require_load_verification": true,
    "require_output_policy": true,
    "require_selected_hyperparameters": true,
    "require_validation_artifacts": true
  }
}
```

### 14.8 Save/load, artifacts and resume contract

```json
{
  "persistence": {
    "code_order": "posix_lexicographic",
    "code_paths": [
      "config.py",
      "configs/modeling.json",
      "src/modeling/**/*.py",
      "src/models/**/*.py"
    ],
    "code_stream": "path_utf8_nul_uint64be_byte_length_file_bytes",
    "labels_exact": true,
    "output_path": "outputs/modeling/<variant>/<task>/<model>/<implementation>/<run_id>/",
    "overwrite": false,
    "reference": "joblib",
    "reload_atol": 1e-12,
    "reload_rtol": 1e-10,
    "resume_hashes": [
      "config",
      "inputs",
      "code",
      "dependency_lock",
      "runtime"
    ],
    "resume_require_artifacts": true,
    "resume_require_load_verification": true,
    "resume_status": "completed",
    "runtime_fields": [
      "python_full_version",
      "platform",
      "machine",
      "numpy",
      "pandas",
      "scikit-learn",
      "xgboost",
      "joblib",
      "normalized_numpy_show_config_sha256"
    ],
    "scratch": "npz_strict_json"
  }
}
```

### 14.9 Artifact requirements by task/lifecycle

```json
{
  "artifacts": {
    "assignment_columns": [
      "Date",
      "cluster_id",
      "distance",
      "assignment_method"
    ],
    "clustering_extension": [
      "extension.json",
      "validation_assignments.csv",
      "extension_metrics.json",
      "limitations.json"
    ],
    "clustering_train": [
      "config.json",
      "manifest.json",
      "model.npz",
      "preprocessor.npz",
      "train_assignments.csv",
      "internal_metrics.json",
      "cluster_summary.json",
      "figures",
      "load_verification.json"
    ],
    "failed": [
      "stage",
      "exception_type",
      "message",
      "traceback_path",
      "timing",
      "hashes"
    ],
    "incompatible": [
      "mismatches"
    ],
    "mandatory_completed_only": true,
    "pilot_satisfies_finalized": false,
    "prediction_columns": [
      "Date",
      "target",
      "raw_prediction_or_score",
      "final_prediction",
      "threshold_or_clipping_indicator"
    ],
    "run_roles": [
      "pilot_only",
      "candidate",
      "finalized_scratch",
      "finalized_reference",
      "internal_fold",
      "optional_stability"
    ],
    "skipped": [
      "reason",
      "required"
    ],
    "supervised_finalized": [
      "test_identity.json",
      "test_metrics.json",
      "test_predictions.csv",
      "test_access.json"
    ],
    "supervised_tuned": [
      "config.json",
      "manifest.json",
      "model.npz",
      "preprocessor.npz",
      "search_results.json",
      "validation_metrics.json",
      "validation_predictions.csv",
      "load_verification.json"
    ]
  }
}
```

### 14.10 Scratch/reference comparison contract

```json
{
  "references": {
    "adaboost": {
      "api_difference": "record_installed_syntax_in_manifest_and_tests",
      "comparison": "behavioral-reference",
      "constructor": "AdaBoostClassifier",
      "parameters": {
        "criterion": "gini",
        "estimator": "DecisionTreeClassifier",
        "max_depth": 1,
        "random_state": 42
      }
    },
    "elastic_net": {
      "comparison": "objective-parity",
      "constructor": "ElasticNet",
      "parameters": {
        "fit_intercept": true,
        "max_iter": 5000,
        "selection": "cyclic",
        "tol": 1e-06
      }
    },
    "logistic": {
      "comparison": "objective-behavioral",
      "constructor": "SGDClassifier",
      "l2_mapping": "lambda_zero_penalty_none_else_l2_alpha_lambda",
      "parameters": {
        "eta0": 0.05,
        "learning_rate": "constant",
        "loss": "log_loss",
        "random_state": 42,
        "shuffle": false
      }
    },
    "mlp": {
      "comparison": "behavioral-reference",
      "constructor": "MLPClassifier",
      "parameters": {
        "activation": "relu",
        "batch_size": 32,
        "early_stopping": false,
        "random_state": 42,
        "shuffle": false,
        "solver": "sgd",
        "tol": 0
      },
      "selected_parameters": [
        "max_iter_selected_epoch",
        "n_iter_no_change_selected_epoch_plus_one",
        "hidden_layer_sizes",
        "learning_rate_init",
        "alpha"
      ]
    },
    "others": {
      "comparison": "behavioral-reference",
      "internal_state_equality": false,
      "serialize_constructor_parameters": true
    },
    "slp": {
      "comparison": "behavioral-reference",
      "constructor": "SGDClassifier",
      "parameters": {
        "early_stopping": false,
        "learning_rate": "constant",
        "loss": "log_loss",
        "random_state": 42,
        "shuffle": false,
        "tol": null
      },
      "selected_parameters": [
        "max_iter_selected_epoch",
        "eta0_selected_lr",
        "alpha_selected_l2",
        "penalty_none_if_zero"
      ]
    },
    "svm": {
      "comparison": "behavioral-reference",
      "constructor": "LinearSVC",
      "parameters": {
        "fit_intercept": true,
        "loss": "hinge",
        "random_state": 42
      }
    }
  }
}
```
<!-- END GENERATED MODELING CONFIG -->

### 14.11 ข้อที่ยังต้องยืนยันภายนอกแต่ไม่บล็อก M2/M3

เหลือเพียงการยืนยันกับอาจารย์ว่า **Elastic Net** และ **AdaBoost** ได้รับการยอมรับเป็น extracurricular models ตาม PDF หรือไม่ ระหว่างรอให้ implementตาม frozen scopeนี้และห้ามใช้ Testเพื่อเลือก candidateสำรอง

## 15. งานแรกสำหรับรอบ Implementation ถัดไป

เริ่ม **M2 + M3 pilot** โดยยังไม่กระจาย implement 19 models:

1. ทำ Python 3.12.3 dependency gate, เพิ่ม `requirements.in`, `requirements-lock.txt`, compatibility `requirements.txt`, import smoke testและ existing targeted tests
2. สร้าง `configs/modeling.json`, schema validator และ `src/modeling/config_sync.py`; ตรวจ canonical hashesและ generated-plan blockด้วย `--check`
3. เพิ่ม `src/modeling/contracts.py` และ `datasets.py` เพื่อโหลด/validateสอง variantsแบบ read-only, เลือก `FEATURE_COLUMNS` explicit และ enforce phase-specific Test gate
4. เพิ่ม deterministic Standardizer/PCA primitives, metric/loss/threshold policies, lifecycle-aware artifact manifest, strict JSON, code/runtime fingerprints และ save/load compatibility check
5. Implement Multiple Linear Regression scratch + Scikit-learn referenceเป็น pilot
6. รัน Train/Validation-only pilotบน With-Spikeและ Non-Spikeด้วย fresh fitทุกส่วน; Non-Spikeติด `pilot_only=true`, ไม่เปิด Test ไม่สรุป paired result และหยุด Non-Spikeไว้จน M9
7. เขียน testsยืนยัน input immutability, no target leakage, config/plan sync, Test gate, per-variant fit, reload prediction equality, resume incompatibility detection และ artifact provenance

Pilot นี้ลดความเสี่ยงสูงสุดก่อนขยายไป modelอื่น และพิสูจน์ reuse contractจริงโดยไม่เปลี่ยน accepted data pipeline
