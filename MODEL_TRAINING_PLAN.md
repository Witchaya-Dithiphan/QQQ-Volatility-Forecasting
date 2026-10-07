# QQQ 5-Day Annualized Volatility Forecasting — Model Training Plan

> **สถานะ:** เอกสารแผนหลัก (single source of truth) สำหรับ Modeling Phase  
> **วันที่จัดทำ:** 2026-10-08  
> **Deadline ตาม PDF:** 2026-10-12 — Final presentation และส่งเอกสารทั้งหมด  
> **ขอบเขตรอบนี้:** ตรวจ requirement/repository และวางแผนเท่านั้น ยังไม่มี model implementation, training หรือ performance result

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
| Model dependencies | Missing / decision pending | `requirements.txt` ยังไม่มี `scikit-learn` หรือ `xgboost` |
| Models/metrics/predictions | Missing | ยังไม่มีหลักฐาน model artifacts หรือ quantitative model results |
| Python environment | ตรวจใน audit นี้ | `.venv` ปัจจุบันเป็น Python 3.12.3; หลักฐานตรวจรับเดิมในเอกสารใช้ Python 3.11.9 |

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
| PDF-07 | Explicit PDF requirement | หน้า 2, Classification list | Logistic, Decision Tree, Random Forest, Stacking, GB, XGBoost, Naive Bayes (if any), SVM, dimensionality reduction, k-NN, k-Means, Agglomerative, Perceptron/SLP, MLP และ better model | แยก predictive classification 12 รายการและ clustering 2 รายการ; dimensionality reduction เป็น configuration | Mandatory scope interpretation |
| PDF-08 | Explicit PDF requirement | หน้า 3, new models | อย่างน้อย 1 new regression model และ 1 new classification modelที่ไม่ใช่ตัวเดียวกัน พร้อมอธิบายกลไก, scratch และ ready-to-use | เลือก Elastic Net และ AdaBoost ตาม user instruction; ยังควรยืนยันว่าอาจารย์ยอมรับว่า extracurricular | Mandatory; approval ambiguity |
| PDF-09 | Explicit PDF requirement | หน้า 3, results | Quantitative results, benchmarking, analysis, discussion, Loss, Confusion Matrix, Accuracy, Precision, Sensitivity, Specificity/TNR, F1, ROC/AUC, performance curve และ R² | กำหนด task-specific metrics และ plots; ไม่ใช้ metric ที่ไม่เหมาะกับ task | Mandatory |
| PDF-10 | Explicit PDF requirement | หน้า 5, scoring | Regression และ Classification ทุกโมเดล built from scratch; ready-to-use Scikit-learn/PyTorch ใช้ประกอบเพื่อยืนยันได้ | ห้าม wrapper `.fit()` หรือเขียน metrics เองแล้วเรียก model ว่า scratch | Mandatory; 40 points |
| PDF-11 | Explicit PDF requirement | หน้า 5, scoring | Evaluation มี Accuracy, loss, F1, performance curve ฯลฯ | ต้องมี persisted metrics/figures ที่ trace กลับ config และ predictions ได้ | Mandatory; 30 points |
| PDF-12 | Explicit PDF / administrative | หน้า 4, group/dataset channel | ช่องทางลงข้อมูลสมาชิกและ dataset อยู่ใน Google Classroom/Google Sheet; หน้า 3–4 ยังเตือนห้ามใช้ dataset ซ้ำกับกลุ่มที่เสนอแล้ว | ไม่เปลี่ยน modeling implementation; ทีมต้องยืนยันการลงทะเบียน QQQ นอก repository | Administrative; verify with instructor/classroom |
| USER-01 | User instruction | รอบนี้ | เพิ่ม Elastic Net และ AdaBoost | อยู่ใน 18-item inventory | Mandatory for project scope |
| USER-02 | User instruction | รอบนี้ | ส่ง With-Spike ให้ครบก่อน แล้ว reuse กับ Non-Spike | Milestones ใช้ With-Spike-first; ห้ามอ้าง paired comparison จน Non-Spike เสร็จ | Priority decision |
| USER-03 | User instruction | รอบนี้ | รักษา Phase 1/2 data contract; ห้าม recompute features/targets/Q75 หลังกรอง | Modeling loader ต้องเลือก explicit features และใช้ accepted CSVs แบบ read-only | Mandatory guardrail |
| USER-04 | User instruction | รอบนี้ | สร้าง requirement matrix, reusable framework, scratch/reference table, milestones, artifact/resume design | อยู่ในเอกสารนี้ | Mandatory documentation |
| DEC-01 | Proposed implementation decision | Modeling | Dataset variant เป็น config `with_spike`/`non_spike`; model codeไม่ hard-code variant | ทำให้ reuse ได้และลด divergence | Proposed; freeze before implementation |
| DEC-02 | Proposed implementation decision | Modeling | ใช้ NumPy/Pandas สำหรับ scratch numerical operations และ Scikit-learn/XGBoost เป็น reference เมื่อเหมาะสม | Core learning logic ต้องเขียนเอง | Proposed; dependencies pending |
| DEC-03 | Proposed implementation decision | Evaluation | Train สำหรับ fit, Validation สำหรับ tuning/threshold, Test หลัง freeze เท่านั้น | ลด leakage และทำ paired protocol ได้ | Proposed; should be frozen |
| AMB-01 | Ambiguity | PDF หน้า 2 | `Perceptron & Single-layer perceptron (SLP)` อาจหมายถึงหนึ่งหัวข้อหรือสอง implementation | แผนนี้นับเป็น **หนึ่งรายการหลัก** แต่ควรอธิบายว่า binary perceptron เป็น single-layer unit | ต้องยืนยันหากอาจารย์ต้องการสองรายการ |
| AMB-02 | Ambiguity | PDF หน้า 2 | Naive Bayes มีคำว่า `if any` | รวมไว้ก่อนตาม user instruction; ไม่ตัดเป็น optional | Scope assumption |
| AMB-03 | Ambiguity | PDF หน้า 2 | Dimensionality reduction สำหรับ RF/SVM ไม่ชัดว่าเป็น requirement แยกหรือ preprocessing variant | นับเป็น configuration เพิ่ม เช่น PCA+SVM; ไม่เพิ่ม model count โดยอัตโนมัติ | Scope assumption |
| AMB-04 | Ambiguity | PDF หน้า 2 | Regression bullet เขียน `better and extracurricular classification model` | ใช้หน้า 3 และหน้า 5 เป็นหลักว่า ต้องมี better regression หนึ่งตัวและ better classification หนึ่งตัว | Document inconsistency |
| AMB-05 | Ambiguity | Algorithm fidelity | `XGBoost` ต้องมีองค์ประกอบเฉพาะ ไม่ใช่เรียก generic gradient boosting ว่า XGBoost | Scratch implementation ต้องระบุ regularized second-order boosting, shrinkage และ tree objectiveที่ตรงชื่อ | High-risk implementation |
| AMB-06 | Ambiguity | Clustering | PDF จัด k-Means/Agglomerative ใต้ classification แต่เป็น unsupervised | รายงานแยกเป็น clustering task; ไม่ประเมินเหมือน supervised classifierโดยอัตโนมัติ | Proposed interpretation |

## 3. Model Inventory และสมมติฐานการนับ

### 3.1 รายการหลักเบื้องต้น: 18 รายการ

**Regression — 4**

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Elastic Net — extracurricular candidate

**Classification — 12**

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Stacking Classifier
5. Gradient Boosting
6. XGBoost
7. Naive Bayes
8. Support Vector Machine
9. k-Nearest Neighbors
10. Perceptron / Single-layer Perceptron
11. Multi-layer Perceptron
12. AdaBoost — extracurricular candidate

**Clustering — 2**

1. k-Means
2. Agglomerative Clustering

### 3.2 สิ่งที่เลข 18 ไม่ได้หมายถึง

- PDF ไม่ได้พิมพ์ว่า “ต้องมี 18 โมเดล” โดยตรง; 18 เป็นผลการตีความ inventory ข้างต้น
- 18 ไม่รวม reference implementation เป็นโมเดลหลักใหม่
- 18 ไม่รวม With-Spike/Non-Spike เป็นคนละ algorithm
- 18 ไม่รวม PCA/no-PCA, raw/clipped predictions, threshold choices หรือ hyperparameter trials เป็นโมเดลใหม่
- ถ้าทำ 18 algorithms × 2 dataset variants จะได้ **36 primary scratch training runs** เป็นอย่างน้อย แต่จำนวน artifact runs จริงมากกว่านี้เมื่อรวม reference, tuning, seed stability, failure/retry และ save/load verification
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
notebooks/05_clustering.ipynb   # proposed only if time permits/needed
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
| Polynomial Regression | สร้าง polynomial basis แล้ว OLS | ระบุ single-feature หรือ multivariate interactions; degreeเลือกด้วย Validation; term-count guard | `PolynomialFeatures` + `LinearRegression` | ตรวจ basis/term orderและ predictions; เสี่ยง term explosion |
| Elastic Net | MSE + L1/L2; coordinate descent/proximal update พร้อม convergence | standardization; `alpha`, `l1_ratio`, max iterations, tolerance | `sklearn.linear_model.ElasticNet` | objective decreases/KKT-style checks; compare predictions/objectiveภายใต้ scalingเดียวกัน |
| Logistic Regression | Binary cross-entropy + gradient update | standardization, learning rate, regularization, class weight, tolerance | `sklearn.linear_model.LogisticRegression` | gradient sanity, probability range, separable/non-separable fixtures |
| Decision Tree | Recursive splitเลือก impurity reduction; stopping/pruning policy | max depth, min samples, criterion, feature tie-break | `DecisionTreeClassifier` | hand-checkable splits; deterministic tie policy; risk medium |
| Random Forest | Bootstrap trees + random feature subsets + vote/probability aggregation | trees, depth, max features, bootstrap, seed | `RandomForestClassifier` | bootstrap/feature-subset tests; deterministic seed; runtime risk |
| Stacking Classifier | Train base learnersบน chronological OOS folds; fit meta-learnerบน OOS scores | base set, fold boundaries/purge, meta model | `StackingClassifier` ใช้เป็น conceptual/reference แต่ต้อง custom chronology | leakage riskสูง; assertทุก meta-featureมาจากอดีต-only fold |
| Gradient Boosting | Sequential additive trees fit negative gradients/residuals | estimators, depth, learning rate, loss | `GradientBoostingClassifier` | loss/history decreasesบน fixtures; distinguishจาก AdaBoost/XGBoost |
| XGBoost | Regularized second-order boosting; gradients/Hessians, gain, leaf weights, shrinkage | depth, learning rate, lambda/gamma, bins/split search, estimators | `xgboost.XGBClassifier` หากเพิ่ม dependency | ความเสี่ยงสูงมาก; ห้ามใช้ generic GB แล้วเปลี่ยนชื่อ; scopeต้อง freeze |
| Naive Bayes | Gaussian class priors/means/variances และ log-posterior | variance smoothing | `GaussianNB` | closed-form fixture/log-space checks; one-class/zero variance policy |
| Support Vector Machine | Linear soft-margin hinge lossด้วย subgradient/SMO-liteตาม scopeที่ freeze | scaling, `C`, kernel decision; PCA variant | `SVC`/`LinearSVC` | ความเสี่ยงสูงถ้า kernel SVMเต็มรูปแบบ; scoreไม่เรียก probability |
| k-Nearest Neighbors | Distance computation, neighbor selection, voting/tie rule | scaling, `k`, metric, weights | `KNeighborsClassifier` | exact toy distances/ties; runtime/memory riskต่ำ-กลาง |
| Perceptron / SLP | Linear score, step activation, mistake-driven update | scaling, learning rate, epochs, shuffle policy/seed | `Perceptron` | linearly separable fixture; scoreเป็น decision scoreไม่ใช่ probability |
| Multi-layer Perceptron | Forward pass, activations, BCE, backpropagation, optimizer | hidden sizes, activation, learning rate, epochs, early stopping | `MLPClassifier` หรือ PyTorch reference | gradient checks; chronological Validation early stopping; convergence riskสูง |
| AdaBoost | Weighted weak learners, weighted error, sample-weight update, learner weights, vote | stumps, estimators, learning rate; freeze SAMME-like binary variant | `AdaBoostClassifier` | toy weight updates; noisy-label sensitivity; define failureเมื่อ error ≥ 0.5 |
| k-Means | Initialize centroids, assign, recompute, inertia, convergence | scaling, `k`, init, restarts, seed | `KMeans` | toy clusters/inertia monotonicity; label permutation-aware comparison |
| Agglomerative Clustering | Pairwise distancesและ iterative linkage merges | scaling, linkage, metric, cluster count | `AgglomerativeClustering` | verify merge sequence/dendrogram fixture; O(n²) risk; ไม่มี standard out-of-sample `predict` |

### Dimensionality reduction

เสนอ `PCA` แบบ scratch (center data, covariance/SVD/eigendecomposition, explained variance, transform) เป็น preprocessing configuration สำหรับอย่างน้อย SVM และ Random Forest ตามข้อความ PDF โดย:

- fit PCA จาก Train ของ variantนั้นเท่านั้น
- เลือกจำนวน componentsด้วย Validation เท่านั้น
- fit PCA ใหม่สำหรับ Non-Spike
- ไม่เพิ่ม PCA เป็น predictive model itemลำดับที่ 19 โดยอัตโนมัติ

## 7. Leakage-Safe Training Protocol

1. โหลด variant Train ที่เลือกและ Original Validation/Test แบบ read-only
2. Validate schema, explicit feature order, targets, finite values, dates และ checksums
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
- **MLP:** early stoppingดู Validationตาม protocol; ห้ามสุ่มแบ่งอนาคตเข้า Train
- **Class weighting/resampling:** ทำเฉพาะ Train; Validation/Test distributionคงเดิม
- **Simple Linear:** feature choiceใช้ Validation ไม่ใช้ Test
- **Polynomial:** degree/interaction scopeใช้ Validation และมี term-count guard
- **PCA:** component countและ fitห้ามแตะ Validation/Testในการคำนวณ basis
- **Random models:** seedคงที่สำหรับ reproducibility; multi-seed stabilityเป็น analysisเสริม ไม่แทน primary run

## 8. Evaluation และ Presentation Contract

### 8.1 Regression

ผลหลัก:

- MSE/Loss, MAE, RMSE, R²
- Train-mean baseline
- Historical-volatility persistence baseline เช่นใช้ `historical_volatility_5d` เป็น prediction โดยต้องประกาศสูตรล่วงหน้า
- Actual vs predicted, residual plot/distribution, prediction CSVจับคู่ `Date`
- รายงานหน่วย targetเป็น decimal annualized volatility; ถ้าแสดง percentให้ระบุว่าเป็น percentage pointsอย่างถูกต้อง
- Negative prediction policyต้อง freezeก่อน Test; ถ้า clipที่ 0 ให้เก็บทั้ง raw และ clipped metrics/predictions

### 8.2 Classification

ผลหลัก:

- Confusion Matrix, Accuracy, Precision, Recall/Sensitivity, Specificity/TNR, F1
- ROC curve และ ROC-AUC เมื่อมี continuous score
- PR curve และ Average Precision เป็น metricเสริมสำหรับ imbalance
- Majority-class baseline
- zero-division และ one-class subset policyเป็น `N/A` พร้อมเหตุผล ไม่ใส่ค่าปลอม
- Perceptron/SVM ใช้ decision scoreสำหรับ ranking metrics; ไม่เรียก scoreว่า probability

### 8.3 Clustering

- Internal metrics เช่น inertia (k-Means), Silhouette score และ cluster size stability
- เลือกจำนวน clustersจาก Train analysis ไม่ใช้ Test labels
- ถ้าจะเทียบ clusterกับ `target_high_volatility` ต้องสร้าง mappingจาก Trainเท่านั้นและระบุว่าเป็น post-hoc analysis
- Agglomerative ไม่มี standard out-of-sample predict: primary evaluationควรเป็น Train structure/stability หรือเสนอ frozen extensionเช่น nearest-centroid assignmentโดยประกาศว่าไม่ใช่ standard algorithm; ห้าม fitใหม่บน Testแล้วเรียกว่า forecast
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
  metrics_test.json
  predictions_validation.csv
  predictions_test.csv
  figures/
  run_status.json
```

แต่ละ runต้องเก็บ:

- variant, task, model, scratch/reference และ run/config ID
- feature names/order, target definition, preprocessing และ selected hyperparameters
- Train/Validation/Test paths, SHA-256, rows/date ranges
- seeds, dependency versions, Python version, git revision และ dirty status
- fitted model/preprocessor หรือ state arraysที่ loadกลับได้
- predictionsจับคู่ `Date`
- metrics, convergence/failure/skipped status, timing และ warnings
- strict JSON (`allow_nan=False`) และ portable relative paths

Resume ต้องตรวจ config hash, input hashes, code revision, artifact completenessและ load test; ห้าม skipเพียงเพราะ directoryมีอยู่ รองรับรันทีละ model และสถานะ `completed`, `failed`, `skipped`, `incompatible`, `running`

CLI ตัวอย่างต่อไปนี้เป็น **planned interface ยังรันไม่ได้**:

```text
python -m src.modeling.runner train --variant with_spike --task regression --model elastic_net --implementation scratch --seed 41 --search-budget 12 --output-root outputs/modeling
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
- **Checklist:** matrix, 18-item inventory, With-Spike-first, mandatory/optional, open decisions
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

- **Goal:** พิสูจน์ frameworkกับหนึ่งโมเดลง่ายทั้งสอง variantsก่อนเปิด Test โดยไม่ถือว่าเริ่ม Non-Spike delivery phase
- **Dependencies:** M2
- **Files:** Multiple Linear Regression scratch/reference + tests
- **Checklist:** fresh scaler/modelต่อ variant, ใช้ Train/Validation เท่านั้น, save/load reload equality; ไม่เปิด Test และไม่ใช้ pilotเลือก protocolจาก variantที่ได้คะแนนดีกว่า
- **Verification:** synthetic fixture + artifact completeness; pilotมีไว้พิสูจน์ code reuse ไม่ใช่ผลส่งหรือ paired conclusion
- **DoD:** ทั้งสอง variantsรันผ่าน plumbingด้วย configต่างเฉพาะ data provenance จากนั้นหยุด Non-Spikeไว้จน M9 และกลับไปทำ With-Spike inventory M4–M8 ให้ครบ
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

- **Goal:** Logistic, Naive Bayes, k-NN, Perceptron/SLP, Decision Tree
- **Dependencies:** M2
- **Checklist:** scores/probabilitiesถูกประเภท, imbalance/threshold policy, confusion metrics
- **Verification:** toy fixtures/reference comparisons
- **DoD:** 5 modelsมี scratch/referenceและ required metrics
- **Priority/Risk/Effort:** P0 / กลาง / 18–30h

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

ณ 2026-10-08 เหลือประมาณ 4 วันปฏิทินก่อน deadline วันที่ 2026-10-12 แผนเต็ม 18 scratch algorithms + reference + tuning + save/load + report/slides มี effortมากกว่าช่วงเวลาที่เหลืออย่างชัดเจน โดยเฉพาะ XGBoost, SVM, MLP, Stacking และ Agglomerative ดังนั้นห้ามรับประกันว่าจะเสร็จครบโดยไม่มีทีม/implementationเดิมเพิ่มเติม

ลำดับ critical path:

1. Freeze ambiguitiesและ algorithm fidelityทันที โดยเฉพาะ Perceptron/SLP, PCA requirement, SVM kernel, XGBoost scope และ clustering evaluation
2. สร้าง M2 infrastructureและ M3 Train/Validation-only pilotก่อน เพื่อพิสูจน์ reuse โดยไม่เปิด Testหรือเริ่ม Non-Spike submission work แล้วหยุด Non-Spikeไว้จนหลัง M8
3. ทำ With-Spike scratch modelsตามลำดับ: regression → basic classification → advanced/high-risk → clustering
4. ทำ reference comparisons, save/load testsและเอกสารควบคู่ ไม่รอท้ายสุด
5. Freeze protocolก่อน Test; สร้าง With-Spike submission package
6. หลัง With-Spike frameworkนิ่งจึง rerun Non-Spikeด้วย fresh fits
7. ทำ paired comparisonเฉพาะเมื่อทั้งสอง variantsครบจริง

ถ้าเวลาขาด ให้ลด **optional enhancements** ก่อน: broad hyperparameter grids, multi-seed stability, detector sensitivity variants, interactive dashboards, extensive ablations และ cosmetic plots ห้ามย้าย modelที่ตีความว่า PDFบังคับไปเป็น optionalเพียงเพราะทำยาก

## 12. Mandatory vs Optional

### จำเป็นเพื่อส่งตาม requirementที่ตีความ

- 18-item inventoryตามสมมติฐานที่ระบุ หรือคำยืนยันอาจารย์ที่แก้ scope
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
| เวลาไม่พอสำหรับ 18 scratch modelsก่อน 2026-10-12 | ส่งไม่ครบหรือ verificationตื้น | ทำ critical path, จำกัด search budget, ลด optional workก่อน; รายงานสถานะจริง | ทีมโครงการ / ปิด scopeทันที |
| XGBoost/SVM/MLP/Stacking fidelityไม่ตรงชื่อ | เสียคะแนน from-scratch/algorithm mechanism | freeze minimum faithful algorithm, synthetic testsและ reference comparisonก่อน full data | model implementer / ก่อนเริ่มแต่ละ model |
| `.venv` 3.12.3 ต่างจาก recorded 3.11.9 และ dependenciesไม่ pin | reproduceไม่ได้หรือ library incompatibility | freeze Python/dependency versions, สร้าง environment manifestและ smoke test | infrastructure owner / M2 |
| Elastic Net/AdaBoostไม่ถูกยอมรับเป็น extracurricular | ไม่ผ่าน requirement new-model | ขอคำยืนยันอาจารย์; เตรียมเหตุผลและ candidateสำรองโดยไม่เปลี่ยน Test protocol | team lead / ก่อน final scope freeze |
| Test feedbackจาก With-Spikeปนการออกแบบ Non-Spike | paired comparisonมี leakage/bias | freeze Non-Spike protocol/search spacesก่อนเปิด With-Spike Test; log test-access timestamp | experiment owner / ก่อน M8 Test |
| Stacking purgeบน Non-Spikeใช้ row gapแทน original timeline | future leakage | foldด้วย original dates/positionsและ explicit purge tests | stacking owner / M6 |
| Broad tuning/computeเกินเวลา | ไม่มี artifactsครบหรือ rerunไม่ได้ | fixed small budgets, per-model timeout, resume/status artifacts | experiment owner / M2 ก่อน batch runs |
| Agglomerativeไม่มี standard OOS predict | อ้าง forecastไม่ถูกต้อง | จำกัด primary evaluationเป็น Train structure/stability หรือประกาศ extensionชัดเจน | clustering owner / M7 |

## 14. Open Decisions ที่ต้องปิดก่อน Implementation

1. อาจารย์นับ Perceptronและ SLPเป็นหนึ่งหรือสอง deliverables
2. PCA ต้องมีทั้ง RFและ SVMหรือเพียง demonstrationอย่างน้อยหนึ่ง configuration
3. SVM scratch scope: linear soft-marginเป็นขั้นต่ำที่ยอมรับได้หรือจำเป็นต้องมี kernel
4. XGBoost scratch fidelityขั้นต่ำและ reference dependency `xgboost`
5. Stacking base learners/meta-learnerชุดใดและ chronological fold/purge schedule
6. AdaBoost variant/base learner; แผนนี้เสนอ binary SAMME-style decision stumps
7. Polynomial scope: single featureหรือ multivariate interactions; แผนเริ่มจาก single featureเลือกด้วย Validationเพื่อลด term explosion
8. Clusteringควรใช้ featuresทั้งหมดหรือ standardized subset และต้องมี notebookแยกหรือรวม classification notebook
9. Elastic Net/AdaBoostได้รับการยอมรับเป็น extracurricular modelsหรือไม่
10. Dependency/Python versionสำหรับ submission: ทำ environmentให้ reproducibleและแก้ความต่างระหว่าง current 3.12.3กับ recorded 3.11.9

## 15. งานแรกสำหรับรอบ Implementation ถัดไป

เริ่ม **M2 + M3 pilot** โดยยังไม่กระจาย implement 18 models:

1. เพิ่ม `src/modeling/contracts.py` และ `datasets.py` เพื่อโหลด/validateสอง variantsแบบ read-onlyและเลือก `FEATURE_COLUMNS`อย่าง explicit
2. เพิ่ม standardizer scratch, metric primitives, artifact manifest, strict JSON และ save/load compatibility check
3. Implement Multiple Linear Regression scratch + Scikit-learn referenceเป็น pilot
4. รัน Train/Validation-only pilotบน With-Spikeและ Non-Spikeด้วย fresh fitทุกส่วนเพื่อพิสูจน์ plumbing; ไม่เปิด Test ไม่สรุป paired result และหยุด Non-Spikeไว้จนหลัง With-Spike submission milestone
5. เขียน testsยืนยัน input immutability, no target leakage, per-variant fit, reload prediction equality และ artifact provenance

Pilot นี้ลดความเสี่ยงสูงสุดก่อนขยายไป modelอื่น และพิสูจน์ reuse contractจริงโดยไม่เปลี่ยน accepted data pipeline
