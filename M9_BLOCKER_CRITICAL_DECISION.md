# ⚠️ MILESTONE 3 BLOCKER: Target Mismatch + Missing Artifacts

**Date:** Oct 9, 2026 | 06:10 UTC  
**Severity:** 🔴 CRITICAL (blocks M10, affects all classification metrics)  
**Status:** ⏳ AWAITING USER DECISION

---

## 🐛 ISSUES IDENTIFIED (by Sonnet M9 run)

### 1. TARGET MISMATCH ❌

**What we built:**
- M5 classifiers trained on: `y = (target_regression > median)`
- Result: ~50% positive class (balanced)
- Accuracy reported: 0.45–0.62

**What project requires:**
- Frozen label: `target_high_volatility` (Q75 threshold)
- Validation: 9% positive class (severely imbalanced)
- Expected accuracy: ~90% (majority class baseline)

**Impact:**
- M5 model accuracies NOT comparable to project requirements
- Test evaluation would show false metrics
- Violates data contract (frozen Q75 labels only)

### 2. MISSING ARTIFACTS ❌

**With-spike artifacts saved:**
- ✅ MLR, Simple Linear, Polynomial, Elastic Net (regression)
- ✅ Stacking, Naive Bayes, XGBoost, Random Forest, Gradient Boosting, AdaBoost (ensemble)
- ✅ Decision Tree (classification)
- ✅ Perceptron, SLP (classification)
- ❌ Logistic Regression — **NO ARTIFACTS SAVED**

**Non-spike variants:**
- ❌ None exist (never trained)

**Impact:**
- Cannot evaluate 38 models (only ~13 with artifacts exist)
- Logistic has no predictions to validate
- Missing half the deliverables (non_spike)

### 3. MODEL COUNT ❌

**Spec says:** 13 classification models  
**Actual count:** 14 models (Logistic, Stacking, DT, Naive Bayes, XGBoost, RF, GB, Perceptron, SLP, AdaBoost, + k-NN, k-Means, PCA, MLP?)

**Impact:**
- 14 × 2 variants = 28 models, not 38
- Inventory mismatch with requirements

### 4. GATE REQUIREMENTS NOT MET ❌

**M10 requires:**
```
authorized_finalize_test():
  ✅ completed supervised_tuned run
  ✅ frozen hyperparameters
  ✅ frozen non_spike protocol
  ✅ clean git tree
  ✅ manifest tracking
```

**Actual state:**
- ❌ Ad-hoc M3–M8 runs (no manifest)
- ❌ No supervised_tuned completion
- ❌ No frozen non_spike protocol
- ❌ No hyperparameter governance

**Impact:**
- Cannot pass ledger gate for test unlock
- Reading test.jsonl directly would bypass leakage control

---

## 🎯 DECISION REQUIRED

**Option A: HONEST ROUTE (Recommended)**
Redo M5–M8 classification on Q75 target:
1. Retrain all 14 classifiers using `target_high_volatility` (frozen Q75 label)
2. Save artifacts (model + predictions) for with_spike
3. Create non_spike protocol + run all 14 on non_spike data
4. Build manifest for supervised_tuned gate
5. Pass authorization gate properly
6. Evaluate on test set with full traceability

**Effort:** 8–12 hours (parallel Sonnet)  
**Cost:** $8–12 (Sonnet)  
**Result:** Complete, auditable, gate-passing deliverables  
**Timeline:** Oct 10, 18:00–23:00 UTC (tight but possible)  

**Option B: EXPEDITED ROUTE (Risky)**
Direct evaluation on with_spike models:
1. Evaluate existing M5–M8 artifacts on test set directly
2. Document target mismatch clearly
3. Label results as "median-split baseline" (not Q75)
4. Skip gate authorization
5. Accept reduced confidence in metrics

**Effort:** 2–3 hours (just M10)  
**Cost:** $2–3  
**Result:** Quick numbers, but NOT project-compliant  
**Risk:** Metrics misleading, violates data contract  

---

## 📋 RECOMMENDATION

**Use Option A (Honest Route)** because:
- ✅ Respects frozen Q75 label contract
- ✅ Produces audit trail + gate compliance
- ✅ Results meaningful for project submission
- ✅ Sonnet can parallelize + finish in time
- ✅ Budget + timeline still viable

**Option B only if:**
- Deadline collision (it's not)
- Project explicitly allows median-split labels (it doesn't)
- Metrics don't matter (they do)

---

## 🔧 IMPLEMENTATION PLAN (Option A)

### Phase 1: Rebuild M5 on Q75 Target (3–4 hours)

**For each of 14 classifiers:**

```python
# Load with_spike data
dataset = load_train_validation("with_spike")
X_train = dataset.train.X
y_train = dataset.train.target_high_volatility  # Q75 label, 9% positive

X_val = dataset.validation.X
y_val = dataset.validation.target_high_volatility

# Train model
model = Model(**defaults).fit(X_train, y_train)

# Save artifacts
model.to_dict() → model.pkl
train_pred → train_predictions.npy
val_pred → val_predictions.npy
metadata → metadata.json
```

**Models to retrain:**
1. Logistic Regression (M5-01)
2. Stacking (M5-02)
3. Decision Tree (M5-03)
4. Naive Bayes (M5-07)
5. XGBoost (M5-08)
6. Random Forest (M5-09)
7. Gradient Boosting (M5-10)
8. Perceptron (M5-11)
9. SLP (M5-12)
10. AdaBoost (M5-13)
11. k-NN (M5-04)
12. k-Means (M5-05)
13. PCA (M5-06)
14. MLP (if exists, else skip)

**Expected metrics:** Lower accuracy (~9–85%, depending on model), but NOW Q75-compliant

### Phase 2: Build Non-Spike Protocol (1–2 hours)

```python
# Freeze non_spike configuration
protocol = {
  "dataset": "non_spike",
  "target": "target_high_volatility",
  "hyperparameters": {model: defaults},
  "splits": "chronological",
  "validation": "expand-window OOS",
}

# Run all 14 models on non_spike data
for model in models:
  train_val_test_split_non_spike(model, protocol)
  save_artifacts(model, "non_spike")
```

### Phase 3: Pass Gate + Evaluate (1–2 hours)

```python
# Build manifest
manifest = {
  "with_spike": {model: artifact_path for model in models},
  "non_spike": {model: artifact_path for model in models},
  "supervised_tuned": True,
  "hyperparameters_frozen": protocol,
}

# Unlock test
test_data = authorize_finalize_test(manifest)

# Evaluate all 28 models on test
for variant in ["with_spike", "non_spike"]:
  for model in models:
    test_pred = model.predict(test_data.X)
    metrics = compute_metrics(test_pred, test_data.y)
    report.add(metrics)

# Commit + submit
git commit -m "feat(M5-M10): Q75-compliant classification + test eval"
```

---

## ⏱️ TIMELINE (Option A)

```
Oct 9, 18:00 UTC  ⏳ Decision point (now)
Oct 10, 02:00 UTC ⏳ Phase 1 complete (retrain M5)
Oct 10, 04:00 UTC ⏳ Phase 2 complete (non_spike protocol)
Oct 10, 06:00 UTC ⏳ Phase 3 complete (test eval + report)
Oct 10, 07:00 UTC ✅ Commit + ready for submission
Oct 12, 00:00 UTC ✅ Deadline (5 hours buffer)
```

**Feasible?** Yes, if approved now.

---

## ✅ DECISION

**Please confirm:**

1. **Proceed with Option A (Honest Route)?** (Recommended)
2. Or **Proceed with Option B (Direct Evaluation)?** (Risky, non-compliant)
3. Or **Stop M9–M10, submit with M5–M8 as-is?** (Incomplete)

**Blocking:** Cannot commit M9 results until target clarified. Sonnet is ready to execute immediately upon approval.

