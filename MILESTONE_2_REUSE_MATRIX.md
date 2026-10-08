# 🔄 REUSE MATRIX: Milestone 2 (M5–M8) for M9–M10 Q75 Fix

**Date:** Oct 9, 2026  
**Purpose:** Identify what can be reused from M5–M8 implementations for retaining on Q75 target  
**Status:** ✅ HIGH REUSE POTENTIAL (70–80%)

---

## 📊 WHAT CAN BE REUSED

### ✅ ENSEMBLE MODELS (100% Reusable)

| Model | File | Implementation | Reuse Strategy |
|-------|------|----------------|-----------------|
| **Stacking** | `ensemble/stacking.py` | ✅ Scratch, robust | Retrain on Q75 target (same algorithm) |
| **XGBoost** | `ensemble/xgboost.py` | ✅ Scratch, proven | Retrain on Q75 (already M5-08 baseline) |
| **Random Forest** | `ensemble/random_forest.py` | ✅ Wrapper ready | Retrain on Q75 (sklearn, no changes) |
| **Gradient Boosting** | `ensemble/gradient_boosting.py` | ✅ Wrapper ready | Retrain on Q75 (sklearn, no changes) |
| **AdaBoost** | `ensemble/adaboost.py` | ✅ Wrapper ready | Retrain on Q75 (sklearn, no changes) |

**Effort to retrain:** 1 hour per model × 5 = 5 hours total (parallel: 1 hour)  
**Code changes:** ZERO — just fit() on Q75 target  
**Confidence:** 🟢 100%

---

### ✅ CLASSIFICATION MODELS (100% Reusable)

| Model | File | Implementation | Reuse Strategy |
|-------|------|----------------|-----------------|
| **Logistic Regression** | `classification/logistic.py` | ✅ Scratch GD | Retrain on Q75 (fix missing artifacts) |
| **Decision Tree** | `classification/decision_tree.py` | ✅ Scratch Gini | Retrain on Q75 (same algorithm) |
| **Naive Bayes** | `classification/naive_bayes.py` | ✅ Scratch Gaussian | Retrain on Q75 (same algorithm) |
| **Perceptron** | `classification/perceptron.py` | ✅ Scratch simple rule | Retrain on Q75 (same algorithm) |
| **SLP** | `classification/slp.py` | ✅ Scratch tanh+MSE | Retrain on Q75 (same algorithm) |
| **k-NN** | `classification/knn.py` | ✅ Scratch Euclidean | Retrain on Q75 (no hyperparams to tune) |

**Effort to retrain:** 1–2 hours per model × 6 = 6–12 hours total (parallel: 2 hours)  
**Code changes:** ZERO — just fit() on Q75 target  
**Confidence:** 🟢 100%

---

### ✅ INFRASTRUCTURE (100% Reusable)

| Component | File | Current Use | Reuse |
|-----------|------|------------|-------|
| **SklearnEnsemble wrapper** | `ensemble/sklearn_wrapper.py` | RF, GB, Ada base | Reuse as-is ✅ |
| **TDD patterns** | `tests/modeling/` | 225 tests | Reuse test structure ✅ |
| **Artifact saving** | `test_*_real_data()` | Predictions + metadata | Reuse as-is ✅ |
| **Serialization** | `to_dict()/from_dict()` | All models | Reuse as-is ✅ |
| **Data loader** | `src/modeling/datasets.py` | With-spike/non-spike | Reuse as-is ✅ |

**Effort to adapt:** ZERO  
**Confidence:** 🟢 100%

---

## 🎯 RETRAIN PLAN (Reusing M5–M8)

### Phase 1: Quick Retrain (2–3 hours, Sonnet parallel)

```bash
# For each of 11 models (5 ensemble + 6 classification):

# Load Q75 target (not median split)
dataset = load_train_validation("with_spike")
X_train, y_train = dataset.train.X, dataset.train.target_high_volatility

# Use EXISTING model class (no changes!)
model = Logistic()  # or RF, DT, NB, etc.
model.fit(X_train, y_train)

# Use EXISTING artifact saving (no changes!)
model.to_dict() → artifacts/
train_predictions.npy
val_predictions.npy
metadata.json
```

**Code reused:**
- ✅ Model class (100%)
- ✅ Artifact structure (100%)
- ✅ Test patterns (100%)
- ✅ Data loader (100%)

**What changes:**
- ❌ Only the target variable (median → Q75)

### Phase 2: Non-Spike Protocol (1 hour)

```bash
# Same retraining, but on non_spike data
for model in [Logistic, RF, DT, NB, ...]:
  X_train, y_train = load_non_spike_data()
  model.fit(X_train, y_train)
  save_artifacts(model, "non_spike")
```

---

## 💾 ARTIFACTS REUSED

### Test Fixture Functions (100% reusable)
```python
def _synthetic():
    """Generate synthetic data — REUSE from M5"""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)

# Used in:
# - test_fit_synthetic (RED→GREEN)
# - test_predict_shape
# - test_predict_proba
```

### Real Data Test Pattern (100% reusable)
```python
def test_real_data():
    """PATTERN REUSED for Q75 target"""
    dataset = load_train_validation("with_spike")
    X_train, y_train = dataset.train.X, dataset.train.target_high_volatility  # ← only change
    
    model = Model().fit(X_train, y_train)
    
    # Save artifacts (EXACT SAME CODE)
    out_dir = Path("outputs/modeling/with_spike/...")
    np.save(out_dir / "train_predictions.npy", model.predict(X_train))
    np.save(out_dir / "val_predictions.npy", model.predict(val_X))
```

---

## ✅ REUSE SUMMARY

| Artifact | Reusable? | Effort | Confidence |
|----------|-----------|--------|------------|
| Model implementations | ✅ 100% | 0 hours | 🟢 100% |
| Test structure | ✅ 100% | 0 hours | 🟢 100% |
| Artifact saving | ✅ 100% | 0 hours | 🟢 100% |
| Data loading | ✅ 100% | 0 hours | 🟢 100% |
| Serialization | ✅ 100% | 0 hours | 🟢 100% |
| Git patterns | ✅ 100% | 0 hours | 🟢 100% |

**Total code changes needed:** ZERO lines (just retrain!)  
**Total reuse:** 🟢 **70–80% of Milestone 2 work**

---

## 🚀 EXECUTION PLAN (Using M5–M8 Reuse)

```bash
# Step 1: Retrain all 11 models on Q75 target (parallel Sonnet, 2–3 hours)
for model in [Logistic, RF, DT, NB, Perceptron, SLP, XGBoost, Ada, Stacking, GB, kNN]:
  model.fit(X_train_q75, y_train_q75)  # ← only change from M5
  save_artifacts()  # ← EXACT same code as M5

# Step 2: Run non_spike retrain (1 hour, same pattern)
for model in models:
  model.fit(X_train_non_spike_q75, y_train_non_spike_q75)
  save_artifacts("non_spike")

# Step 3: Pass gate + evaluate test (1–2 hours)
manifest = build_manifest(models, artifacts)
test_eval = evaluate_all_models(test_set)

# Step 4: Commit (using M5–M8 patterns)
git commit -m "feat(M5-M10): Q75-compliant retrain (reusing M5-M8 code, 100% compatible)"
```

---

## 📊 COST/TIME SAVINGS

**Without Reuse (from scratch):**
- Rewrite 11 model classes: 8–12 hours
- Rewrite tests: 2–3 hours
- Rewrite artifact saving: 1–2 hours
- Total: 12–18 hours

**With Reuse (M5–M8):**
- Retrain only (code unchanged): 3–4 hours
- New tests: 0 hours (reuse patterns)
- Artifact saving: 0 hours (reuse code)
- Total: **3–4 hours** ✅

**Savings:** 9–14 hours (75% less work!)  
**Cost savings:** $9–14 (Sonnet budget)  

---

## ✅ RECOMMENDATION

**Use M5–M8 models as-is:**
1. ✅ Retrain on Q75 target (fit() only)
2. ✅ Reuse all test patterns (no changes)
3. ✅ Reuse artifact saving (copy-paste)
4. ✅ Reuse data loaders (no changes)
5. ✅ Reuse git patterns (same commit style)

**Result:**
- Complete reuse of 70–80% of Milestone 2 work
- 9–14 hours saved
- 3–4 hours total for M9–M10 fix
- No code duplication
- Full compliance with Q75 target

**Confidence:** 🟢 **98%** (proven code, just different target)

