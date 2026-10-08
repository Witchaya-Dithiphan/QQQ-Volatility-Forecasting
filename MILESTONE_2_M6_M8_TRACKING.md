# MILESTONE 2: M6–M8 DETAILED TRACKING

**Status:** 🚀 **IMPLEMENTATION STARTED**  
**Date:** Oct 9, 2026 04:55 UTC  
**Process:** proc_612ff5a83380 (Sonnet, async)  
**Budget:** $12 USD max  

---

## 📋 TASK BREAKDOWN & TRACKING

### TASK 1: M5-09 Random Forest Classifier ⏳ IN PROGRESS

**Assigned to:** Claude Sonnet  
**Mode:** Full implementation (plan approved)  
**Duration:** 2–3 hours  
**Status:** ⏳ STARTED

#### Spec:
```
File: src/modeling/ensemble/random_forest.py
Tests: tests/modeling/test_ensemble_random_forest.py (4 tests)
Model: sklearn.ensemble.RandomForestClassifier (wrapper)
Artifacts: outputs/modeling/with_spike/ensemble/random_forest/scratch/M5_09/
```

#### Tests (in order):
- [ ] test_fit_synthetic (RED → GREEN)
- [ ] test_predict_shape
- [ ] test_predict_proba (sum to 1, [0,1] range)
- [ ] test_load_train_validation_real_data (with artifact capture)

#### Deliverables:
- [ ] Implementation file (`random_forest.py`)
- [ ] Test file (4 tests, all pass)
- [ ] Artifacts directory + files
- [ ] Commit to main

#### Acceptance:
- [x] Plan approved
- [ ] All 4 tests passing
- [ ] Artifacts saved
- [ ] Committed
- [ ] Ready for Task 2

---

### TASK 2: M5-10 Gradient Boosting ⏳ QUEUED

**Assigned to:** Claude Sonnet  
**Blocker:** Task 1 complete  
**Duration:** 2–3 hours  

#### Spec:
```
File: src/modeling/ensemble/gradient_boosting.py
Tests: tests/modeling/test_ensemble_gradient_boosting.py (4 tests)
Model: sklearn.ensemble.GradientBoostingClassifier
Artifacts: outputs/modeling/with_spike/ensemble/gradient_boosting/scratch/M5_10/
```

#### Tests (same pattern as RF):
- [ ] test_fit_synthetic
- [ ] test_predict_shape
- [ ] test_predict_proba
- [ ] test_load_train_validation_real_data

#### Status:
- [ ] Awaiting Task 1 completion

---

### TASK 3: M5-11 Perceptron ⏳ QUEUED

**Assigned to:** Claude Sonnet  
**Blocker:** Task 2 complete  
**Duration:** 1–2 hours  
**Implementation:** Scratch (simple weight update: w += η * (y - ŷ) * x)

#### Spec:
```
File: src/modeling/classification/perceptron.py
Tests: tests/modeling/test_classification_perceptron.py (5 tests)
Model: Scratch simple perceptron (binary classification only)
Artifacts: outputs/modeling/with_spike/classification/perceptron/scratch/M5_11/
```

#### Tests:
- [ ] test_fit_synthetic (RED)
- [ ] test_predict_shape
- [ ] test_converge_linearly_separable (loss decreases)
- [ ] test_predict_proba (binary threshold [0,1])
- [ ] test_load_train_validation_real_data

#### Status:
- [ ] Awaiting Task 2 completion

---

### TASK 4: M5-12 Single-Layer Perceptron (SLP) ⏳ QUEUED

**Assigned to:** Claude Sonnet  
**Blocker:** Task 3 complete  
**Duration:** 1–2 hours  
**Implementation:** Scratch (tanh activation, MSE loss, ±1 targets — different from Logistic Regression)

#### Spec:
```
File: src/modeling/classification/slp.py
Tests: tests/modeling/test_classification_slp.py (5 tests)
Model: Scratch SLP (tanh + MSE, not sigmoid + CE)
Artifacts: outputs/modeling/with_spike/classification/slp/scratch/M5_12/
```

#### Notes:
- Different from M5-01 (Logistic): tanh + MSE vs sigmoid + cross-entropy
- ±1 targets (not 0/1)
- Still gradient descent

#### Tests:
- [ ] test_fit_synthetic
- [ ] test_predict_shape
- [ ] test_converge_loss_decrease
- [ ] test_predict_proba
- [ ] test_load_train_validation_real_data

#### Status:
- [ ] Awaiting Task 3 completion

---

### TASK 5: M5-13 AdaBoost ⏳ QUEUED

**Assigned to:** Claude Sonnet  
**Blocker:** Task 4 complete  
**Duration:** 2–3 hours  

#### Spec:
```
File: src/modeling/ensemble/adaboost.py
Tests: tests/modeling/test_ensemble_adaboost.py (5 tests)
Model: sklearn.ensemble.AdaBoostClassifier (wrapper)
Artifacts: outputs/modeling/with_spike/ensemble/adaboost/scratch/M5_13/
```

#### Tests:
- [ ] test_fit_synthetic
- [ ] test_predict_shape
- [ ] test_predict_proba
- [ ] test_boosting_effect (accuracy increases per iteration)
- [ ] test_load_train_validation_real_data

#### Status:
- [ ] Awaiting Task 4 completion

---

## 🎯 SONNET'S NOTES (Approved):

1. **Existing wrappers in ensemble_models.py** → Will leave alone
2. **Random Forest test rewrite** → Total tests ~223 (not 227, but acceptable)
3. **SLP activation** → Use tanh + MSE (different from Logistic)
4. **.gitignore check** → Will verify `outputs/` excluded
5. **Real data tests** → No accuracy bar, just check non-NaN + shape
6. **Commits** → One per task to main (`feat(M5-XX): ...`)
7. **Artifacts** → Train/val predictions + metadata JSON per model

---

## 📊 EXPECTED RESULTS

### Tests:
```
Current:  204 tests
Task 1:   +4 tests (Random Forest)
Task 2:   +4 tests (Gradient Boosting)
Task 3:   +5 tests (Perceptron)
Task 4:   +5 tests (SLP)
Task 5:   +5 tests (AdaBoost)
────────────────────
Expected: 227 tests ✅ (target: 200+)
```

### Cost:
```
Budget:   $12 USD max
Estimate: ~$6–8 (Tasks 1–5)
Remaining: ~$4 USD
```

### Timeline:
```
Task 1: 2–3 hours  (start now)
Task 2: 2–3 hours  (after Task 1)
Task 3: 1–2 hours  (after Task 2)
Task 4: 1–2 hours  (after Task 3)
Task 5: 2–3 hours  (after Task 4)
────────────────────
Total:  9–15 hours
ETA:    Oct 10, 13:00–19:00 UTC
```

---

## ✅ APPROVAL & GO-AHEAD

**Sonnet's plan:** ✅ APPROVED  
**Mode:** Full implementation (not plan mode)  
**Start:** M5-09 Random Forest  
**Sequencing:** Sequential (Task 1 → 2 → 3 → 4 → 5)  
**Budget:** Approved up to $12 USD  
**Commits:** Direct to main, one per task  

**Status:** 🚀 **GO AHEAD CONFIRMED**

---

## 📝 CHECKPOINTS

After Sonnet reports **TASK 1 COMPLETE**:
- [ ] Review test results (4/4 pass)
- [ ] Verify artifacts saved
- [ ] Check commit hash
- [ ] Approve Task 2 start

After Sonnet reports **TASK 2 COMPLETE**:
- [ ] Review test results (4/4 pass)
- [ ] Verify artifacts saved
- [ ] Check commit hash
- [ ] Approve Task 3 start

... (repeat for Tasks 3–5)

After Sonnet reports **TASK 5 COMPLETE**:
- [ ] Full test suite: `pytest tests/modeling -q`
- [ ] Expected: 227/227 pass (or ~223)
- [ ] Artifacts: 5 new directories (RF, GB, Perceptron, SLP, AdaBoost)
- [ ] Commits: 5 new commits to main
- [ ] **MILESTONE 2 COMPLETE** → Proceed to M9–M10

---

## 🎯 WHAT'S NEXT AFTER M2

### MILESTONE 3: M9–M10 Finalization
- M9: Stability checks (random seed variations, cross-validation)
- M10: Test split unlock + final metrics on held-out data
- Estimated: 5–6 hours
- ETA: Oct 10, 18:00–00:00 UTC

### Final Deliverables:
```
✅ 38 models trained (19 algorithms × 2 variants: with/non-spike)
✅ 227+ tests passing
✅ All artifacts saved (predictions + metadata)
✅ Ready for production deployment
```

---

## 📞 CONTACT POINTS

- **Sonnet process:** proc_612ff5a83380
- **Log file:** /tmp/m6_m8_sonnet_dispatch.log
- **Spec file:** /tmp/m5_m8_sonnet_spec.txt
- **Monitor:** Check back in ~4 hours for Task 1 completion

**Status:** ✅ **EVERYTHING SET, SONNET RUNNING**

