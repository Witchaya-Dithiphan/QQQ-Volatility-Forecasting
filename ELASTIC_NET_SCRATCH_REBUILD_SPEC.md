# REGRESSION AUDIT & REBUILD SPEC

## Audit Results

✅ Simple Linear Regression — **SCRATCH** (covariance/variance formulas)
✅ Multiple Linear Regression — **SCRATCH** (normal equation via lstsq)
✅ Polynomial Regression — **SCRATCH** (basis expansion + OLS)
❌ Elastic Net — **SKLEARN WRAPPER** (uses sklearn.linear_model.ElasticNet)

## Required Fix: Elastic Net

### Current Implementation (❌ Rejected)
```python
from sklearn.linear_model import ElasticNet as SKElasticNet
class ElasticNet:
    def fit(self, X, y):
        self._sklearn_model.fit(X, y)  # ← sklearn only, no scratch
```

### Requirement
**Build Elastic Net from scratch using:**
- L1 regularization (Lasso): λ₁||β||₁
- L2 regularization (Ridge): λ₂||β||₂²
- Coordinate descent optimizer (proven stable)
- No sklearn.linear_model dependency

### Specification

**Algorithm: Coordinate Descent for Elastic Net**

```
Objective: minimize ||y - Xβ||² + α(ρ||β||₁ + (1-ρ)||β||₂²)

where:
  α = regularization strength (hyperparameter)
  ρ = l1_ratio (0=Ridge/L2, 1=Lasso/L1, 0.5=Elastic Net mix)

Coordinate Descent Update (per feature j):
  1. Compute residual r = y - X(β excluding β_j)
  2. Update β_j using soft-thresholding:
     β_j := sign(ρ*α) * max(0, |grad_j - ρ*α| / (2*(1-ρ)*α + 2))
  3. Repeat for all features until convergence

Convergence: ||β_new - β_old|| < tol
```

**Implementation Details:**
- Standardize X, y (fit on standardized, predict on original scale)
- Initialize β = 0
- Iterate: for each feature j, apply coordinate descent update
- Max iterations: 1000 (default)
- Convergence tolerance: 1e-4 (default)
- Handle zero variance features (skip or warn)

**Serialization:**
- `to_dict()`: coefficients, intercept, alpha, l1_ratio, feature_mean, feature_std
- `from_dict()`: restore all state

**Tests (existing, must pass):**
```
test_fit_synthetic — synthetic data fit
test_predict_shape — shape match
test_real_data — Train/Validation evaluation with artifact save
```

### Reusable from M2

✅ Data loader (load_train_validation)
✅ Artifact directory structure
✅ Metrics computation
✅ Test patterns
✅ Serialization interface (to_dict/from_dict)

### Output

File: `src/modeling/regression/elastic_net.py`
- ~150–200 lines
- Scratch coordinate descent (no sklearn)
- Compatible with existing tests
- M2 patterns reused (to_dict/from_dict, artifact saving)

### Timeline

Effort: 2–3 hours (Sonnet with medium model)
Budget: $2–3
Confidence: 95% (coordinate descent well-documented, tested in academic literature)

### Acceptance Criteria

✅ No sklearn.linear_model imports
✅ Coordinate descent converges on synthetic test
✅ Predictions match baseline (within 1% tolerance) on real data
✅ to_dict/from_dict roundtrip works
✅ Artifact save successful
✅ All tests pass
