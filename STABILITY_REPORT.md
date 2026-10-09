# M9 Stability Report

**Scope:** 14 models (the prompt says 13 but lists 14), `with_spike` variant only. Train/Validation only — the Test split was **not** read; the 'test accuracy' in the shuffle check is validation.

**Method:** seeds 42/123/999 on full train → validation predictions; 5 contiguous (time-ordered) CV folds on train with a 5-row purge around the held-out block (the 5-day target overlaps rows); 3 row-shuffles of train → validation metric vs unshuffled. Regression metric = MSE (shuffle Δ = relative %); classification metric = accuracy (shuffle Δ = percentage points). Seed variance = % of validation predictions that differ from seed 42 (classification) or mean relative abs change (regression). Code: `src/modeling/stability.py`; raw numbers: `outputs/reports/stability_results.json`.

**Criteria as written:** seed <2%, CV std <0.05 *of mean* (read as std/mean <5%), shuffle within ±2%. Unseeded models (no random component) trivially show 0% seed variance.

## Regression
| Model | Seed variance (<2%) | CV std (<5% of mean) | Shuffle Δ (≤2) | Status |
|---|---|---|---|---|
| MLR | n/a (deterministic) | 0.0083 (76.9% of mean) | 0.00% | ❌ fails: CV |
| Simple Linear | n/a (deterministic) | 0.0167 (76.5% of mean) | 0.00% | ❌ fails: CV |
| Polynomial | n/a (deterministic) | 0.1844 (202.4% of mean) | 0.00% | ❌ fails: CV |
| Elastic Net | n/a (deterministic) | 0.0157 (98.3% of mean) | 0.00% | ❌ fails: CV |

## Classification — target the saved M5–M8 models used (train-median split of `target_volatility_5d`)
| Model | Seed variance (<2%) | CV std (<5% of mean) | Shuffle Δ (≤2) | Status |
|---|---|---|---|---|
| Logistic | 100.00% | 0.1633 (41.3% of mean) | 0.00 pp | ❌ fails: seed, CV |
| Stacking | 1.08% | 0.0663 (9.3% of mean) | 5.93 pp | ❌ fails: CV, shuffle |
| Decision Tree | 0.00% | 0.2175 (40.7% of mean) | 0.00 pp | ❌ fails: CV |
| Naive Bayes | n/a (deterministic) | 0.0907 (12.7% of mean) | 0.00 pp | ❌ fails: CV |
| XGBoost | 0.00% | 0.0323 (4.2% of mean) | 0.00 pp | ✅ STABLE |
| Random Forest | 16.71% | 0.0215 (2.9% of mean) | 2.43 pp | ❌ fails: seed, shuffle |
| Gradient Boosting | 0.00% | 0.0303 (4.0% of mean) | 0.00 pp | ✅ STABLE |
| Perceptron | n/a (deterministic) | 0.2255 (46.6% of mean) | 15.36 pp | ❌ fails: CV, shuffle |
| SLP | 100.00% | 0.1729 (47.0% of mean) | 0.00 pp | ❌ fails: seed, CV |
| AdaBoost | 0.00% | 0.0612 (8.2% of mean) | 0.00 pp | ❌ fails: CV |

## Classification — frozen contract target (`target_high_volatility`, train-Q75; 25% positive in train, 9% in validation)
| Model | Seed variance (<2%) | CV std (<5% of mean) | Shuffle Δ (≤2) | Status |
|---|---|---|---|---|
| Logistic | 5.39% | 0.3153 (65.4% of mean) | 13.48 pp | ❌ fails: seed, CV, shuffle |
| Stacking | 0.27% | 0.0998 (12.4% of mean) | 2.16 pp | ❌ fails: CV, shuffle |
| Decision Tree | 0.00% | 0.1519 (20.3% of mean) | 0.00 pp | ❌ fails: CV |
| Naive Bayes | n/a (deterministic) | 0.1079 (13.3% of mean) | 0.00 pp | ❌ fails: CV |
| XGBoost | 0.00% | 0.0975 (11.9% of mean) | 0.00 pp | ❌ fails: CV |
| Random Forest | 5.39% | 0.1170 (14.8% of mean) | 0.81 pp | ❌ fails: seed, CV |
| Gradient Boosting | 0.00% | 0.1098 (13.5% of mean) | 0.00 pp | ❌ fails: CV |
| Perceptron | n/a (deterministic) | 0.1508 (20.1% of mean) | 0.27 pp | ❌ fails: CV |
| SLP | 0.00% | 0.1508 (20.1% of mean) | 0.00 pp | ❌ fails: CV |
| AdaBoost | 0.00% | 0.1013 (12.7% of mean) | 0.00 pp | ❌ fails: CV |

## Findings

1. **The conclusion 'all models stable' is not supported.** Only XGBoost and Gradient Boosting (median target) pass all three checks; no other model does, and none of the regression models pass the CV check.
2. **CV criterion fails almost everywhere** — this is regime shift across time blocks (volatility level differs by period), not model instability: MSE varies 77–200% of its mean across folds. Even under an absolute reading (std < 0.05), Polynomial (0.18) and most classifiers (0.06–0.32) fail.
3. **Degenerate classifiers.** On the median-split target, Logistic, SLP and Perceptron predict a single constant class on raw (unscaled) features, and the class flips with the seed (Logistic/SLP: 100% seed disagreement). On the Q75 target, SLP and Perceptron predict all-0, so their 0.9084 validation accuracy equals the majority-class baseline (1 − 0.0916). Standardizing inputs is the likely fix; I did not change any model.
4. **Random Forest** is seed-sensitive (16.7% prediction disagreement on the median target, 10 trees) — more trees would reduce this.
5. **Target mismatch (important for M10).** The saved M5–M8 classifier predictions/metadata reproduce exactly when trained on `y_reg > median(train)` (≈50% positive), not on the frozen contract label `target_high_volatility` (Q75). Their recorded val accuracies (0.45–0.62) are therefore for a different task than the project contract. Random Forest's saved predictions do not reproduce exactly (82% match; hyperparameters beyond `n_estimators` are unrecorded). Regression models and SLP reproduce saved validation predictions bit-for-bit.
6. Logistic (M5-01) and Decision Tree (M5-06) have no saved artifacts to cross-check; the scratch `classification/logistic.py` was assumed for Logistic.

## Conclusion

**Not ready for final evaluation as-is.** Regression models are deterministic and shuffle-invariant but have unstable fold-to-fold error; several classifiers are degenerate. See the M10 note in the handoff.
