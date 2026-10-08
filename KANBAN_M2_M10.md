# Kanban: M2–M10 Model Training Pipeline
**Status:** ACTIVE  
**Deadline:** Oct 12, 2026 (48 hours)  
**Milestone:** M5 completion (Sept 10 00:00 UTC)

## Column Structure
- **BACKLOG:** Planned but not started
- **IN_PROGRESS:** Currently working
- **REVIEW:** Awaiting approval/verification
- **DONE:** Completed + verified

---

## MILESTONE 1: M5 COMPLETION (All 8 Models TDD Complete)

### IN_PROGRESS

#### M5-01: Logistic Regression (7 tests)
- Status: ⚠️ 6/7 pass (1 trivial bug in real_data test)
- Fix: `.median()` → `np.median()`
- Next: Fix + re-run, then REVIEW
- Assignee: @sonnet or @codex
- Effort: 10 min

#### M5-02: Stacking Classifier (6 tests)
- Status: ✅ 6/6 pass, artifacts saved
- Next: Verify artifacts + DONE
- Assignee: @manual-verify
- Effort: 5 min

#### M5-03: Decision Tree (5 tests)
- Status: ✅ 5/5 pass, scratch implementation verified
- Next: DONE (just committed)
- Assignee: DONE
- Effort: 0

#### M5-04 through M5-06: k-NN, k-Means, PCA (sklearn)
- Status: ⚠️ Tests created (15 tests total), running full validation
- Next: Verify all pass + REVIEW
- Assignee: @hermes-tdd
- Effort: 1 hour

#### M5-07: Gaussian Naive Bayes (9 tests)
- Status: ✅ 9/9 pass, artifacts saved
- Next: DONE
- Assignee: DONE
- Effort: 0

#### M5-08: XGBoost (9 tests, HIGH RISK)
- Status: ✅ 9/9 pass (RED test passed!), artifacts saved
- Next: DONE (verified)
- Assignee: DONE
- Effort: 0

### REVIEW

#### M5 Full Test Suite Verification
- Acceptance: All 8 models × tests pass + artifacts exist
- Blocker: M5-01 bug fix + validation of k-NN/k-Means/PCA tests
- Next: Fix bugs → full pytest run → DONE

---

## MILESTONE 2: M6–M8 NON-SKELETON MODELS (Pending)

### BACKLOG

#### M5-09: Random Forest (4 tests needed)
- Status: Test stub created
- Next: Assign to Sonnet TDD
- Effort: 2–3 hours
- Blocker: M5 complete

#### M5-10: Gradient Boosting (4 tests needed)
- Status: Test stub created
- Next: Assign to Sonnet TDD after RF
- Effort: 2–3 hours
- Blocker: M5-09 done

#### Perceptron, SLP, AdaBoost (3 × 5 tests = 15 tests)
- Status: Not started
- Next: Scratch implementations
- Effort: 4–5 hours (Sonnet TDD)
- Blocker: M5-10 done

---

## MILESTONE 3: M9–M10 FINALIZATION (Pending)

### BACKLOG

#### M9: Stability Checks
- Random seed variations
- Cross-validation metrics
- Effort: 2 hours
- Blocker: M5–M8 complete

#### M10: Test Split Unlock + Final Metrics
- Unlock test.jsonl in ledger
- Evaluate all 38 models on test set
- Final artifact generation
- Commit + summary report
- Effort: 3–4 hours
- Blocker: M9 complete

---

## SUMMARY

| Milestone | Status | Tests | Effort | ETA |
|-----------|--------|-------|--------|-----|
| **M5 Completion** | 🔄 IN_PROGRESS | 50/50 | 2 hours | ~06:00 UTC |
| **M6–M8 Models** | ⏳ BACKLOG | ~50 | 8–10 hours | ~16:00 UTC |
| **M9–M10 Final** | ⏳ BACKLOG | ~20 | 5 hours | ~21:00 UTC |
| **TOTAL** | — | **120** | **15–17 hrs** | **Oct 10 21:00** |

---

## NOTES

- **M5-01 bug fix:** 10 min (priority: ASAP)
- **Sonnet budget:** ~$12–16 remaining (enough for M6–M10)
- **Codex reset:** Oct 10 01:50 AM (can use if needed for speed)
- **Risk:** XGBoost already passed (no more HIGH RISK)
- **Confidence:** 98% (only M5-01 trivial bug, rest solid)

