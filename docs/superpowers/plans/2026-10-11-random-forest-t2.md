# Random Forest T2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement scratch Random Forest T2 with strict sklearn parity and a seven-candidate validation search on both variants.

**Architecture:** Extend the existing NumPy DecisionTree to support per-node `max_features="sqrt"` while retaining the T1 default. RandomForest bags those trees using sklearn-compatible seeds and bootstrap counts. The shared trainer runs four raw candidates, then PCA(2,4,6) with the best raw hyperparameters, and persists the global winner and its matching preprocessing.

**Tech Stack:** Python 3.14.8, NumPy, existing ML core; sklearn only in `reference()` and parity tests.

**Spec:** `AGENTS.md` §§4–8; `docs/tasks/codex-tree-family.md` T2; `docs/plan/MODEL_CARDS.md` Random Forest section; `configs/modeling.json`.

## Global Constraints

- Never read the accepted Test split or change the data pipeline.
- Keep model implementation scratch NumPy; sklearn may appear in `reference()` and tests only.
- Use `np.random.RandomState`, defaults for all constructor parameters, and strict label/probability parity.
- Preserve the existing six T1 Decision Tree grid outcomes and read-only X.
- Do not commit `.venv/`, `tmp/`, `outputs/modeling/**`, or generated data.

## Review Focus

- Bootstrap duplicates and omitted rows: tree leaf-size checks must count positive-weight rows as sklearn does; test with zero weights.
- Constant features: per-node feature sampling must continue far enough to find a usable split; test synthetic constant columns.
- Seed reuse: one derived tree seed controls both its bootstrap draw and split feature order; compare bootstrap indices and tree structures.
- PCA winner: reload must use that candidate's validation matrix and persist its standardizer/components; test this path.
- Candidate provenance: seven distinct rows must record PCA dimension and the three PCA rows must reuse the best raw hyperparameters; test search JSON and config.

---

### Task 1: Preserve CART parity with sampled features

**Files:** Modify `src/ml/classification/decision_tree.py`; extend `tests/ml/classification/test_decision_tree.py`.

**Interfaces:** `DecisionTree(..., max_features: str | None = None)`; `fit(X, y, sample_weight=None)` accepts bootstrap counts; existing `state_dict()` remains compatible.

- [x] Add failing tests for zero-weight rows, `sqrt` feature sampling at each node, constant columns, and T1 default parity.
- [x] Run `./.venv/Scripts/python.exe -m pytest tests/ml/classification/test_decision_tree.py -q` and record the expected failures.
- [x] Implement the minimum feature-order/active-row change; `reference()` passes `max_features` through.
- [x] Re-run focused tests and confirm T1 real Train/Validation parity remains exact.
- [x] Request independent review and resolve important findings.

### Task 2: Implement and verify scratch forest

**Files:** Create `src/ml/classification/random_forest.py` and `tests/ml/classification/test_random_forest.py`; update `src/ml/registry.py`.

**Interfaces:** `RandomForest(BaseModel)` with defaulted `n_estimators`, `max_depth`, `min_samples_leaf=5`, `max_features="sqrt"`, `bootstrap=True`, `random_state=42`; usual prediction, state, reload, and reference methods.

- [x] Write failing tests: one tree without bootstrap/feature cap agrees with DecisionTree using the derived seed; bootstrap indices agree with sklearn; probability aggregation is an arithmetic mean; invalid inputs fail; round-trip preserves predictions.
- [x] Run the new tests and record the expected failures.
- [x] Implement per-tree seeds from `RandomState(random_state).randint(np.iinfo(np.int32).max, size=n_estimators)`, bootstrap counts via `RandomState(seed_i).randint(0,n,n)`, and mean probability.
- [x] Add strict parity tests on synthetic and real Train/Validation data, including tree structure checks; fix discrepancies without relaxing tolerance.
- [x] Register `random_forest` and verify CLI discovers it.
- [x] Request independent review and resolve important findings.

### Task 3: Seven-candidate search and truthful results

**Files:** Modify `src/ml/core/trainer.py` and `tests/ml/core/test_trainer.py`; update `docs/tasks/codex-tree-family.md`, `docs/models/random_forest/MODEL_PROGRESS.md`, and the Codex model-status index if present.

**Interfaces:** Existing `train_model()` and CLI remain unchanged; RF search records 4 raw grid candidates plus PCA 2/4/6 with the best raw params, then persists the global winner.

- [x] Add failing trainer tests for seven search rows, PCA metadata, best-raw parameter reuse, and reload of a PCA-winning candidate.
- [x] Run the focused trainer tests and record expected failures.
- [x] Add a focused RF branch in trainer using existing `Standardizer`, `PCA`, `evaluate`, and `select_candidate`; preserve existing behavior for other models.
- [x] Run the trainer, Decision Tree, and forest focused tests; fix regressions.
- [x] Run `./.venv/Scripts/python.exe -m src.ml.run train --model random_forest --variant with_spike` and again for `non_spike`; inspect all seven rows, artifacts, metrics, and load verification.
- [x] Update T2 and model progress with actual outputs, parity evidence, pitfalls, and the leaderboard dependency; do not claim full DoD until leaderboard rows exist.
- [x] Request final independent review and run fresh verification before completion.
