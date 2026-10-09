# Recovery Gate Report

Date: 2026-10-09. Final verdict: **FAIL/BLOCKED**.

## Scope and baseline

The Recovery Gate task began from clean HEAD `afb4ae5`. The earlier partial
KANBAN and PROJECT_STATUS changes are task changes, not pre-existing changes.
This implementer preserved the valid partial implementation and fixed only the
four reviewed blockers. Everything remains uncommitted. No dependencies were
installed. No Test CSV was opened, read, hashed, copied or created during this
fix pass. The authorization test uses an in-memory synthetic split with the
file reader mocked. No production model was retrained, tuned or evaluated.
Existing workflow tests exercise synthetic fixtures as part of the requested suite.

The inventory summary was produced during this Recovery Gate's earlier audit pass
and consumed by the final fix pass. The final fix and documentation passes did
not inspect or modify legacy outputs; inventory figures below are audit evidence.

## Fixes and TDD evidence

- `_check_artifacts` opens every manifest-declared NPZ with
  `np.load(..., allow_pickle=False)` and materializes every member. Failures
  create the `unsafe_object_npz` artifact issue. Classification, resume,
  completion and finalization share this check. Save/load validation is preserved.
- M5/M6 classify every existing directory before reading its manifest or
  computing current hashes. Only completed classification reaches resume.
  Nonexistent paths remain `missing`.
- Removed the untracked global `tests/conftest.py` patch. A non-autouse fixture
  in `test_finalize_gate.py` isolates only the three successful authorization
  tests. A default-path regression uses mocked filesystem operations so it
  cannot mutate the production ledger.
- Corrected PROJECT_STATUS and KANBAN to retain FAIL/BLOCKED and the final
  legacy test count. Removed unsupported training/module and verification claims.

Regression tests were written before production fixes. After correcting an
initial fixture setup error (immutable JSON writer used to overwrite a file),
legacy regressions showed **11 failed, 14 passed**: object NPZ incorrectly
classified completed, both corrupt manifests raised JSONDecodeError, and both
helpers passed invalid directories to resume. The focused default-ledger and
unsafe-finalization tests then showed **2 failed, 11 deselected** before fixes.
After fixes the combined legacy/finalization suite showed **38 passed**.
The legacy file contains **25 tests**; finalization contains **13 tests**.

## Entrypoint evidence and remaining gate blockers

Source inspection confirms only M6 has a proven model-training CLI (`main` / `__main__` in
`src/modeling/m6_training.py`). M5 exposes the library function
`train_classifier` in `src/modeling/classification_training.py`; no proven M5 CLI.
M3/M4/M7 have no proven retrain entrypoint. The general runner's infrastructure
no-op is not proof of a model training entrypoint. No retrain command was executed.

All four reviewed fix blockers are addressed. The Recovery Gate remains
FAIL/BLOCKED: modern replacement artifacts have not been produced or validated
in this pass, and M3/M4/M7 retrain entrypoints remain unproven. Stop here.

## Actual verification

All test runs used:
`C:/Users/ADMIN/Documents/Repository/QQQ-Volatility-Forecasting/.venv/Scripts/python.exe`.
Commands ran from the repository root; `.venv/Scripts/python.exe` below resolves
exactly to that interpreter.

```powershell
.venv/Scripts/python.exe -m src.modeling.config_sync --check
```

Exit 0: `Modeling config and generated plan block match`. No config was changed.

```powershell
.venv/Scripts/python.exe -m pytest tests/modeling/test_legacy_detection.py -q --tb=short
.venv/Scripts/python.exe -m pytest tests/modeling/test_finalize_gate.py -q --tb=short -k 'default_ledger or unsafe_npz'
.venv/Scripts/python.exe -m pytest tests/modeling/test_legacy_detection.py tests/modeling/test_finalize_gate.py -q --tb=short
.venv/Scripts/python.exe -m pytest tests/modeling/test_legacy_detection.py tests/modeling/test_persistence.py tests/modeling/test_finalize_gate.py tests/modeling/test_runner.py tests/modeling/test_config.py tests/modeling/test_datasets.py tests/modeling/test_artifacts.py tests/modeling/test_classification_workflow.py tests/modeling/test_m6_workflow.py -q --tb=short
```

First two commands are the red evidence described above. Combined focused
suite: 38 passed. Required nine-file suite: **204 passed, 15 warnings**, exit 0.
Warnings are NumPy's `Install pyyaml for better output` warning (14 runner,
1 artifacts); no filter or installation was needed. The required suite was
repeated after final code cleanup. No full-suite result is claimed.

```powershell
git diff --exit-code -- outputs/.test_ledger.json
git diff --name-only -- outputs/modeling
git diff --check
```

Ledger check: exit 0, empty output; no restoration was necessary.
Modeling output check: empty output. No files under
`outputs/modeling/with_spike` were modified, deleted or renamed.
Whitespace check: exit 0. Final file inventory and HEAD diff stat are included
in the final handoff; untracked files are excluded from the diff stat.

## M2–M10 implementation and completion matrix

`MODEL_TRAINING_PLAN.md` sections M2–M10 are authoritative. Inventory is
**19 algorithms: regression 4 + classification 13 + clustering 2**. PCA is
preprocessing, not a third clustering model. Historical Kanban card labels do
not redefine these milestones. “Verified now” below means checkpoint verification,
not new tests in this documentation pass. Source presence is not artifact completion.

| Milestone | implemented | verified now | incomplete | blocked |
| --- | --- | --- | --- | --- |
| M2 Shared infrastructure | Loader, contracts, preprocessing, metrics, persistence, runner and config sync code | Targeted infrastructure tests; strict NPZ validation, status/resume and finalization protections; config sync match | Production lifecycle completion across inventory not established | Modern replacement artifacts absent |
| M3 Pilot reuse proof | Multiple linear regression source exists | No current two-variant production pilot verification | Fresh scratch/reference pilot, reload and pilot-only artifact proof | Retrain BLOCKED: no proven CLI |
| M4 With-Spike regression | Four regression source implementations exist | No current production training/reference or artifact verification | Four complete modern scratch/reference runs and frozen evaluation evidence | Retrain BLOCKED: no proven CLI |
| M5 Basic classification | Six planned classifier implementations and library `train_classifier` | Targeted classification workflow and classification-first status checks | Complete production scratch/reference artifacts and metrics | Retrain BLOCKED: no proven CLI; `train_classifier` is library-only |
| M6 Advanced classifiers | Seven-model training workflow, scratch/reference support and CLI | Targeted M6 workflow/status tests; source-proven CLI | Production searches and complete modern replacement runs | Gate completion blocked by missing artifacts; CLI template available |
| M7 Clustering/PCA | Clustering and preprocessing source exists | No current production clustering/PCA completion verification | Two clustering artifact sets and required PCA mechanism evidence | Retrain BLOCKED: no proven CLI |
| M8 With-Spike submission | Status/report documentation exists | No complete artifact-to-submission trace verified | Complete notebooks, report, slides and demos tied to modern artifacts | M4–M7 completion |
| M9 Non-Spike reruns | Shared variant plumbing exists | No fresh full-inventory reruns verified | Fresh fits, provenance parity and complete artifacts | M8 and frozen protocol; no production reruns performed |
| M10 Paired comparison/final documentation | Finalization gate code exists | Targeted gate protections only | Paired joins, final metrics and final deliverables | M9; Test remains gated |

## Legacy inventory and historical evidence

Source: `outputs/reports/model_outputs_audit_summary.json`, generated in this
Recovery Gate's earlier audit pass and consumed by the final fix pass.
Audit counts: **26 run directories; 13 empty / 13 nonempty; 7 model.npz;
0 manifests; 0 modern-complete**. All 26 are legacy-incompatible. No legacy
files were modified. Paths below are project-relative; their common prefix is
`outputs/modeling/with_spike/` (append each listed suffix to that prefix).

| Nonempty path suffix | Audit summary |
| --- | --- |
| classification/knn/scratch/M5_04 | Metadata and train/validation predictions |
| classification/naive_bayes/scratch/M5_07 | Metadata, model.npz, predictions and probabilities |
| classification/perceptron/scratch/M5_11 | Metadata and predictions |
| classification/slp/scratch/M5_12 | Metadata and predictions |
| ensemble/adaboost/scratch/M5_13 | Metadata and predictions |
| ensemble/gradient_boosting/scratch/M5_10 | Metadata and predictions |
| ensemble/random_forest/scratch/M5_09 | Metadata and predictions |
| ensemble/stacking/scratch/M5_02 | Metadata, model JSON/NPZ, predictions/probabilities; probable object NPZ |
| ensemble/xgboost/scratch/M5_08 | Metadata, loss history, model NPZ, predictions/probabilities; confirmed unsafe object NPZ |
| regression/elastic_net/scratch/M4_04 | Metadata, model JSON/NPZ, predictions; historical Validation MSE 0.00598 |
| regression/mlr/scratch/M3_pilot_run | Metadata, model JSON/NPZ, predictions; historical Validation MSE 0.00474 |
| regression/polynomial/scratch/M4_03 | Metadata, model JSON/NPZ, predictions; historical Validation MSE 0.00477 |
| regression/simple_linear/scratch/M4_01 | Metadata, model JSON/NPZ, predictions; historical Validation MSE 0.00546 |

The 13 empty suffixes are:

- `advanced/knn/M8_02`, `advanced/mlp/M8_03`, `advanced/svm/M8_01`
- `classification/decision_tree/M5_06`, `classification/logistic/M5_01`, `classification/perceptron/M5_04`, `classification/slp/M5_05`
- `clustering/agglomerative/M7_02`, `clustering/kmeans/M7_01`, `clustering/pca/M7_03`
- `ensemble/adaboost/M6_03`, `ensemble/gradient_boosting/M6_02`, `ensemble/random_forest/M6_01`

The legacy PCA directory is an inventory entry, not an additional algorithm.
The supplied/verified earlier audit identifies legacy median-target classification
as incompatible with the canonical Original-Train Q75 target. Legacy classifier
accuracy is therefore not current Q75 performance evidence. Regression's continuous
target avoids that median-label incompatibility, but its recorded MSE values are
**historical audit evidence only**, not modern contract completion, fresh validation,
or permission to resume/finalize. Missing manifests disqualify every run.

## Canonical new-run artifact contract

`src/modeling/persistence.py::required_artifacts(manifest)` selects requirements
from `configs/modeling.json` by lifecycle and implementation. This function and
the config are authoritative; a single static list does not fit every model.
Every modern run needs `manifest.json` with lifecycle/status/role, selected
hyperparameters, config/input/code/dependency-lock/runtime identity, config ID,
Git revision/dirty state, timing, artifact checksums and load-verification receipt.

Configured supervised-tuned artifacts: `config.json`, `manifest.json`,
`model.npz`, `preprocessor.npz`, `search_results.json`,
`validation_metrics.json`, `validation_predictions.csv`, `load_verification.json`.
Scratch adds `model.json`; all applicable model runs add `preprocessor.json`.
Reference implementations substitute `model.joblib` for `model.npz` where applicable.
Scratch NPZ state must be strict numeric, loaded with `allow_pickle=False` and
all members materialized; structured state belongs in strict JSON sidecars.
Reload predictions/scores must pass the recorded equality/tolerance checks, with
receipt agreement between `load_verification.json` and the manifest.

Supervised-finalized adds `test_identity.json`, `test_metrics.json`,
`test_predictions.csv`, `test_access.json`, subject to the authorization gate.
Clustering-train requires `config.json`, `manifest.json`, `model.npz`,
`preprocessor.npz`, `train_assignments.csv`, `internal_metrics.json`,
`cluster_summary.json`, `figures`, `load_verification.json`, plus the applicable
JSON sidecars/reference substitution. Clustering-extension adds `extension.json`,
`validation_assignments.csv`, `extension_metrics.json`, `limitations.json`.
Infrastructure-noop uses `config.json`, `state.npz`, `state.json`,
`load_verification.json` with its manifest; it does not prove model training.

## Exact next commands and status

Config check (already verified in the checkpoint; not rerun in this docs pass):

```powershell
.venv/Scripts/python.exe -m src.modeling.config_sync --check
```

Source-proven M6 CLI template (not executed; placeholders require a fresh identity):

```powershell
.venv/Scripts/python.exe -m src.modeling.m6_training train --model random_forest --variant with_spike --out-root <NEW_OUTPUT_ROOT> --run-id <NEW_RUN_ID>
```

M3/M4/M5/M7 retrain is **BLOCKED because no proven CLI exists**. M5
`train_classifier` is library-only. No commands are invented for these milestones.
The next engineering action is to implement and test orchestration entrypoints,
then prove fresh Train/Validation artifact completion. That action was not performed.

## Files changed and artifact paths/completeness

All eight current task files (including prior implementation changes):

1. `KANBAN_M2_M10.md`
2. `PROJECT_STATUS.md`
3. `src/modeling/classification_training.py`
4. `src/modeling/m6_training.py`
5. `src/modeling/persistence.py`
6. `tests/modeling/test_finalize_gate.py`
7. `tests/modeling/test_legacy_detection.py` (untracked)
8. `RECOVERY_GATE_REPORT.md` (untracked)

`outputs/reports/model_outputs_audit_summary.json` exists as generated evidence,
is ignored by Git and is not tracked. This documentation pass changes only its
`generated_by` from `recovery_gate_pass` to `recovery_gate_audit`; inventory data
is preserved. The report is evidence, not a modern model artifact.
Legacy artifact completeness at `outputs/modeling/with_spike/` is zero modern
completed runs according to the earlier audit. No new production artifact path
or complete replacement run is claimed. Ledger: `outputs/.test_ledger.json`.

## Checkpoint commands, test totals, identities and limitations

Claude/Codex implementation activity is summarized as source/test orchestration,
inspection, edits and cleanup; repository verification commands are listed in
Actual verification above. Hermes's final targeted verification used the same
nine-file command listed there and reported **204 passed, 15 warnings, 0 failed,
0 skipped**. Focused verification was **38 passed**. Config sync matched and
ledger/modeling-output diffs were clean. These are checkpoint results, not new
executions in this documentation pass. No full suite is claimed.

Git evidence commands across the checkpoint/documentation handoff include:

```powershell
git status --short
git diff --stat
git diff --exit-code -- outputs/.test_ledger.json
git diff --name-only -- outputs/modeling
git diff --check
```

Baseline HEAD: `afb4ae5`; configured identity: `cfg-f904b7f92596`.
Earlier audit evidence: Train **1733**, Validation **371**, Q75 threshold
**0.2530580184684854**, fit on **Original Train only**, classification using
strict **>**. Supplied audit prevalence: **24.99% Train / 9.16% Validation**;
these are recorded audit evidence, not CSV measurements in this pass.
Inventory counts and historical regression MSE are audit evidence as labeled above.
No new dataset or artifact hashes were computed; provenance requirements are a
contract, not proof that legacy runs satisfy them.

Known unrelated stability failure remains recorded as a limitation; it was not
rerun in Hermes's final targeted suite or in this documentation pass. The 15
warnings in that suite are the narrow NumPy PyYAML warnings described above.
RED evidence is retained only with the fixture correction and exact counts
already recorded. No dependency installation, retraining, Test CSV access,
legacy inspection/modification or commit occurred in this documentation pass.

**Final verdict: FAIL/BLOCKED.** Production/test fixes pass targeted verification,
but modern replacement artifacts remain incomplete, M3/M4/M5/M7 orchestration
entrypoints are unproven, and downstream submission/rerun/paired-comparison gates
are unsatisfied. PROJECT_STATUS and KANBAN retain this verdict.