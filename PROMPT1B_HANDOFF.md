# Prompt 1B Handoff — Revised Test-Access Policy

## Status

Prompt 1B implementation remains blocked and must not resume from the overbroad policy that globally prohibited accepted Test reads across the repository suite.

The five Claude partial-remediation files were restored to HEAD in a targeted rollback:

- `src/build_experiment_datasets.py`
- `src/build_spike_reports.py`
- `tests/test_experiment_datasets.py`
- `tests/test_spike_contract.py`
- `tests/test_spike_reports.py`

Prompt 1 and Prompt 1A modeling/orchestration changes remain preserved.

## Revised policy

### Data-pipeline scope

Legacy Phase 1/Phase 2 tests may read the accepted Test split **read-only** only when their direct purpose is to:

- create the chronological Test split;
- validate schema, identity, or integrity;
- prove that Test rows are not filtered;
- prove that Test data is not modified; or
- verify deterministic data-pipeline/report behavior.

This allowance is limited to Phase 1/Phase 2 data preparation, integrity, audit, and reporting code. It does not authorize model fitting, model selection, or use of Test outcomes for modeling decisions.

### Modeling scope

Accepted Test access remains strictly prohibited for:

- `src/modeling/**` training and orchestration;
- M3–M7 modeling workflows;
- preprocessing fit;
- feature selection;
- hyperparameter search;
- model fitting;
- model selection;
- Validation comparison; and
- the clean M3 pilot.

The modeling guard must protect modeling tests and training processes. It must not globally block legitimate read-only Phase 1/Phase 2 data-pipeline verification.

## Rollback boundary

This rollback did not open, read, hash, copy, or inspect accepted Test CSV content. No tests, training, pilot, commit, amend, or push were performed.
