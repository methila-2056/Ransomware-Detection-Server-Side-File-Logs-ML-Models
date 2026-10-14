# ASSRD prototype

Self-contained prototype of the **Adaptive Server-Side Ransomware Detection
(ASSRD)** framework described in the Phase I report abstract. It depends on
nothing from the main project's `ml/` package.

> **What the data is.** Attack traffic is produced by a behavioural
> **emulator** (`assrd/data.py`), not by executed malware. Benign profiles are
> modelled on the base paper's Table 3; attack profiles are modelled from
> published family characteristics. Every result file records this provenance.
> This is the prototype's principal limitation.

## Status

Implemented in phases. Current status is reported in the section below.

### Phase 1 — foundations (this phase)
- `assrd/compat.py` — scikit-learn/xgboost compatibility repair (verified necessary)
- `assrd/data.py` — behavioural emulator + episode-aware, leakage-safe split
- `assrd/features.py` — the 3 raw baseline counts and the 26 proposed temporal features
- `tests/test_phase1.py` — 7 targeted tests

Later phases: models + cost threshold, evaluation/ablation/transfer,
explainability + drift, live monitor, reproducible reports.

## Environment

Python 3.12 · numpy 2.2.3 · pandas 2.2.3 · scikit-learn 1.6.1 · xgboost 2.1.3 ·
watchdog 6.0.0.

### Compatibility caveat

scikit-learn 1.6 + xgboost 2.1.3 is an incompatible pair (xgboost orders
`ClassifierMixin` after `BaseEstimator`, breaking sklearn's `__sklearn_tags__`).
`assrd/compat.py` repairs this at import time, and tuning must run under joblib's
**threading** backend because `loky` workers are separate interpreters that do
not carry the patch. Both facts are asserted by a targeted test rather than
assumed.

## Running the Phase 1 tests

```powershell
cd prototype
python -m pytest tests/test_phase1.py -v
```
