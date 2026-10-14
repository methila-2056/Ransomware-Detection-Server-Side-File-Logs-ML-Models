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

### Phase 1 — foundations
- `assrd/compat.py` — scikit-learn/xgboost compatibility repair (verified necessary)
- `assrd/data.py` — behavioural emulator + episode-aware, leakage-safe split
- `assrd/features.py` — the 3 raw baseline counts and the 26 proposed temporal features
- `tests/test_phase1.py` — 7 targeted tests

### Phase 2 — models, cost-sensitive tuning, threshold selection
- `assrd/metrics.py` — confusion matrix, sensitivity/specificity/precision/F1,
  ROC-AUC, average precision, and the `C_FP`/`C_FN` cost terms
- `assrd/models.py` — the five classifiers, their search spaces, cost-weighted
  fitting, and validation-only threshold selection
- `assrd/phase2.py` — the experiment driver and CLI; writes `results/`
- `tests/test_phase2.py` — 34 targeted tests

**Cost model.** `C_FP = 1`, `C_FN = 10` (`assrd.metrics`), applied twice for
two distinct reasons:

* **during fitting** — `cost_sample_weights()` gives every attack window weight
  10 and every benign window weight 1, so the minimiser sees the asymmetry;
* **after fitting** — `select_threshold()` sweeps a fixed grid and picks the
  threshold minimising `C_FP*FP + C_FN*FN` **on the validation split**. Ties are
  broken towards the *larger* threshold (fewer alarms for equal cost).

Tuning is scored on `recall`, which under a 10:1 asymmetry is the
threshold-independent quantity that best tracks the objective.

**Leakage boundary.** Tuning sees the training split only; the threshold comes
from validation only; the test split is scored once, at the already-frozen
threshold. `run_phase2` raises rather than trains if any two partitions share an
attack episode. `test_threshold_oracle_diagnostic` records what a test-optimal
threshold *would* have been, purely to expose the generalisation gap — it is
never used to score anything.

**joblib backend.** Every `GridSearchCV` runs inside
`assrd.compat.threading_backend()`. `loky` workers are separate interpreters and
would not carry the xgboost tag patch, so a test spies on the context manager to
prove it is entered rather than merely available.

Later phases: ablation/transfer evaluation (Phase 3), explainability + drift
(Phase 4), live monitor (Phase 5), reproducible reports.

## Running

```powershell
cd prototype
python -m pytest -v                          # Phase 1 + Phase 2 (41 tests)
python -m assrd.phase2 --n-windows 20000 --seed 42 --folds 5
```

Results land in `results/phase2_results.json` plus two CSVs
(`_models.csv`, `_confusion.csv`). Every figure in them is measured on the run
that wrote the file; nothing is asserted in advance.

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

## Running the Phase 1 tests alone

```powershell
cd prototype
python -m pytest tests/test_phase1.py -v
```

## Known Phase 2 limitations

Measured, not assumed. Recorded here so they travel with the numbers.

* **Thin validation episodes.** With 24 attack episodes and a 10 % validation
  fraction, validation receives 2 episodes (~129 attack windows) and test 2 more.
  Threshold selection therefore rests on a small number of independent episodes;
  confidence intervals over episodes, not over windows, are the honest error bar.
  Increasing `n_episodes` is the fix.
* **Stealth windows are unevenly distributed.** The seed-42 trace puts 256
  stealth windows in train, 0 in validation and 38 in test, so the reported
  threshold is selected without ever seeing a rate-limited window. Evasion
  robustness is Phase 3 work and is *not* demonstrated here.
* **Emulated data.** See the provenance note at the top. No field detection rate
  is claimed.
* **Precision is prevalence-bound.** At ~5.6 % attack prevalence in test, a
  detector with high sensitivity pays for it in false alarms; precision of
  0.79–0.93 reflects the class balance, not a defect.
* **The raw-count vs temporal-feature ablation is not run.** Phase 2 uses all 29
  model inputs. No claim about what the 26 temporal features add over the 3 raw
  counts is made or implied here.
