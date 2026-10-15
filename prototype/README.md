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
- `assrd/models.py` — the six classifiers, their search spaces, cost-weighted
  fitting, and validation-only threshold selection
- `assrd/phase2.py` — the experiment driver and CLI; writes `results/`
- `tests/test_phase2.py` — 39 targeted tests

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

### Phase 3 — feature ablation and robustness
- `assrd/splits.py` — stratified, episode-aware allocator; validates that the
  training reserve and minimum stealth-episode targets can be met and raises
  `SplitConstraintError` (with a reported shortfall) rather than silently
  under-filling a stratum; emits a composition manifest
- `assrd/ablation.py` — the nested feature-group arms, `fit_fixed` (Phase 2
  `best_params` applied without re-tuning), per-subgroup evaluation, and
  bootstrap threshold-stability analysis
- `assrd/phase3.py` — the driver, the M1–M4 matrix, the C1–C5 comparisons,
  test-oracle quarantine, preflight self-check, artifact writers and CLI
- `tests/test_phase3.py` — 34 targeted tests (75 in total with Phases 1–2)

**Feature arms (nested except the last).** `A_raw_only` (3 baseline counts) →
`B_trailing_level` (15) → `C_plus_activity` (19) → `D_plus_volatility` (27) →
`E_full` (29); plus the non-nested `T_temporal_only` (26, dropping the raw
counts). Each arm is exactly a column set, so B ⊂ C ⊂ D ⊂ E by construction.

**One unambiguous matrix.** Two axes are crossed and every cell is reported:

| mode | cost weighting | decision threshold |
|------|----------------|--------------------|
| M1 **(primary)** | cost-weighted | selected on validation |
| M2 | cost-weighted | fixed 0.5 |
| M3 | unweighted | selected on validation |
| M4 | unweighted | fixed 0.5 |

The primary comparisons, declared before the run: **C1** `E_full` vs
`A_raw_only` (M1); **C2–C4** M1 vs M2/M3/M4 per arm; **C5** `E_full` vs
`T_temporal_only` (M1).

**Tuning is off the ablation path.** `fit_fixed` applies Phase 2's recorded
`best_params`; a test spies on the search constructor to prove the ablation path
builds no `GridSearchCV`. `--retune-full` separately re-tunes *only* the
`E_full` arm (recorded as `E_full_retuned`) to quantify the tuning effect.

**The test oracle is quarantined.** The test-optimal threshold is computed only
as a labelled upper bound on the generalisation gap. It lives in a separate
`diagnostics` block with `achievable_deployment_performance: false`, never
enters model/feature/hyperparameter/threshold selection, and is checked three
ways (static source scan, runtime spy, preflight self-check).

**Preflight before any fit.** `python -m assrd.phase3 --quick --self-check`
evaluates leakage, composition, reproducibility, output schema, oracle
isolation, runtime/memory feasibility and Phase 2 artifact preservation, then
exits **without fitting a model**.

**Honest subgroup support.** The manifest reports total/attack/benign episodes,
per-family and stealth/non-stealth episodes in every split, and the
(family, stealth) cells present. Support is flagged `absent` / `insufficient` /
`thin` / `adequate`, and no comparison is drawn from an `insufficient` or
`absent` cell. Family coverage is guaranteed; individual (family, stealth)
cells are reported, not guaranteed.

Later phases: explainability + drift (Phase 4), live monitor (Phase 5),
reproducible reports.

## Running

```powershell
cd prototype
python -m pytest -q                          # all tests (114: 1 skipped, needs symlink rights)
python -m assrd.phase2 --n-windows 20000 --seed 42 --folds 5

python -m assrd.phase3 --quick --self-check  # preflight only; fits nothing
python -m assrd.phase3 --quick               # acceptance run (small trace)
python -m assrd.phase3                       # full experiment

# six-model re-baselining (Logistic Regression added; originals left untouched)
python -m assrd.rebaseline check
python -m assrd.rebaseline run-phase2
python -m assrd.rebaseline compare-params    # GR3 advisory gate
python -m assrd.rebaseline run-phase3 --mode quick
python -m assrd.rebaseline run-phase3 --mode full
python -m assrd.rebaseline preflight --stage full
python -m assrd.rebaseline verify            # Tier A: 9/9 byte-identical
```

Phase 2 results land in `results/phase2_results.json` plus two CSVs
(`_models.csv`, `_confusion.csv`). Phase 3 writes `phase3_results.json` plus
`_splits`, `_ablation`, `_per_family`, `_thresholds` and `_diagnostics` CSVs —
separate basenames, so a Phase 3 run never overwrites Phase 2 evidence. Every
figure in them is measured on the run that wrote the file; nothing is asserted
in advance.

### Six-model re-baselining

Adding Logistic Regression as a sixth model would otherwise make Phase 3 fall
back to hardcoded parameters for the new model while using tuned parameters for
the other five. `assrd/rebaseline.py` re-runs Phase 2 first and passes the
resulting six parameter sets into Phase 3 explicitly.

* The nine **original** Phase 2/Phase 3 artifacts stay byte-for-byte identical;
  their SHA-256 digests are embedded in `rebaseline.py` and re-verified on every
  command.
* New output is written to `results/rebaseline_6model/` (authoritative) and
  `results/rebaseline_6model/quick/` (smoke test, marked non-authoritative).
* `results/_rebaseline_forensics/` holds the lock and any forensic snapshot. It
  must exist before use and is **never created automatically**.
* Every mutating command takes an exclusive lock outside the directory it
  protects, and any destructive reset first captures a sealed, uniquely created
  forensic snapshot. There is no bypass flag.

These are **emulator** results on generated data, not real-world detection.

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
  Increasing `n_episodes` is the fix. Phase 3's allocator splits more
  generously (20 % / 20 %) and reports what it actually produced.
* **Stealth windows are unevenly distributed.** The seed-42 trace puts 256
  stealth windows in train, 0 in validation and 38 in test, so the reported
  threshold is selected without ever seeing a rate-limited window. Phase 3
  raises minimum stealth-episode targets for validation and test to address this.
* **Emulated data.** See the provenance note at the top. No field detection rate
  is claimed.
* **Precision is prevalence-bound.** At ~5.6 % attack prevalence in test, a
  detector with high sensitivity pays for it in false alarms; precision of
  0.79–0.93 reflects the class balance, not a defect.
* **The raw-count vs temporal-feature ablation is not run.** Phase 2 uses all 29
  model inputs. That ablation is Phase 3's job and is not claimed here.

## Known Phase 3 limitations

* **Emulated data.** See the provenance note at the top; no field detection rate
  is claimed.
* **Transferred hyperparameters.** `best_params` come from Phase 2, selected on
  a different trace. That transfer assumption is documented and `--retune-full`
  probes it rather than hiding it.
* **Thin cells.** Some (family, stealth) cells hold too few independent episodes
  to support a comparison and are flagged with a support flag instead.
* **Stealth is modelled.** Robustness is measured against the emulator's
  rate-limited generator; it bounds behaviour on modelled evasion, not real
  evasion.
