# Deliverable 4 — Prototype Technical Audit

> ## ⚠️ PARTIALLY SUPERSEDED
>
> This audit traced an earlier revision of the pipeline. Several findings have since been fixed.
> Current status of the ones raised here:
>
> | Finding | Status now |
> |---|---|
> | Tests overwrite `data/`, `models/` and results artefacts | **Resolved.** Training writes only to an explicit `output_dir`/`models_dir`; a CI gate verifies tracked files are unchanged after the suite. |
> | Empty `monitoring/` and `scripts/` packages | **Resolved.** `monitoring/agent.py` is populated; `scripts/` holds the POSIX launchers alongside the `.bat` files. |
> | POST endpoints unauthenticated | **Resolved.** Token auth plus a 127.0.0.1 default bind. |
> | xgboost / scikit-learn incompatibility masked by a version pin | **Resolved.** Diagnosed to an MRO ordering fault in xgboost 2.1.3 against sklearn 1.6's tag API; handled at runtime in `ml/compat.py`. |
> | `predict()` swallows failures | **Partly resolved.** Failures are logged rather than silently swallowed, and the serving path reports an explicit insufficient-history status. |
> | UI badge overstates CV folds | **Resolved in code.** The pipeline now uses 5-fold stratified CV and the badge agrees. |
> | Separability report mislabelled features | **Resolved.** It received the 26-column sequence matrix while hard-coding three count names, so rolling aggregates were labelled `create_count` and a benign maximum of 408 was reported where Table 3 caps at 60. It now requires explicit `feature_names`. |
>
> **Three defects found after this audit was written**, recorded here so they are not lost:
>
> 1. The serving path fed a 26-feature model three raw counts, so every prediction failed.
> 2. `predict()` returned `max(proba)` — confidence in the *predicted* class — where callers
>    read it as attack probability. An idle directory reported 0.98 and raised an alert.
> 3. `group_aware_split` could leave the training subset with no attack windows; it now raises
>    rather than producing a meaningless sensitivity figure.

**Scope:** every executable and configuration file in the project, traced from input to output.
**Rule applied:** a feature is credited only where the implementation (not the UI label or README)
supports it.

---

## Archived audit follows

---

## 1. System architecture (as implemented)

```
 Monitored directory (demo_server/)
        │  OS file events
        ▼
 watchdog Observer ──on_created/on_deleted/on_moved──▶ event_buffer (deque, thread-safe)
        │                                                     │
        │                                    1-second aggregation (monitor_loop)
        ▼                                                     ▼
 [cc, rc, dc] feature vector ──▶ StandardScaler? ──▶ best_model.pkl ──▶ label + confidence
                                                                              │
                                            ┌─────────────────────────────────┤
                                            ▼                                 ▼
                                 current_prediction (global)          SQLite predictions table
                                            │                                 │
                                            ▼                                 ▼
                                  /api/current-prediction            /api/detection-history
                                            └────────► Flask dashboard (index.html + dashboard.js)
```

- **Components:** `watchdog` observer → in-memory deque → 1-s aggregation loop → `predict()` →
  Flask REST JSON → vanilla-JS dashboard. SQLite persists every window's prediction.
- **Threading:** a monitor thread plus a Flask (threaded) server; a `threading.Lock` guards the
  deque. No locking around `current_prediction` or `store_prediction`.
- **Model loading:** `load_model()` auto-trains if `best_model.pkl` is missing (`app.py:112-115`).

## 2. Core algorithm / pipeline (evidence)

| Stage | Implementation | File:line |
|---|---|---|
| Event capture | `on_created→create`, `on_deleted→delete`, `on_moved→rename` | `app.py:152-159` |
| Buffering | timestamped dict appended under lock | `app.py:145-150` |
| Window pruning | keep events with `t ≥ now-1.0` | `app.py:166-174` |
| Feature counting | count create/rename/delete in window | `app.py:176-178` |
| Inference | `model.predict`, `model.predict_proba`, max prob = confidence | `app.py:126-137` |
| Persist | INSERT into `predictions` | `app.py:94-104` |

## 3. Data inputs, transformations, outputs

- **Input:** OS file events (create/delete/move) on `demo_server/`. Directories ignored.
- **Transformation:** raw events → per-1-s counts `(cc, rc, dc)`; optional `StandardScaler` for SVM.
- **Output:** `NORMAL`/`RANSOMWARE` + confidence; JSON APIs; SQLite history; dashboard.

## 4. Models, frameworks, dependencies

| Dependency | Role | Notes |
|---|---|---|
| Flask ≥2.3 | HTTP API + dashboard | registered in `requirements.txt` |
| scikit-learn ≥1.3,<1.6 | DT/RF/SVM/AdaBoost, split, grid search, metrics | **installed env has 1.6.1** (see issue E4) |
| XGBoost ≥2.0,<3 | gradient boosting | installed 2.1.3 |
| joblib | model persistence | `.pkl` |
| watchdog ≥3.0 | real-time FS events | optional at runtime (`HAS_WATCHDOG`) |
| numpy/pandas/scipy | data handling | — |

## 5. Backend, frontend, infrastructure

- **Backend:** `app.py`, single process; REST endpoints: `/api/status`, `/api/current-prediction`,
  `/api/live-events`, `/api/model-results`, `/api/detection-history`, `/api/start-monitoring`,
  `/api/stop-monitoring`, `/api/simulate/normal`, `/api/simulate/ransomware`.
- **Frontend:** `templates/index.html` + `static/dashboard.js` (poll every 1.5 s) + `static/style.css`.
- **Infrastructure:** local only; CI via GitHub Actions (`ci.yml`) runs `pytest` and an
  end-to-end training smoke test on Python 3.11/3.12.

> **UI vs. reality:** the dashboard badge states "5-fold CV + GridSearchCV"
> (`templates/index.html:163`), but the code uses **3-fold** `StratifiedKFold`
> (`ml/train.py:200`). The UI overstates the cross-validation count.

## 6. Security, reliability, performance, scalability

- **Security:** `app.run(host="0.0.0.0", ...)` binds to all interfaces with Flask's dev server and
  `debug=False`; no auth on the API or the state-changing POST endpoints — acceptable for a local
  demo, **not** production.
- **Reliability:** if `watchdog` is absent, monitoring silently produces zero events and always
  predicts NORMAL; no health warning beyond `watchdog:false` in `/api/status`.
- **Performance:** per-window inference in the sub-millisecond to ~2 ms range (measured, see
  results); the 1-s loop sleeps 1 s, so throughput is bounded by design, not by inference.
- **Scalability:** single monitored directory, single global model, in-memory buffer; no sharding,
  no queue, no horizontal scaling.

## 7. Implemented vs. planned / incomplete

| Feature | Status |
|---|---|
| Real-time create/rename/delete monitoring | **Implemented** |
| 1-s window aggregation | **Implemented** |
| Five classifiers + grid search + persistence | **Implemented** |
| Cost-sensitive model selection | **Implemented** |
| REST API + live dashboard + history | **Implemented** |
| Synthetic benign/ransomware simulators | **Implemented** (demo only) |
| `monitoring/` helper module | **Not implemented** (empty package) |
| `scripts/` utilities | **Not implemented** (empty directory) |
| Alerting/notification, auth, multi-directory | **Not implemented** |
| Evaluation on real data | **Not implemented** |

## 8. Error handling and known limitations

- Broad `try/except` around simulators; per-file cleanup in `finally`.
- `predict()` returns `("NORMAL", 0.0)` if no model — a silent failure mode.
- Simulation filenames are random; no deterministic seed in the live simulators.
- The 1-s window has no hysteresis: a single quiet window can flip the label to NORMAL.

## 9. Test coverage and reproducibility

- `tests/test_all.py`: dataset generation, cost matrix, full training, best-model persistence,
  file monitoring primitives, prediction sanity, all Flask endpoints, aggregation logic.
- **No test asserts detection quality**, because the synthetic task is trivial; tests assert
  plumbing, not efficacy.
- `ci.yml` reproduces install → tests → train on Python 3.11/3.12.
- Reproducibility caveat: `test_all.py` calls `train_models` which **overwrites**
  `data/model_results.json` and `reports/model_results.csv`; therefore the committed tables reflect
  the last **test** run (400 samples), not the documented 1500+1500 training run.

## 10. Implementation-to-research mapping

| Research objective | Supporting component | Evidence strength |
|---|---|---|
| Detect from 3 count features / 1-s window | `ml/train.py`, `app.py` | Strong (implemented) |
| Compare 5 classifiers | `ml/train.py:148-265` | Strong (implemented) |
| Cost-sensitive selection | `ml/train.py:50,120-122,269` | Strong (implemented) |
| Quasi real-time operation | `app.py` monitor loop + dashboard | Strong (implemented) |
| Real detection effectiveness | — | **Weak / unsupported** (synthetic only) |

## 11. Audit findings (numbered)

- **E1 (Critical):** evaluation data is synthetic and trivially separable → perfect scores are an
  artifact. No real detection capability is demonstrated.
- **E2 (Major):** results in README vs `model_results.json` disagree (test-set size, latencies).
- **E3 (Major):** "best model = Decision Tree" is an arbitrary tie-break among four cost-0 models.
- **E4 (Major):** environment pins `scikit-learn<1.6` but installed is 1.6.1; the pin's stated
  reason (XGBoost incompatibility) conflicts with a successful run.
- **E5 (Minor):** dashboard claims 5-fold CV; code uses 3-fold.
- **E6 (Minor):** `monitoring/` and `scripts/` are empty but described in README structure.
- **E7 (Minor):** no auth on state-changing endpoints; dev server bound to 0.0.0.0.
