# Deliverable 1 — Project Inventory and Research-Materials Audit

**Project:** Ransomware Detection from Server-Side File-Operation Logs
**Directory audited:** `C:\Users\methi\Documents\Ranomware Detection`
**Audit date:** 2026-10-09
**Auditor role:** Senior researcher / technical auditor

---

## 1. Summary of what the directory actually contains

The directory is a **single Python implementation project** (Flask + scikit-learn + watchdog) that
reproduces the methodology of one base paper. It is **not** a repository that also stores the base
paper PDF, separate reference papers, an early "initial prototype", or a conference call/template
beyond a generic IEEE Word template.

| Category | Present? | Notes |
|---|---|---|
| Base research paper (PDF/HTML) | **No (file absent)** | Cited in code/README by DOI only. The paper text is **not** in the directory. |
| Reference papers | **No** | No PDFs, `.bib`, or `.tex` files anywhere. |
| Initial (early) prototype | **No distinct copy** | Only one implementation; no earlier version preserved. |
| Full prototype / source code | **Yes** | `app.py`, `ml/train.py`, front-end, tests, CI. |
| Architecture diagrams | Partial | One Mermaid diagram in `README.md`; UI is code, no exported image. |
| Datasets | Yes (synthetic) | `data/training_dataset.csv` (400 rows). |
| Trained models | Yes | `models/*.pkl`, `models/best_model_info.json`. |
| Experiment results | Yes | `data/model_results.json`, `reports/model_results.csv`. |
| Documentation | Partial | `README.md` only; `docs/` is empty. |
| Conference instructions | Partial | `ieee (1).doc` = generic IEEE conference template (not a specific conference CFP). |

Original files were **not modified, moved, renamed, or overwritten**.

---

## 2. Complete file classification

### 2.1 Base research paper (cited, not stored)
- **Reference:** Aranyi, G., Miseta, T. & Szucs, V. (2026). *Ransomware detection based on
  server-side file operation logs using machine learning.* Journal on Information Security, 8.
  DOI: 10.1186/s13635-026-00229-7.
- **Evidence of use:** Cited in `README.md:14`, `app.py:4-6`, `ml/train.py:4-6`,
  `tests/test_all.py:3`, `templates/index.html:20-21`.
- **Limitation:** The actual paper content could not be read; all statements about the paper's
  internal detail are therefore **unverified** and must be attributed as "as reported by the
  implementation", not as first-hand reading.

### 2.2 Template / formatting instructions
- `ieee (1).doc` (97,280 bytes) — the standard *"Template for Preparation of Papers for IEEE
  Sponsored Conferences & Symposia"* (classic IEEEtran-style Word template: title, author block,
  abstract, numbered Roman section headings, two-column body, references in `[n]` style).
- **Extracted guidance used:** section numbering (I, II, III…), `Fig.` abbreviation, table titles
  above tables, figure captions below figures, references numbered in square brackets, American
  spelling, SI units, no underlining.

### 2.3 Full prototype — source code
| File | Format | Purpose | Evidence provided | Limitations |
|---|---|---|---|---|
| `app.py` | Python | Flask app: watchdog monitoring, 1-s window aggregation, prediction, SQLite logging, REST API, activity simulators | Core runtime pipeline | No rolling-window smoothing; single global model; no alerting back-end beyond UI |
| `ml/train.py` | Python | Synthetic data generators, cost function, 5-model training with `GridSearchCV`, model selection & persistence | Core experiment | Data is synthetic and trivially separable |
| `ml/__init__.py` | Python | Package marker | — | — |
| `monitoring/__init__.py` | Python | Empty package marker | — | Monitoring helpers described in README are **not implemented** here |
| `static/dashboard.js` | JavaScript | Dashboard polling, rendering, charts (vanilla JS) | UI live behaviour | — |
| `static/style.css` | CSS | Dashboard styling | — | — |
| `templates/index.html` | HTML | Dashboard layout | UI architecture | — |
| `tests/test_all.py` | Python | 6 unittest classes: dataset, cost matrix, training, monitoring, prediction, Flask API | Test coverage evidence | Training tests re-write `data/model_results.json` as a side effect |
| `.github/workflows/ci.yml` | YAML | CI on Python 3.11/3.12 + end-to-end training check | Reproducibility evidence | — |
| `setup.bat`, `run.bat`, `test.bat` | Batch | Install/train, run app, run tests | Reproducibility | Windows-only |

### 2.4 Trained model artifacts (`models/`)
- `decision_tree.pkl`, `random_forest.pkl`, `svm.pkl`, `adaboost.pkl`, `xgboost.pkl`
- `best_model.pkl` (copy of the selected model), `scaler.pkl`, `best_model_info.json`
- **`best_model_info.json` names "Decision Tree"** as the selected best model.

### 2.5 Data and results
- `data/training_dataset.csv` — 400 rows (200 normal, 200 ransomware), columns
  `create_count, rename_count, delete_count, label`. Ransomware rows count ~10–80 creates;
  normal rows ~0–8 creates → **strongly separated**.
- `data/model_results.json` — per-model metrics, ROC points, confusion matrices, best params.
- `reports/model_results.csv` — summary table of the same run.
- `data/detections.db` — SQLite log of live predictions (runtime artifact).

### 2.6 Runtime artifacts
- `demo_server/` — ~200 small `.txt` files generated by the live simulators (benign "doc_*"/"renamed_*"
  and rename residue). These are **test fixtures created at run time**, not research data.
- `__pycache__/`, `.pytest_cache/` — build/cache artifacts, not research evidence.

### 2.7 Unreadable / missing content
| Item | Problem | Action |
|---|---|---|
| Base paper full text | Not in directory | Request the PDF, or treat all paper-internal detail as unverified |
| Reference papers | None present | Literature review cannot be built from directory files |
| `docs/` | Empty directory | No documentation to extract |
| `scripts/` | Empty directory | Monitoring scripts referenced in structure are absent |
| Conference name / CFP | Unknown | Only a generic IEEE template is present |

---

## 3. Relationships between materials

```
Aranyi et al. (2026)  ──method──▶  ml/train.py  ──artifacts──▶  models/*.pkl + data/model_results.json
       (base paper,                    │
        cited only)                     ▼
                                   app.py ◀── templates/ + static/ (dashboard)
                                        │
                                        ▼
                                   data/detections.db + demo_server/ (live sim)
                                        │
                                        ▼
                                   tests/test_all.py  ◀── .github/workflows/ci.yml
```

The chain **base paper → training → model artifacts → live app → tests** is coherent and closed
within the repository. The only external dependency (the base paper itself) is referenced but absent.

---

## 4. Files that provide evidence for paper claims

| Paper claim (planned) | Supporting file(s) |
|---|---|
| Features are create/rename/delete per 1-s window | `ml/train.py:136-142`, `app.py:176-180` |
| Cost matrix C_FP=1, C_FN=10 | `ml/train.py:50`, `tests/test_all.py:57-69` |
| Five models trained with grid search | `ml/train.py:148-217`, `data/model_results.json` |
| Reported metrics / confusion matrices | `data/model_results.json`, `reports/model_results.csv` |
| Real-time monitoring + dashboard | `app.py`, `templates/index.html`, `static/dashboard.js` |
| Tests / reproducibility pipeline | `tests/test_all.py`, `.github/workflows/ci.yml` |

---

## 5. Blockers and uncertainties (must be resolved before submission)

1. **Target conference unidentified** — only a generic IEEE template exists. Formatting follows that
   template; page limit and specific CFP rules are unknown.
2. **Author names / affiliations / emails** are not present anywhere in the directory. The paper uses
   clearly marked placeholders rather than invented identities.
3. **Base paper full text unavailable** — its own reported numbers (dataset size, exact F1/accuracy,
   window size, feature set) could not be verified against the source.
4. **No real-world data** — all results come from a synthetic generator; the perfect scores are an
   artifact of separability and must not be presented as deployment performance.
5. **Internal inconsistency in reported numbers** — the `README.md` results table (e.g., "600 test
   samples", Decision Tree latency 0.90 ms, Random Forest 12.02 ms) does **not** match
   `data/model_results.json` (80 test samples; Decision Tree 0.126 ms, Random Forest 1.833 ms).
   The JSON/CSV are treated as the ground-truth artifacts because they are machine-generated.
