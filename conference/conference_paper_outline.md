# Deliverable 7 — Conference Paper Outline

**Format:** IEEE conference template (as provided in `ieee (1).doc`).
**Working title:** *A Reproducible Live Testbed for Ransomware Detection from Server-Side File-
Operation Logs: Implementation and Cost-Sensitive Evaluation of Five Machine-Learning Classifiers*

**Framing decision (from the novelty audit):** the paper is a **reproduction + evaluation study**.
It must not claim a new algorithm or state-of-the-art performance. Perfect synthetic scores are
presented explicitly as an artifact of separability.

**Author block:** `[AUTHOR NAME]`, `[AFFILIATION]`, `[EMAIL]` — placeholders, not invented.

---

## Section plan

| # | Section | Purpose | Key points | Supporting files | Figures/Tables | Evidence gaps |
|---|---|---|---|---|---|---|
| — | Title | Identify work | Reproduction + testbed framing | — | — | — |
| — | Abstract | 150–250 words | Problem, method, synthetic result, honest limitation | All | — | — |
| — | Keywords | Indexing | ransomware detection; file-operation logs; cost-sensitive learning; scikit-learn; anomaly detection | — | — | — |
| I | Introduction | Motivation & contribution | Server-side telemetry; why counts; contribution = reproducible testbed; contributions list | README, app.py | Fig. 1 (architecture) | — |
| II | Related Work | Situate | Host behaviour (CryptoLock, ShieldFS, UNVEIL, EldeRan), network (Cabaj), ML foundations; base paper | literature matrix | Table I (comparison) | Reference list must be verified; base paper absent |
| III | Method | Define approach | 3 features, 1-s window, labels, cost matrix, 5 models, grid search (recall), model selection | ml/train.py | Eq. (1) cost; Table II (models & grids) | Paper-internal detail unverified |
| IV | System Design & Implementation | Show engineering | watchdog → buffer → window → predict → SQLite → Flask/UI; API list; tech stack | app.py, index.html, dashboard.js | Fig. 1/2; Table III (stack) | `monitoring/`, `scripts/` empty |
| V | Experimental Setup | Reproducibility | Synthetic generator (profiles/families), 80/20 stratified, 3-fold CV, seeds, environment | ml/train.py, ci.yml | Table IV (data ranges) | Environment pin conflict |
| VI | Results | Report **actual** numbers | Per-model metrics on 80-sample synthetic test set; ROC; latency/train time; cost | model_results.json | Table V; Fig. 3 (metrics), Fig. 4 (ROC) | Single run, synthetic |
| VII | Discussion | Interpret honestly | Perfect separation explained by generator; tie-break arbitrary; SVM 1 FN | analysis | — | Real-world validation needed |
| VIII | Limitations & Threats | Disclose | Synthetic data, no drift/evasion, single host, no CIs, dev server | audit | Table VI | — |
| IX | Conclusion & Future Work | Close | Reproducible baseline; future = real logs, features, drift | plan | — | — |
| — | Acknowledgment | (optional) | Sponsor if any — omit if none | — | — | — |
| — | References | IEEE style | Base paper + verified external refs | reference report | — | — |

---

## Figures and tables required (Phase 9)

| Item | Type | Source | Section | Caption intent |
|---|---|---|---|---|
| Fig. 1 | Architecture | Original (from `app.py`/README) | IV | End-to-end pipeline: events → window → model → log/UI |
| Fig. 2 | Dashboard screenshot | `[SCREENSHOT REQUIRED]` | IV | Live monitoring UI |
| Fig. 3 | Bar chart of metrics | `model_results.json` | VI | Accuracy/precision/recall/F1 per model (test N=80) |
| Fig. 4 | Cost by model | `model_results.json` | VI | Total cost; SVM=10, others=0 |
| Fig. 5 | Feature scatter | `training_dataset.csv` | V | Create vs. delete counts showing separability (this *is* the key finding) |
| Table I | Related-work comparison | Literature matrix | II | Methods, data, limitations |
| Table II | Model config & grids | `ml/train.py` | III | Classifiers and search spaces |
| Table III | Software stack | `requirements.txt` | IV | Versions/roles |
| Table IV | Synthetic data ranges | `ml/train.py` | V | Benign vs. ransomware count ranges |
| Table V | Results | `model_results.json` | VI | Full metrics per model |
| Table VI | Threats to validity | Audit | VIII | Risk / mitigation |

**Figure policy:** every figure is generated from real project artifacts. No fabricated curves.
Screenshot (Fig. 2) is optional; if unavailable it is omitted rather than simulated.

---

## Consistency contract (check before finalizing)

- Abstract numbers = Table V numbers = `model_results.json`.
- The words "synthetic" and "simulated" must qualify every performance statement.
- No novelty/superiority adjective is left unqualified.
- Every `[x]` citation resolves to a reference entry and vice versa.
