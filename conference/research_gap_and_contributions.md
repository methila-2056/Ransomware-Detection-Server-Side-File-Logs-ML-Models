# Deliverable 5 — Research Contribution and Novelty Assessment

**Method:** the base paper, the external literature, and the actual implementation are compared
component by component. Every claim below is labelled with its evidence and confidence.

---

## 1. Contribution classification

### A. Inherited from existing literature (NOT this project's contribution)
| Item | Owner |
|---|---|
| Server-side file-operation telemetry as a detection surface | Base paper (Aranyi et al., 2026) and host-behavioural literature |
| Feature design: counts of create/rename/delete per time window | Base paper |
| 1-second aggregation window | Base paper (as implemented) |
| Five-model comparison (DT, RF, SVM, AdaBoost, XGBoost) | Base paper / standard ML |
| Cost matrix C_FP=1, C_FN=10 | Base paper (tests reference "paper Table 8") |

### B. Engineering / implementation contributions (defensible, this project)
1. **End-to-end open-source testbed** wiring real OS file events (watchdog) to a live classifier and
   a dashboard — the paper's abstract pipeline made executable and runnable in one command.
2. **Real-time evaluation harness** (1-s loop, SQLite history, REST API, UI) that turns a static
   study into an inspectable live system.
3. **Reproducible experiment pipeline** (`GridSearchCV`, fixed seeds, artifact persistence) plus a
   CI smoke test that re-trains the full pipeline on two Python versions.
4. **Synthetic activity simulators** for benign and ransomware-like behaviour used to exercise the
   live path.

### C. Adaptations / improvements (modest)
- Calibrated SVM wrapper (`CalibratedClassifierCV`) so SVM yields probabilities comparable to the
  tree ensembles for the `predict_proba` path.
- Model-agnostic, cost-driven selection with latency and training-time instrumentation alongside the
  cost matrix.

### D. Potentially novel technical / methodological contributions
**None established.** The pipeline is a reproduction. No new algorithm, feature, or decision rule is
introduced that is demonstrably beyond the base paper.

### E. Contributions still requiring experimental validation
| Claim candidate | Validation needed |
|---|---|
| The testbed detects ransomware-like behaviour in real time | Run against real ransomware traces / a real server workload |
| The three-feature model generalises beyond synthetic data | Held-out real file-operation logs |
| The system is robust to behaviour drift / evasion | Low-and-slow and threshold-aware attack scenarios |
| Cost-sensitive selection yields operationally better security | ROC/cost curves on realistic class priors |

---

## 2. Closest related work and differences

| Proposed item | Closest approach | Difference |
|---|---|---|
| Count-feature server-side detection | Base paper (Aranyi et al., 2026) | This project is an **implementation/reproduction**, not a new method |
| Real-time FS behaviour monitoring | CryptoLock/ShieldFS (Scaife et al.; Continella et al.) | Those use content entropy + kernel drivers; the project uses 3 counts in user space |
| Operation-sequence detection | UNVEIL (Kharaz et al.) | UNVEIL uses sandboxed event sequences; the project uses live aggregate counts |
| Five-classifier cost-sensitive comparison | Standard ML benchmarking | Ordering/selection is a reproduction of the base paper |

---

## 3. Evidence required to substantiate each candidate claim

| Claim | Evidence that would substantiate it | Exists now? |
|---|---|---|
| "We reproduce the pipeline" | Source code + tests + artifacts | **Yes** |
| "Models achieve perfect separation" | `model_results.json` (synthetic) | **Yes, but synthetic** |
| "It works on real ransomware" | Real-sample evaluation | **No** |
| "It is more accurate/faster than prior work" | Head-to-head benchmark on common data | **No — must not be claimed** |
| "It is novel" | A method not present in prior work | **No** |

---

## 4. Honest novelty statement for the paper

> This work contributes a **reproducible, open implementation and live testbed** of the server-side
> count-based ransomware-detection method of Aranyi et al. (2026), together with a transparent
> cost-sensitive evaluation and an explicit analysis of the limitations of synthetic evaluation.
> **It does not claim a new detection algorithm or state-of-the-art accuracy.** The perfect scores
> reported here are a property of the synthetic generator and are presented as such.

---

## 5. Recommended realistic research direction (if a stronger contribution is desired)

To move from "reproduction" to "contribution", the minimum credible upgrade is:

1. Obtain **real, timestamped server-side file-operation logs** (benign + ransomware) or a public
   equivalent, and re-evaluate.
2. Add **at least one feature family** that is not a raw count — e.g., extension-rename ratio,
   per-window path diversity, or inter-arrival timing — and run an **ablation** to isolate its value.
3. Add a **drift/evasion experiment** (rate-limited encryption) to show where the count threshold
   breaks.
4. Report **confidence intervals across seeds** and **cost curves**, not single-run point estimates.

Until step 1 is done, the paper must remain a reproduction/evaluation study.
