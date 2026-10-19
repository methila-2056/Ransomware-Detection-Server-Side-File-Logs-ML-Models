# Deliverable 6 — Experimental Evidence and Evaluation Plan

This document separates (i) what the current artifacts **actually demonstrate**, (ii) what they
**cannot** support, and (iii) a reproducible plan for the missing evidence. Proposed experiments are
labelled **PLAN** and are never presented as completed.

---

## 1. Evidence currently in the repository

| Artifact | What it is | What it can support | What it cannot support |
|---|---|---|---|
| `data/training_dataset.csv` (400 rows) | Synthetic, generator-defined ranges | Feature definition; separability analysis | Real-world performance |
| `data/model_results.json` | 80-sample test metrics for 5 models | Plumbing correctness; cost-matrix mechanics | Generalisation claims |
| `reports/model_results.csv` | Same run, summary | Reproducing the table | Latency benchmarking (single machine, 20 reps) |
| `tests/test_all.py` | Unit/integration tests | Correctness of pipeline | Detection efficacy |
| `data/detections.db`, `demo_server/*` | Runtime logs from live simulators | Demo that the live loop works | Real behaviour |

**Verdict:** the current evidence supports the *engineering* claims (B) only.

---

## 2. Claims that ARE supported now

- C1: The three-feature, 1-second-window pipeline is implemented end-to-end. *(code + tests)*
- C2: On the synthetic dataset, all five classifiers separate the classes at ≥0.975 recall.
  *(`model_results.json`)* — a statement **about the synthetic data**, not about real ransomware.
- C3: The cost-sensitive rule selects cost 0 for four models and 10 for the SVM. *(artifacts)*
- C4: The live loop classifies windows and logs them; the dashboard reflects state. *(code + DB)*

## 3. Claims that REQUIRE new experiments

- R1: real detection performance; R2: robustness to drift/evasion; R3: advantage of any specific
  model; R4: deployment cost/throughput; R5: value of additional features.

---

## 4. `[EXPERIMENTAL RESULT REQUIRED]` placeholders used in the paper

| Placeholder | Meaning |
|---|---|
| `[REAL-WORLD RESULT REQUIRED]` | Metrics on real file-operation logs |
| `[DRIFT RESULT REQUIRED]` | Metrics under behaviour drift |
| `[EVASION RESULT REQUIRED]` | Metrics under rate-limited ransomware |
| `[CROSS-SEED CI REQUIRED]` | Mean ± 95% CI across ≥10 seeds |

---

## 5. Reproducible evaluation plan (**PLAN**)

### 5.1 Research questions
- **RQ1 (reproduction):** Do the five classifiers reproduce the paper's relative ranking under the
  cost matrix?
- **RQ2 (realism):** How does performance change when the synthetic data is replaced by real
  server-side file-operation logs?
- **RQ3 (robustness):** At what encryption rate does the count threshold stop detecting ransomware?
- **RQ4 (feature value):** Does adding timing/path features reduce cost under drift? (ablation)

### 5.2 Datasets
- **D1 (current):** synthetic generator, 2000+2000 balanced.
- **D2 (PLAN):** real server-side file-operation logs (benign + ransomware). Candidate sources to be
  identified and license-checked before use. **No dataset is currently available in the directory.**
- Preprocessing: parse events → 1-s windows → `(cc, rc, dc, label)`; identical to training.

### 5.3 Splits & protocol
- Stratified 80/20 hold-out **and** 5-fold stratified CV.
- **Group-aware split if D2 contains multiple hosts/attacks** (no window from the same attack in both
  train and test) — mandatory to avoid optimistic bias.
- ≥10 seeds; report mean ± 95% CI.

### 5.4 Baselines
- The five classifiers (same hyperparameter grids); a majority-class and a threshold-on-counts
  heuristic baseline.

### 5.5 Metrics
- Security-relevant: recall (sensitivity), precision, false-negative rate, total cost `1·FP + 10·FN`,
  ROC AUC (threshold-independent).
- Operational: detection latency (windows to first alert), windows/second throughput.
- Report **cost normalised per 1,000 windows** for comparability.

### 5.6 Environment
- Python 3.11/3.12; scikit-learn 1.3–1.5; XGBoost 2.x; record CPU/RAM and OS. Deterministic seeds.

### 5.7 Ablations
- Feature ablation: drop each of create/rename/delete; add timing features.
- Window ablation: 1 s vs 2 s vs 5 s.
- Cost sensitivity: C_FN/C_FP ∈ {1, 5, 10, 50}.

### 5.8 Threats to validity
- **Construct:** counts may not capture content-agnostic multi-format encryption.
- **Internal:** simulator parameters may leak class identity; D2 group-aware split mitigates.
- **External:** single-server scope; Windows-centric event semantics.
- **Statistical:** single-run results are point estimates; CIs required.
- **Reproducibility:** pin all versions; the current `requirements.txt` pin conflicts with the
  installed environment (see audit issue E4) and must be reconciled.

---

## 6. Minimum viable addition before submission

If time is limited, the single highest-value addition is **D2 + a group-aware, multi-seed evaluation
with cost curves**. Everything else improves rigour but does not change the paper's core message.
