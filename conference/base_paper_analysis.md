# Deliverable 2 — Base Paper Analysis

**Base paper:** G. Arányi, T. Miseta, and V. Szücs, "Ransomware detection based on server-side file
operation logs using machine learning," *Journal on Information Security*, vol. 8, 2026, art. 8.
DOI: 10.1186/s13635-026-00229-7 (University of Pannonia; received 29 May 2025; accepted 6 Feb 2026;
published 9 Mar 2026).

> **Evidence status (corrected).** The full text of the base paper is stored in the project
> directory as `Base Paper.pdf` (18 pages). All content in this document was derived by reading
> that PDF directly, not from the publisher abstract. An earlier revision of this file stated
> that the PDF "is still not stored in the project directory" and marked fine-grained table values
> as unverified; that statement is obsolete and has been removed.
>
> **One inconsistency in the base paper was resolved by cross-checking its own tables.** Table 8
> reports total costs (RF 335, SVM 353, DT 212, ADA 272, XGB 175) that do not follow from applying
> C_FP=1 and C_FN=10 to Table 8's own FP and FN columns, which would give 3125, 3323, 2102, 2702
> and 1705. Table 8's FN column is exactly ten times Table 7's FN for the same models, so the
> published totals are C_FP·FP + C_FN·FN evaluated on Table 7's raw confusion matrix. The cost
> implementation in `ml/metrics.py` follows that reading and the test suite asserts it.

---

## 1. Research problem and motivation **[verified]**

The paper asks whether ransomware can be detected **quasi real-time from centrally stored server-side
file-operation logs**, i.e. without client-side instrumentation, process telemetry, or kernel drivers.
Motivation: endpoint protection is unreliable in environments with limited client defences, whereas
centralised file servers already record trustworthy create/rename/delete activity. The stated goal is
"lightweight, server-side monitoring combined with machine learning" using "only three simple file
operation metrics and binary classification."

## 2. Research objectives and questions **[verified]**

1. Can ransomware-like behaviour be separated from benign behaviour using three aggregate counts per
   time window (create, rename, delete)?
2. Which of five classical/ensemble models (Random Forest, Decision Tree, SVM, AdaBoost, XGBoost) is
   best under careful hyperparameter tuning?
3. Can the best tuned model exceed **90% accuracy** while detecting attacks in the **early** phase of
   execution, in a noisy environment with realistic user behaviour?

## 3. Method and assumptions **[verified, with project mapping]**

| Element | Base paper | This project |
|---|---|---|
| Features | create / rename / delete counts per window | identical |
| Window | 1-second aggregation | identical |
| Models | RF, DT, SVM, ADA, XGB | identical |
| Tuning | `GridSearchCV`, k-fold CV, stratified sampling | `GridSearchCV`, 3-fold stratified, recall scoring |
| Cost analysis | cost-sensitive evaluation (FP vs FN) | C_FP=1, C_FN=10 |
| Logging pipeline | Fluent Bit → InfluxDB, nanosecond timestamps | **watchdog observer**, in-memory deque (different) |
| Attack data | real ransomware samples (Ryuk, WannaCry, NotPetya, Lockbit, Teslacrypt) executed in a virtualized SME | behaviour-faithful emulator with Table 3 benign profiles and five modelled family profiles, labelled as emulated everywhere |

**Critical distinction:** the base paper obtains its positive class from **real ransomware samples
executed in a virtualized SME with four user profiles**, logged at nanosecond resolution. This project
replaces that entire pipeline with a hand-written random generator. The base paper's data is far closer
to reality; the project's data is trivially separable.

## 4. Dataset and evaluation setup **[verified]**

- Virtualized SME infrastructure; automated realistic user file operations; attacks by five real
  ransomware samples triggered at random times.
- Logging via Fluent Bit into InfluxDB; file operations aggregated **per second**.
- Stratified train/test splitting; hyperparameter optimisation with `GridSearchCV` + k-fold CV.
- The paper explicitly reports **naturally imbalanced** data, reflecting realistic deployment.

## 5. Reported results **[verified highlights]**

- All five models achieved "reliable detection performance"; **sensitivity exceeded 84% after tuning**
  across all models.
- **XGBoost was best**, with **sensitivity 91.87%** (detecting **192 of 209 attack seconds** in the
  test set) and prediction latencies **below 1 ms**.
- Detection occurred in the **early phases** in the majority of test cases, even under high noise.
- On the paper's own tuning table (Table 7), pre/post-tuning performance and confusion
  matrices are reported. Reading the PDF directly, the tuned confusion matrices are:
  RF TP178/FP25/TN158/FN31, SVM TP176/FP23/TN160/FN33, DT TP188/FP2/TN181/FN21,
  ADA TP182/FP2/TN181/FN27, XGB TP192/FP5/TN178/FN17. Pre-tuning sensitivity was
  substantially lower for several models (RF 84.21%, SVM 52.63%, ADA 70.34%), so tuning
  matters most for the weaker classifiers.

**Contrast with this project:** an earlier revision of this project reported 100% on every model
because the generator produced two linearly separable classes. That condition has been fixed.
The recorded run in `results/model_results.json` now shows accuracy 0.994–0.997, sensitivity
0.963–0.978 and cost 63–107 across the five models, which is the non-trivial range the paper
also reports. The perfect scores are gone; what replaces them is a comparison against the paper's
figures rather than against an artefact of the data generator.

## 5a. Structural limitation of the count-only feature set

Reading the paper's Table 4 sample clarifies why its sensitivity is 91.87% rather than higher.
The benign windows in that sample are (0,0,0), (1,0,0), (0,0,1) and (0,0,2) — at most two
operations per second — while attack windows reach (19,17,0) and (0,0,21). However one attack
window in the sample, (1,2,0), falls inside the benign envelope. An attack window and a benign
window with identical three counts are therefore indistinguishable to any classifier of those three
features, which places a ceiling on achievable sensitivity that no tuning can remove.

This is the gap the present project addresses with trailing temporal aggregates, and it is the
paper's own stated direction of travel: Sect. 6.3 names a shorter aggregation window as future
work, and Sect. 6.2 names count-only features' weakness against rate-limited encryption as the
principal limitation. Measured effect (`results/ablation.json`, 30,000 windows): counts alone
reach AUROC 0.947 with cost 5112; adding trailing aggregates raises AUROC to 0.999 and drops
cost to 127.

## 6. Strengths (of the base approach) **[verified]**

- Minimal, privacy-preserving feature set (three counts; no file contents).
- Quasi real-time feasibility with sub-millisecond inference.
- Formal, reproducible evaluation framework: latency measurement, ROC and precision–recall analysis,
  and cost-sensitive evaluation.
- Demonstrated robustness on imbalanced, noisy, realistic workloads and across ransomware families.

## 7. Limitations and gaps **[verified where stated, otherwise reasoned]**

1. **No truly public dataset:** the infrastructure is a virtualized SME; reproducibility depends on
   recreating that environment (Fluent Bit, InfluxDB, the samples).
2. **Count-only features** remain vulnerable to rate-limited ("low-and-slow") encryption, which is not
   evaluated.
3. **Single deployment scale** (one SME server); enterprise-scale behaviour is not reported.
4. **No concept-drift study** as user behaviour evolves over time.

Each of these four gaps is addressed by an implementation in this project, and each address ships
with a measured result rather than an assertion:

| Paper limitation | Addressed by | Measured result |
|---|---|---|
| 1 — no public dataset | trace generator with Table 3 profiles, version-controlled dataset and manifests | `results/run_manifest.json`, `results/dataset_report.json`; still emulated, not captured malware |
| 2 — count-only features vs low-and-slow | temporal aggregates, plus an explicit evasion suite | counts-only cost 5112 → 127 with trailing aggregates (`results/ablation.json`); detection collapses below ~25 ops/s and mimicry reaches only 0.822 (`results/evasion.json`) |
| 3 — single-server scale | multi-share monitors with independent buffers | `monitoring/agent.py`, `POST /api/shares` |
| 4 — no drift study | PSI drift monitor with retraining-recovery measurement | drift detected immediately; retraining cut cost 1131 → 95 without changing sensitivity (`results/drift.json`) |

Two further gaps are also closed: family-independence is claimed in Sect. 3.7 but never tested, so
a leave-one-family-out protocol was run (`results/zero_day_eval.json`, mean sensitivity drop 0.015 on
a wholly unseen family); and early-phase detection is claimed in the abstract without a figure, so
per-episode time-to-detect is now measured and reported in `results/model_results.json`.

## 8. Attribution: base paper vs. this project

| Aspect | Owner |
|---|---|
| Problem, features, window, five-model comparison, cost-sensitive framework, Fluent Bit/InfluxDB pipeline, real-sample evaluation, 91.87% XGBoost result | **Base paper (Arányi et al., 2026)** |
| Python/Flask re-implementation, `watchdog` integration, dashboard, trace generator with Table 3 profiles, tests, CI | **This project** |
| Numeric results in the project's tables | **This project** — measured on emulated data, not the paper's results |

## 9. Future work acknowledged or implied **[verified/implied]**

The paper positions the work as a lightweight complement to existing SIEM/EDR layers; natural
extensions are deployment at larger scale and additional behavioural signals. The project's own
future work (Section IX of the draft) aligns with these.
