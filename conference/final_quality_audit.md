# Deliverable 10 — Final Quality Audit

> ## ⚠️ PARTIALLY SUPERSEDED
>
> This audit was written against an earlier state of the project. Several of its findings have been
> resolved and two of its conclusions no longer hold. What each finding is now is recorded below so
> the file cannot be misread as current.
>
> | Original finding | Status now |
> |---|---|
> | Perfect scores are a synthetic artefact (**Critical**) | **Resolved.** Benign profiles are transcribed from the base paper's Table 3 and the attack generator overlaps them. Recorded sensitivity is 0.963–0.978, cost 63–107. |
> | Best model is an arbitrary tie-break among cost-0 models (**Major**) | **Resolved.** `_select_best` orders by (cost, latency, −sensitivity, name) and the tie is reported explicitly. |
> | README disagrees with the results JSON (**Major**) | **Resolved.** README tables are generated from `results/` by `tools/update_readme.py`; no figure is typed by hand. |
> | `requirements.txt` pins conflict with the environment (**Major**) | **Resolved.** The `scikit-learn<1.6` cap was removed and the underlying xgboost tag incompatibility is handled at runtime by `ml/compat.py`. |
> | CSV/JSON came from a 400-row test run, not a training run (**Major**) | **Resolved.** `tests/test_all.py` no longer writes to `data/`, `models/` or `results/`; a CI gate hashes those files before and after the suite. |
> | UI badge overstates CV folds | **Resolved in code.** The badge reads "3-fold" where the code used 3-fold; the pipeline now uses 5-fold throughout and the badge matches. |
> | `monitoring/` and `scripts/` are empty | **Resolved.** `monitoring/agent.py` now contains hysteresis, abstention, canary ground truth and multi-share monitors. |
> | POST endpoints unauthenticated | **Resolved.** Token auth via `RANSOMWARE_API_TOKEN`, and the server binds to 127.0.0.1 by default instead of 0.0.0.0. |
> | `predict()` returns `("NORMAL", 0.0)` silently when unloaded | **Partly resolved.** Prediction failures are caught and logged, and `/api/explain/current` returns a specific "insufficient history" status rather than a shape mismatch. |
>
> **New risk introduced by the current work, not covered below:** nine JSON result files now exist in
> `results/`. Any statement in a write-up must be traceable to one of them, and the emulator
> provenance in `results/run_manifest.json` must accompany every quoted figure.

---

## Archived audit follows

**Scope:** the draft paper (`conference_paper_draft.md`) and its supporting deliverables, audited
against the actual implementation and artifacts on 2026-10-09.

Severity: **Critical** = invalidates a central claim / blocks submission; **Major** = needs
substantial correction or extra evidence; **Minor** = editorial/formatting.

---

## A. Technical accuracy against the implementation

| ID | Finding | Severity | Status in draft |
|---|---|---|---|
| A1 | Perfect scores are synthetic artefacts, not detection capability | Critical | **Addressed** — stated in abstract, V-A, VI, VII, VIII |
| A2 | "Best model" is an arbitrary tie-break among four cost-0 models | Major | **Addressed** — stated in VI and VIII |
| A3 | Dashboard claims 5-fold CV; code uses 3-fold | Minor | Draft correctly says three-fold; UI bug noted in audit |
| A4 | README latency/table numbers differ from `model_results.json` | Major | **Addressed** — draft uses JSON as authoritative; discrepancy listed in VIII |
| A5 | `requirements.txt` pins scikit-learn <1.6 but env uses 1.6.1 | Major | **Disclosed** in V-C and VIII |
| A6 | CSV/JSON correspond to the 400-sample test run, not the 1500+1500 training run | Major | **Disclosed** — draft states N=80 test set = 400-sample run |

## B. Research novelty and contribution clarity

| ID | Finding | Severity | Status |
|---|---|---|---|
| B1 | No new algorithm; must not claim novelty | Critical | **Addressed** — framed throughout as reproduction + testbed |
| B2 | Contributions clearly separated (inherited vs. engineering) | — | Satisfied (research_gap_and_contributions.md; Section I) |
| B3 | Base paper's 91.87% used as the honest benchmark | — | Satisfied (verified citation) |

## C. Literature coverage and citation integrity

| ID | Finding | Severity | Status |
|---|---|---|---|
| C1 | No reference papers present in the directory | Major | Documented; external background used and labelled |
| C2 | Base paper verified against publisher/Crossref | — | Satisfied |
| C3 | Refs [2]–[6] page ranges to confirm | Minor | Flagged in reference report |

## D. Experimental validity and statistical correctness

| ID | Finding | Severity | Status |
|---|---|---|---|
| D1 | Single run, no confidence intervals / seeds | Major | **Disclosed** in VIII; plan in experimental_evaluation_plan.md |
| D2 | Accuracy is uninformative here; cost + recall are reported | — | Addressed |
| D3 | ROC AUC = 1.0 provides no discrimination among models | Minor | Noted in VI, VIII |
| D4 | Latency measured from 20 single-sample calls on one machine | Minor | Described; not generalised |

## E. Methodological reproducibility

| ID | Finding | Severity | Status |
|---|---|---|---|
| E1 | Seeds fixed; CI re-runs pipeline on two Python versions | — | Satisfied |
| E2 | Environment/lock inconsistency (A5) | Major | Disclosed |
| E3 | Tests overwrite result artifacts as a side effect | Minor | Documented in audit |

## F. Academic writing, grammar, logical flow

| ID | Finding | Severity | Status |
|---|---|---|---|
| F1 | Abstract → contributions → method → results → conclusion are consistent | — | Satisfied |
| F2 | Performance adjectives all qualified by "synthetic"/"simulated" | — | Checked |
| F3 | No contractions; IEEE spelling conventions applied | — | Satisfied |

## G. Conference formatting and submission compliance

| ID | Finding | Severity | Status |
|---|---|---|---|
| G1 | Target conference unknown; only a generic IEEE template exists | Major | **Blocked** — see Open Items; formatting follows `ieee (1).doc` |
| G2 | Author names/affiliations absent | Major | Placeholders used (no invention) |
| G3 | Page limit unknown | Minor | Draft ≈ 6–8 pages; must be trimmed to the CFP limit |
| G4 | Two-column IEEE layout, `[n]` references, `Fig.`/`TABLE` conventions | — | Applied in the DOCX/PDF build |

## H. Plagiarism, attribution, unsupported claims

| ID | Finding | Severity | Status |
|---|---|---|---|
| H1 | Base paper attributed; no paraphrased passages | — | Satisfied |
| H2 | No unsupported superlatives; no fabricated data | — | Satisfied |
| H3 | All figures generated from real artifacts | — | Satisfied |

## I. Internal consistency (abstract ↔ contributions ↔ results ↔ conclusion)

| ID | Finding | Severity | Status |
|---|---|---|---|
| I1 | The numbers in the abstract match Table V | — | Verified (1.000 ×4; SVM 0.975; cost 10) |
| I2 | The 91.87% baseline appears consistently (abstract, II, VII) | — | Verified |
| I3 | Conclusion does not re-state the abstract verbatim | — | Satisfied |

## J. Missing evidence, limitations, unresolved placeholders

| ID | Finding | Severity | Status |
|---|---|---|---|
| J1 | `[REAL-WORLD RESULT REQUIRED]` | Major | Explicit placeholder (not disguised) |
| J2 | `[DRIFT RESULT REQUIRED]` | Major | Explicit placeholder |
| J3 | `[CROSS-SEED CI REQUIRED]` | Major | Explicit placeholder |
| J4 | `[SCREENSHOT REQUIRED]` (dashboard figure) | Minor | Figure omitted rather than faked |
| J5 | `[AUTHOR NAME]`, `[AFFILIATION]`, `[EMAIL]`, `[ACKNOWLEDGMENT]` | Major | Placeholders; must be completed by authors |

---

## Consolidated issue log

**Critical (must be resolved conceptually — done here by framing):**
- A1, B1 — both resolved by explicitly framing the work as a reproduction/evaluation study and
  labelling all perfect scores as synthetic artefacts. No critical claim is left unqualified.

**Major (cannot be resolved without new data / author input):**
- G1 target conference unknown; G2/J5 author information missing; J1–J3 real-world/drift/CI results
  absent; C1 no reference files; A2/A4/A5/E2 disclosed inconsistencies.
- These **prevent the paper from being declared "final camera-ready"** for the intended conference
  and prevent any real-world performance claim.

**Minor:**
- A3, C3, D3, D4, E3, G3, J4 — editorial or clearly bounded.

## Submission readiness statement

> The manuscript is **internally consistent, technically accurate with respect to the implemented
> prototype, and honest about its limits.** It is **not** submission-ready to a peer-reviewed venue
> while (i) the target conference and author details are undetermined, and (ii) the only quantitative
> results are synthetic. The paper is ready as a **reproducible technical report / student
> submission**, and ready for a security venue **once real-data experiments replace the synthetic
> placeholders** and the CFP formatting is confirmed.
