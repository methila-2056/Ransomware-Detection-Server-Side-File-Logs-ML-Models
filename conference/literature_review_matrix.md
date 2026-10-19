# Deliverable 3 — Literature Review Matrix

> **Important scope note.** The project directory contains **no reference-paper files** (no PDFs,
> `.bib`, or `.tex`). A directory-based literature review is therefore impossible. This document
> instead (a) records that absence as an evidence gap and (b) provides a *general-background*
> matrix from well-known, independently verifiable works so that the paper can situate itself. Each
> entry is marked **[external]** and (unless the base paper) is **not** drawn from the project.
> None of these references reports this project's numbers and none may be cited as project evidence.

---

## 0. References present in the repository

| # | Citation | Present as file? | Status |
|---|---|---|---|
| B | Aranyi, Miseta, Szucs (2026), *J. Inf. Secur.*, 8, DOI 10.1186/s13635-026-00229-7 | **No** | Cited in code/README only |

No other references exist in the directory.

---

## 1. Grouping by theme

**Theme A — Host/FS behavioural ransomware detection (the project's family)**
**Theme B — Network/traffic-based ransomware detection**
**Theme C — Machine-learning algorithm foundations (the classifiers used)**

---

## 2. Matrix

### Theme A — Host / file-system behavioural detection

| # | Citation [external] | Problem | Method | Data / setup | Metrics reported by source | Strengths | Limitations | Relation to this project |
|---|---|---|---|---|---|---|---|---|
| A1 | N. Scaife, H. Carter, P. Traynor, K. Butler, "CryptoLock (and Drop It): Stopping Ransomware Attacks on User Data," *IEEE ICDCS*, 2016, pp. 303–312 | Stop ransomware at the file-system layer | Real-time entropy + I/O monitoring, file backup/drop | Real ransomware on Windows testbed | Detection within first few files; recovery rate | Early, practical, stops damage | Needs a driver/agent; entropy signal noisy for some formats | Same real-time filesystem-telemetry concept; project uses counts instead of content entropy |
| A2 | A. Continella et al., "ShieldFS: A Self-healing, Ransomware-aware Filesystem," *ACSAC*, 2016, pp. 336–347 | Transparent detection + rollback | File-system minifilter with process-level I/O + entropy features, shadow copies | Real ransomware on Windows | Detection/rollback vs. 8 families | Strong accuracy, self-healing | Kernel complexity, Windows-only | Project performs statistical detection without rollback; simpler, less invasive |
| A3 | A. Kharaz, S. Arshad, C. Mulliner, W. Robertson, E. Kirda, "UNVEIL: A Large-Scale, Automated Approach to Detecting Ransomware," *USENIX Security*, 2016, pp. 757–772 | Detect ransomware in sandbox | Dynamic analysis: sequence of filesystem operations + screen-lock detection | Real samples (181 ransomware + 320 benign) | Accuracy; ~96.3% detection claimed by source | Works on unseen samples | Sandbox-only, not live | Closest to the project's operation-sequence idea, but offline/isolated |
| A4 | D. Sgandurra, L. Muñoz-González, R. Mohsen, E. C. Lupu, "Automated Dynamic Analysis of Ransomware," arXiv:1609.03020, 2016 | Automated dynamic feature extraction | 20 features from file I/O, registry, API calls; ML classifiers | Real samples in sandbox | Classification accuracy reported by source (high) | Rich feature set | Sandbox, offline | Project uses only 3 aggregate features → far lighter but less expressive |
| A5 | **Base paper** Aranyi, Miseta, Szucs (2026) **[cited, not present]** | Ransomware detection from server-side file-operation logs | Count features per window + 5 classifiers + cost matrix | (Unverified) **as reported** | (Unverified) | Server-side, low-overhead framing | Cannot be inspected here | This project is a direct implementation/reproducibility study of A5 |

### Theme B — Network / traffic-based detection

| # | Citation [external] | Problem | Method | Data / setup | Metrics reported by source | Strengths | Limitations | Relation to this project |
|---|---|---|---|---|---|---|---|---|
| B1 | K. Cabaj, M. Gregorczyk, W. Mazurczyk, "SDN-based crypto-ransomware detection using HTTP traffic characteristics," *Computers & Electrical Engineering*, 2018 | Network-level crypto-ransomware detection | SDN + HTTP flow statistics | Live SDN testbed with real malware | Detection of crypto-ransomware traffic | No endpoint agent | Only network-observable ransomware | Different telemetry layer; the project uses server-side FS logs |

### Theme C — Classifier algorithm foundations (used by the project)

| # | Citation [external] | Contribution used in the project |
|---|---|---|
| C1 | C. Cortes, V. Vapnik, "Support-vector networks," *Machine Learning*, 20(3):273–297, 1995 | SVM classifier (calibrated RBF) |
| C2 | L. Breiman, "Random forests," *Machine Learning*, 45(1):5–32, 2001 | Random Forest classifier |
| C3 | Y. Freund, R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting," *J. Computer and System Sciences*, 55(1):119–139, 1997 | AdaBoost classifier |
| C4 | T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *ACM SIGKDD*, 2016, pp. 785–794 | XGBoost classifier |
| C5 | J. R. Quinlan, "Induction of decision trees," *Machine Learning*, 1(1):81–106, 1986 | Decision Tree baseline |

---

## 3. Agreements, disagreements, and unresolved problems

- **Agreement:** all host/behavioural works (A1–A4) confirm that file-operation behaviour carries a
  usable ransomware signal, which supports the project's premise.
- **Disagreement on feature richness:** A1/A2 use content entropy and per-process context; A3/A4 use
  dozens of features; the project uses **three** aggregate counts. This is a deliberate
  simplification inherited from A5 but increases vulnerability to low-and-slow evasion.
- **Evaluation disagreement (critical):** published works (A1–A4) evaluate on **real ransomware**.
  This project evaluates on a **synthetic generator**. No number from the project is comparable to
  A1–A4 accuracy figures, and none should be presented as such.
- **Unresolved:** none of the surveyed works, nor this project, demonstrates robustness to drift or
  to adversarial stays below a fixed count threshold. That remains the central open problem.

---

## 4. Logical research gap (established, not manufactured)

1. Server-side file-operation telemetry is a credible, low-cost detection surface (supported by the
   host/behavioural literature and the base paper).
2. Existing real-world evaluations require **real malware** and heavier instrumentation; independent
   **open-source, reproducible reproductions** of the server-side approach are scarce.
3. A gap therefore exists for a **reproducible engineering testbed + transparent evaluation** that:
   - implements the count-based pipeline end-to-end,
   - exposes its cost-sensitive decision rule, and
   - **explicitly characterises where the approach fails** (synthetic separability, drift, evasion).

**This is an engineering/reproducibility gap, not a claim of a new detection algorithm.** The paper
is framed accordingly.
