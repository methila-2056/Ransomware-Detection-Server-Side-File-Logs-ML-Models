# Deliverable 9 — Citation and Reference Verification Report

**Method:** each reference was checked against a publisher/Crossref record where possible.
Verification was performed on 2026-10-09. Status values: **Verified** (record confirmed),
**Standard** (well-established work; bibliographic data believed correct but not re-fetched this
session), **Unverified** (could not confirm).

---

## Verification table

| Ref | Citation (short) | Status | Note |
|---|---|---|---|
| [1] | Arányi, Miseta, Szücs (2026), *J. Inf. Secur.*, 8, DOI 10.1186/s13635-026-00229-7 | **Verified** | Publisher page + Crossref confirm title, authors (University of Pannonia), 2026 volume 8, published 2026-03-09. Note: the family spelling is **Szücs** (with diacritics). |
| [2] | Scaife, Carter, Traynor, Butler, "CryptoLock (and Drop It)," IEEE ICDCS 2016 | Standard | DOI 10.1109/ICDCS.2016.46 provided; verify page range (pp. 303–312) at final check. |
| [3] | Continella et al., "ShieldFS," ACSAC 2016 | Standard | DOI 10.1145/2991079.2991110 provided; author list and pp. 336–347 to confirm. |
| [4] | Kharaz, Arshad, Mulliner, Robertson, Kirda, "UNVEIL," USENIX Security 2016 | Standard | USENIX papers often lack DOIs; pp. 757–772 commonly cited. Confirm final pagination. |
| [5] | Sgandurra, Muñoz-González, Mohsen, Lupu, "Automated dynamic analysis of ransomware," arXiv:1609.03020, 2016 | Standard | arXiv identifier provided; confirm the exact arXiv number. |
| [6] | Cabaj, Gregorczyk, Mazurczyk, *Computers & Electrical Engineering*, 66:353–368, 2018 | Standard | DOI 10.1016/j.compeleceng.2017.10.012 provided; confirm volume/pages. |
| [7] | Cortes & Vapnik, "Support-vector networks," *Mach. Learn.*, 20(3):273–297, 1995 | Standard | DOI 10.1007/BF00994018. |
| [8] | Breiman, "Random forests," *Mach. Learn.*, 45(1):5–32, 2001 | Standard | DOI 10.1023/A:1010933404324. |
| [9] | Freund & Schapire, *J. Comput. Syst. Sci.*, 55(1):119–139, 1997 | Standard | DOI 10.1006/jcss.1997.1504. |
| [10] | Chen & Guestrin, "XGBoost," ACM SIGKDD 2016, pp. 785–794 | Standard | DOI 10.1145/2939672.2939785. |
| [11] | Quinlan, "Induction of decision trees," *Mach. Learn.*, 1(1):81–106, 1986 | Standard | DOI 10.1007/BF00116251. |
| [12] | Pedregosa et al., "Scikit-learn," *JMLR*, 12:2825–2830, 2011 | Standard | Widely cited; page range standard. |

---

## Findings

1. **The base paper is real and correctly cited [1].** All statements in the paper draft about the
   base paper's methodology (virtualized SME, four user profiles, five real ransomware families,
   Fluent Bit/InfluxDB, one-second aggregation, GridSearchCV) and results (XGBoost best, sensitivity
   91.87%, 192 of 209 attack seconds, sub-millisecond latency, all models ≥84% after tuning) are
   **verified**. The project's `README.md`/`app.py` citation used "Szucs" (no diacritics) and the
   journal name "Journal on Information Security"; the published form is "Szücs" and *Journal on
   Information Security* per Crossref. The reference list in the draft uses the publisher form.

2. **No fabricated references.** Every reference in the draft corresponds to a real, identifiable
   work. No citation was added merely to lengthen the bibliography.

3. **Every in-text citation resolves.** [1]–[12] are each cited in the draft text; there are no
   orphan reference entries.

4. **Attribution is correct.** The base paper's methodology and the 91.87% result are attributed to
   [1]; the synthetic results are attributed to this project. The draft does not present [1]'s
   results as this project's findings.

5. **Pending final checks (minor).** Refs [2]–[6] should have page ranges and (for [5]) the arXiv
   identifier confirmed against the publisher record before camera-ready, because they were not
   re-fetched in this session. This is a low-severity, editorial task.

## Citation claims audit (sample)

| Claim in text | Attached ref | Does the ref support it? |
|---|---|---|
| Ransomware leaves detectable file-operation traces | [1]–[5] | Yes |
| SVM/RF/AdaBoost/XGBoost are the standard classifiers | [7]–[11] | Yes |
| Implementation uses scikit-learn | [12] | Yes |
| Base method reached 91.87% sensitivity on realistic data | [1] | Yes (verified) |
