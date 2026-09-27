# Research Protocol: Multimodal MRI Fusion for Psychosis Classification via Multiple Kernel Learning

| | |
|---|---|
| **Status** | DRAFT v1.1 (2026-09-26). Becomes locked when the lock record in §14 is completed, before any model is fitted for confirmatory purposes. A preliminary functional-only analysis has already been run and is disclosed in §15. |
| **Dataset** | COBRE (Center for Biomedical Research Excellence) |
| **Configuration that is part of this protocol** | `configs/model_config.yaml`, `configs/cv_config.yaml`, `configs/data_config.yaml`, `configs/seeds.txt` |
| **Traceability** | The SHA-256 of this file and of every config file is written to `results/metrics/run_info*.json` by each analysis run. |
| **What changed in v1.1** | The functional data are the verified NIAK preprocessed release rather than fMRIPrep output; the raw T1 scans are still being obtained; the sample-definition rule is now a retained-scan-time rule; observed sample counts, measured runtimes and the preliminary exploratory result are recorded. See §13 and Appendix A. |

---

## 0. Decisions to confirm before lock

Each item has a value the code already implements. Confirm or change each, then complete §14.

| # | Decision | Current value | Status | Set in |
|---|---|---|---|---|
| D1 | Patient definition (primary) | All COBRE patient-group participants. Strict schizophrenia only as a sensitivity analysis. | The NIAK phenotype file carries no diagnostic subtype, so the strict-schizophrenia sensitivity analysis needs the official COBRE phenotype file | `data_config.yaml: label_map` |
| D2 | Minimum meaningful ΔAUC (δ) | 0.02 | **to confirm** | `cv_config.yaml: inference.min_meaningful_delta_auc` |
| D3 | Processing route: functional | Verified precomputed derivatives: COBRE preprocessed with **NIAK 0.12.14** (figshare 1160600) | settled by data availability (§13) | `data_config.yaml: niak` |
| D3b | Processing route: structural | FreeSurfer (or FastSurfer) on the raw COBRE T1 scans, once access is granted | **pending data** | `data_config.yaml: software` |
| D4 | Functional atlas | Schaefer 2018, 100 parcels, 7 networks → 4,950 edges | in use; atlas-space caveat in §3.3 | `data_config.yaml: functional.atlas` |
| D5 | Structural features | Desikan–Killiany thickness (68) + surface area (68) + 14 subcortical volumes = 150, each area/volume ÷ eTIV | **depends on D3b.** FastSurfer produces the DKT atlas (31 regions per hemisphere ≈ 138 features) | §3.2 |
| D6 | Decision threshold | Youden's J on pooled inner out-of-fold scores | in use | `model_config.yaml: threshold` |
| D7 | Feature selection | None in the primary analysis | in use | §5.5 |
| D8 | Functional QC rule | **≥ 180 s of usable scan time** after NIAK's scrubbing (primary); ≥ 120 s, NIAK's own criterion, as a labelled sensitivity analysis | **to confirm**; consequences in §2.3 | `data_config.yaml: niak.qc` |
| D9 | Confound set | Functional arm: age, sex, mean FD. Extended to eTIV and surface holes when structural data arrive. | tSNR is unavailable (§3.3) | `data_config.yaml: niak.confound_columns` |

---

## 1. Objective and hypotheses

**Objective.** Classify participants with a schizophrenia-spectrum diagnosis versus healthy controls from structural and resting-state functional MRI, and estimate whether fusing the two modalities with multiple kernel learning (MKL) improves on single-modality models.

**Primary hypothesis (H1).** MKL provides incremental discrimination over the prespecified functional-connectivity model.

- **Estimand:** the mean, over the 50 outer test folds (10 repeats × 5 folds), of the paired difference ΔAUC = AUC(MKL) − AUC(functional). Both AUCs are computed on the same test participants.
- **Why the functional model is the comparator:** it was chosen before any results were seen. The comparator is *not* "the best unimodal model", because picking that after seeing results is a data-dependent choice that would bias the comparison.
- **Decision rule:** §7.3. Superiority is supported only when the difference is positive, distinguishable from the structural-block null, and at least δ = 0.02 in size.
- **Prerequisite:** H1 cannot be tested until the structural features exist (D3b).

**Secondary hypothesis (H2).** MKL outperforms early concatenation, equal-weight kernel averaging, late-fusion stacking, and the structural-only model (§7.4).

- MKL can lower the weight of a less-informative modality. Whether it does so here is *described* using the distribution of the learned weights β across folds (§7.5).
- H2 is not treated as a causal claim about *why* one fusion strategy performs better.

---

## 2. Data

### 2.1 Sources and access
The analysis draws on two components of the same COBRE sample.

| Component | Source | Status | Terms |
|---|---|---|---|
| **Functional (Xf)** | Bellec, P., *COBRE preprocessed with NIAK 0.12.4*, figshare article 1160600, version 15 (published 2015-01-31, modified 2023-05-31): 146 participants (72 patients, 74 controls), preprocessed resting-state BOLD plus per-participant confound files and a phenotype table. Archive SHA-256 `b4fcb4a1572ecd3cd4434ff495cc13880f84222caf01b7cac3bdd7344e1ef816`, verified 2026-09-27. | **held** | CC BY-NC; cite INDI/COBRE and the release |
| **Structural (Xs)** | Raw T1w (multi-echo MPRAGE) from the COBRE release on NITRC/INDI (`COBRE_scan_data.tgz`), or the same data through SchizConnect/COINS | **access requested** | CC BY-NC; NITRC account plus project membership |
| Official phenotype (`COBRE_phenotypic_data.csv`) | NITRC/INDI | pending | needed for diagnostic subtype (D1) |
| Not used | BASC connectomes (figshare 1450804) | held | precomputed with a different atlas; would replace the prespecified feature definition |

- `nilearn.datasets.fetch_cobre` was removed in nilearn 0.9 and is not used.
- `data/README.md` records source URLs, access dates, release versions and SHA-256 checksums.

### 2.2 Site and acquisition
- COBRE is single-site and single-scanner: a 3 T Siemens Trio at the Mind Research Network. Resting-state: TR = 2 s, TE = 29 ms, flip angle 75°, 32 slices, 3 × 3 × 4 mm voxels, 150 volumes (5 minutes). T1: 5-echo MPRAGE, 1 mm isotropic.
- Because the functional data are preprocessed derivatives, there are no BIDS sidecars to audit. Acquisition parameters come from the release README and, once available, from `COBRE_parameters_rest.csv` and `COBRE_parameters_mprage.csv`.
- The analysis estimates out-of-sample performance for participants from the same COBRE setting. It does not establish transportability to other hospitals, scanners, populations or recruitment settings.

### 2.3 Primary sample-definition rule
A participant enters the primary cohort when all of the following hold:
1. a valid diagnostic label (§2.4);
2. recorded age and sex;
3. a usable resting-state run: **at least 180 s of scan time remaining after NIAK's scrubbing** (D8);
4. *(from the multimodal stage onward)* a T1 scan with a FreeSurfer reconstruction that completed and passed visual QC, rated blind to diagnosis.

Criterion 4 applies only once the structural data arrive; the functional-only analysis (§15) uses criteria 1–3.

**Observed counts** (`notebooks/03b`, known before lock; these are sample-definition counts, not outcomes):

| Step | n | patients | controls |
|---|---|---|---|
| Participants in the NIAK release | 146 | 72 | 74 |
| **Usable: ≥ 180 s retained (D8 primary)** | **82** | **31** | **51** |
| Usable: ≥ 120 s retained (sensitivity) | 101 | 43 | 58 |
| Structural available | pending | | |
| **Final intersection** | pending | | |

**This rule has a cost that must be stated in the paper.** Patients moved more, so the 180 s rule removes 41 of 72 patients (57%), against 23 of 74 controls (31%). The surviving sample is motion-matched (mean FD 0.203 mm in controls vs 0.209 mm in patients) but is no longer representative of COBRE's patients: it keeps those who could stay still. The 120 s rule keeps more participants but readmits a motion difference between groups. Both are reported (§8, §15).

**Exclusion order.** Reasons are applied in a fixed order and each participant is counted under the first that applies. The manifest keeps every reason. Reported as `results/metrics/sample_flow.csv`:

| Step | n |
|---|---|
| Initial N (phenotype file) | |
| Excluded: missing label | |
| Excluded: non-analysable label (e.g. disenrolled) | |
| Excluded: missing age/sex | |
| Excluded: insufficient retained functional scan time | |
| Excluded: missing structural data | |
| Excluded: failed structural QC | |
| **Final intersection N** | |
| Patients | |
| Controls | |

### 2.4 Diagnostic labels
- The mapping from raw phenotype values to labels (1 = patient, 0 = control) is set in `configs/data_config.yaml`, filled in after the audit prints the raw values and before features are linked to labels in any model.
- **Value handling.**
  - *Missing values* — empty cells, NaN, `None`, and the tokens in `missing_values` such as "unknown" and "not reported" — are treated as missing and give an exclusion reason.
  - *Matching* ignores case and whitespace, and compares numeric codes as integers (the NIAK phenotype stores diagnosis and sex as floats).
  - *Unlisted values:* a non-missing value the config does not list stops the audit; it is never silently treated as missing.
  - *Known non-analysable values*, such as disenrolled, are listed under `label_map.exclude` and reported as their own exclusion reason.
- **Primary analysis:** all patient-group participants (D1).
- **Sensitivity analysis:** strict schizophrenia only, once the official phenotype file provides subtypes.

### 2.5 Missing data
- No imputation anywhere.
- A participant missing a required element (§2.3) is excluded and the reason reported.
- Medication data, if available, are descriptive only (§8), so missing medication never causes exclusion.

---

## 3. Data processing and features

### 3.1 Processing rule
- **Functional:** verified precomputed derivatives, **NIAK 0.12.14** (figshare 1160600). The protocol's earlier route (fMRIPrep) is not used; see §13.
- **Structural:** a single documented pipeline (FreeSurfer, or FastSurfer where GPU time is required) run on the raw T1 scans, with the version pinned in `configs/data_config.yaml`.
- **What is recorded:** pipeline versions, atlas version, confound strategy, QC thresholds and the feature-construction code (this repository).
- **What is not allowed:** preprocessing choices are never changed based on predictive performance. Any change after the lock is a deviation (§13).

### 3.2 Structural features (Xs) — pending data
**Features (D5):** mean cortical thickness and surface area for each cortical region, plus 14 subcortical volumes from `aseg` (bilateral thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens).
- With FreeSurfer's Desikan–Killiany atlas: 68 + 68 + 14 = **150** features.
- With FastSurfer's DKT atlas: 62 + 62 + 14 = **138** features.
- The actual count is reported after extraction, not assumed.

**Head-size adjustment:** surface area and volumes are divided by eTIV. This uses only the participant's own eTIV, so it cannot leak across folds. Thickness is left unadjusted.

**Quality variable:** `SurfaceHoles` from `aseg.stats`, a linear function of the Euler number, is recorded as a quality covariate and enters the confound set.

**Multi-echo note:** COBRE's MPRAGE has 5 echoes. If the released file contains all echoes, they are combined (root-mean-square) into one volume before reconstruction; this is recorded.

### 3.3 Functional features (Xf), `notebooks/03b`
**What the release already did** (not repeated by us): slice-timing and motion correction; coregistration to each participant's T1 and non-linear normalization to **MNI152 2009a symmetric**, resampled to **3 mm**; **scrubbing** of volumes with FD > 0.5 mm; **nuisance regression** of slow drifts (0.01 Hz cosine basis), principal components of the motion parameters and their squares, and mean white-matter and ventricle signals; **6 mm smoothing**.

**What this protocol does:**
- **Atlas:** Schaefer 2018, 100 parcels, 7 networks. The atlas ships in FSL MNI152 space at 2 mm and is resampled onto the data grid (nearest neighbour). The resulting small misalignment at region borders is a stated limitation (§10); a TemplateFlow atlas in the matching space would remove it.
- **ROI signals:** the mean time series per parcel, z-scored over time. **No further denoising and no further censoring.**
- **Connectivity:** Pearson correlation via `np.corrcoef` (not `ConnectivityMeasure`, whose default applies shrinkage), Fisher z transform (`arctanh`, |r| clipped at 1 − 10⁻⁷), vectorized as the row-major upper triangle excluding the diagonal: **4,950 edges**.
- Each edge's ROI pair and network pair are written to `functional_edges.csv`.

**Functional QC (D8):** at least 180 s retained after NIAK's scrubbing (primary); 120 s as a sensitivity analysis. Mean FD, from the release's phenotype table, is a confound rather than a second exclusion rule.

**Not available with these derivatives:** temporal SNR (meaningless after denoising and smoothing), and the alternative-denoising sensitivity analyses (`simple`, `scrubbing` + global signal regression).

### 3.4 Dimensionality
Xf (4,950 features) is roughly 33 times larger than Xs (≈ 150), and both blocks have far more features than participants (82–101). This motivates kernel fusion with trace normalization (§5.2) and the strict regularization and nested cross-validation design (§5).

---

## 4. Models compared

All kernel models use the same classifier, `sklearn.svm.SVC(kernel="precomputed")`, on linear kernels, differing only in how the kernel is assembled. An SVC on a linear kernel is the L2-regularized, hinge-loss linear SVM.

| Model | Kernel | Weights | Role |
|---|---|---|---|
| `structural` | K_s | — | unimodal baseline |
| `functional` | K_f | — | **primary comparator** |
| `early_fusion` | linear kernel on z-scored [Xs, Xf] | implicit, ∝ feature count | secondary |
| `equal_weight` | 0.5·K_s + 0.5·K_f | fixed | secondary |
| `mkl` | β_s·K_s + β_f·K_f | learned (§5.2) | **primary** |
| `stacking` | L2 logistic regression on the two unimodal SVM scores | learned | secondary |
| `confounds_only` | linear kernel on the confound set (D9) | — | confound baseline (§8) |
| `mkl_alignment`, `functional_residualized`, `mkl_residualized` | see §8 | | sensitivity |

**Why early fusion can let one modality dominate.**
- *The arithmetic part.* After z-scoring, concatenating features is *exactly* a kernel SVM on (p_s·K_s + p_f·K_f)/(p_s + p_f), where K_s and K_f are the trace-normalized modality kernels. The functional block therefore gets an implicit weight of about 4,950 / 5,100 ≈ 0.97. The code reports this implicit weight for every fold.
- *The rest.* Even with equal kernel weights, a modality can dominate through:
  - redundancy and correlation structure, i.e. many correlated edges acting like fewer, stronger features;
  - noise level;
  - how variance is spread across features;
  - interaction with the regularization strength C;
  - upstream preprocessing.
- Trace normalization removes only the difference in overall kernel scale, so the benefit of learned weighting must be shown empirically under identical splits. β is reported as the model's kernel mixture, never as biological importance.

**Late fusion (stacking).**
- Base models are the structural and functional kernel SVMs, each selecting its own C by inner CV on the outer training fold.
- The meta-model is a z-scored L2 logistic regression with C = 1, trained only on out-of-fold base scores: for each inner split the base models are refit on the inner-training participants at the selected C and score the inner-validation participants. The meta-model never sees a score from a base model fit on that participant.
- One small leak remains inside the outer training fold: the base models' C was selected by an inner CV that included those participants. Outer test participants are never touched.

---

## 5. Model pipeline and leakage prevention

### 5.1 Software
| Component | Version |
|---|---|
| Python | 3.12.10 |
| numpy / scipy / pandas | 2.5.1 / 1.18.0 / 3.0.5 |
| scikit-learn (SVC → libsvm; LogisticRegression → lbfgs) | 1.8.0 |
| nilearn / nibabel | 0.14.1 / 5.4.2 |
| joblib / PyYAML | 1.5.3 / 6.0.3 |
| Upstream functional preprocessing | NIAK under Octave 3.8.1 and MINC toolkit 0.3.18, fixed by the release. Version strings differ between sources: the figshare title says **0.12.4**, the archive README says **0.12.14**; both are recorded in `data/README.md`, and the README governs. |
| Structural pipeline | pinned at lock (D3b) |
| MKL | implemented in this repository (`notebooks/04`, mirrored in `src/`). No third-party MKL package: with two kernels, an exact search over the weight simplex is transparent and reproducible. |

### 5.2 Kernel and MKL specification
| Item | Specification |
|---|---|
| Feature scaling | Each feature z-scored with the training-fold mean and SD (`StandardScaler`). Zero-variance features become 0. |
| Kernel | Linear: K = Z Zᵀ |
| **Kernel normalization** | Divide by the mean diagonal of the *training* kernel; the same scalar is applied to the test-by-training cross-kernel. For z-scored features this scalar equals the number of non-constant training features. |
| **Centering** | None. Features are already centred by training-fold standardization, and an SVM with an unregularized intercept is unaffected by a constant shift. Centred kernels appear only inside the alignment sensitivity estimator, and only on training rows. |
| **Weight constraint** | β_s, β_f ≥ 0 and β_s + β_f = 1 (the simplex) |
| **MKL procedure (primary)** | β_s chosen from {0, 0.1, …, 1.0}, with β_f = 1 − β_s, *jointly* with C, maximizing mean inner-CV ROC-AUC. Grid endpoints reproduce the unimodal models; β_s = 0.5 reproduces equal weighting. |
| **MKL regularization parameter** | None beyond the simplex constraint and the 0.1 grid resolution. Overfitting of β is controlled by nesting: β is judged only on outer-test participants. |
| SVM C grid | {10⁻³, 10⁻², 10⁻¹, 1, 10, 100}, identical for all kernel models |
| Class weighting | `class_weight="balanced"` |
| Tie-breaking | Among configurations within 10⁻¹² of the best mean inner AUC: smallest C, then β_s closest to 0.5. |

### 5.3 Nested cross-validation (all models)
**Outer loop:** stratified 5-fold CV, repeated 10 times with the seeds in `configs/seeds.txt`. It produces every reported performance estimate.

**Inner loop:** stratified 5-fold CV inside each outer training fold, seeded from the repeat seed and fold index (`SeedSequence([seed, fold])`). It selects C, β and the decision threshold.

**Pairing:** every model uses identical outer *and* inner splits, so outer-fold predictions are paired across models.

**Repeats** give an empirical distribution of performance over partitions. They are not independent replications and are never interpreted as independent studies.

### 5.4 Leakage rule (applied without exception)
| Operation | Fit on | Applied to |
|---|---|---|
| Feature standardization | training fold | training + held-out |
| Kernel normalization scalar | training kernel | training + cross-kernel |
| Confound regression (sensitivity) | training fold | training + held-out |
| C, β selection | inner CV within outer-training | outer-training refit |
| Alignment weights (sensitivity) | training kernels | — |
| Decision threshold | inner out-of-fold scores | outer test |
| Stacking meta-model | inner out-of-fold base scores | outer test |
| Per-participant operations (Fisher z, eTIV ratio) | that participant only | — |

Imputation is not used (§2.5). Unit tests check that outer-test predictions are unchanged when the features or labels of the *other* test participants are altered.

### 5.5 Feature selection
- **None** in the primary analysis; all features enter the L2-regularized models, so there is no feature-selection threshold to tune.
- Any exploratory analysis that adds feature selection refits it inside every training fold and is labelled exploratory.

### 5.6 Decision threshold (D6)
1. Pool the inner out-of-fold decision scores of the selected configuration.
2. Take the threshold maximizing Youden's J (sensitivity + specificity − 1); ties go to the threshold closest to 0.
3. Classify outer-test participants with score ≥ threshold as patients.

**Caveat:** inner models are trained on about 80% of the outer training fold, so their score scale may differ slightly from the refit model's. **Sensitivity analysis:** metrics are also reported at a threshold of 0.

---

## 6. Evaluation metrics

**Unit of computation:** every metric is computed within each outer test fold and averaged over the 50 folds. Scores are never pooled across folds, because SVM decision scores from different fold models are on different scales. The SD across folds describes partition variability and is not a standard error.

**Metrics:**
- **ROC-AUC.** Ranking discrimination, relatively insensitive to prevalence. It says nothing directly about positive predictive value: a model can have a good AUC and still give poor PPV in an imbalanced population, which is why the metrics below accompany it.
- **PR-AUC (average precision).** Positive class = patient; the chance level equals the test fold's prevalence, reported alongside.
- **Balanced accuracy, sensitivity, specificity** at the §5.6 threshold.
- **Primary comparison metric:** the paired fold-level ΔAUC (MKL − functional).

---

## 7. Statistical validation

### 7.1 Does each model beat chance?
- Diagnosis labels are permuted B = 1,000 times.
- Each permutation reruns the complete nested pipeline for every non-sensitivity model: split generation, standardization, kernel normalization, C/β selection, threshold, evaluation.
- Statistic: each model's mean outer-fold AUC. p = (1 + #{null ≥ observed}) / (1 + B).
- This tests exchangeability within this dataset. It establishes neither causal validity, nor clinical utility, nor cross-site generalization.

### 7.2 Primary test: incremental value of structural MRI
**Why labels are not permuted here.** Permuting labels destroys the association with *both* modalities, so both models fall to chance and ΔAUC centres on zero whether or not structural MRI adds anything. That answers "do the models differ when there is no signal at all?", which is not the primary question.

**Null used instead.** *Structural features carry no information MKL can exploit.* The rows of Xs are permuted across participants while labels, Xf and confounds stay fixed; the complete nested pipeline is rerun for `mkl` and `functional`, B = 1,000 times. Labels are unchanged, so the splits match the observed analysis.

**Statistic and p-value.** Mean paired ΔAUC; one-sided plus-one p-value (1 + #{null ≥ observed}) / (1 + B).

**Caveat.** This null also breaks any dependence between Xs and Xf, so it tests whether Xs is usable at all, not conditional independence (Xs ⟂ y | Xf).

**Uncertainty.** 95% interval from the Nadeau–Bengio corrected repeated-CV t-procedure: variance factor 1/J + n_test/n_train with n_test/n_train = 1/4, J = 50 folds, t with J − 1 degrees of freedom. The correction is approximate; the interval is read together with the permutation test, never alone.

### 7.3 Primary decision rule (fixed before results)
| Result | Criteria | Interpretation |
|---|---|---|
| **Superiority supported** | p_block < 0.05, **and** CI lower bound > 0, **and** mean ΔAUC ≥ δ (0.02) | MKL improves on functional connectivity by at least the prespecified meaningful amount |
| **Detectable but small** | p_block < 0.05 and CI lower bound > 0, but mean ΔAUC < δ | Structural information adds a detectable but not meaningful improvement |
| **No meaningful incremental value** | CI upper bound < δ | A meaningful improvement is ruled out under this design (§9) |
| **Inconclusive** | none of the above | The data cannot distinguish a meaningful improvement from none |

Implemented in `classify_primary_result` (notebook 06, mirrored in `src/metrics.py`) and applied automatically.

### 7.4 Secondary comparisons
- MKL versus `early_fusion`, `equal_weight`, `stacking` and `structural`.
- Each reports the mean paired ΔAUC, the corrected 95% CI and the corrected-t p-value; the four p-values are Holm-adjusted at α = 0.05.
- Every other comparison, and every sensitivity analysis, is labelled exploratory.

### 7.5 Stability analyses
**Prediction stability.** Each participant is tested once per repeat, giving 10 out-of-sample scores per model. Report the mean decision score, its SD, and the proportion of repeats classified as patient. Because a thresholded label hides changes in confidence, the score SD is the primary summary.

**Feature stability.** With no feature-selection step there are no "selected features". For each model and modality the weight vector on standardized features is recovered from the 50 outer-fold fits, w_m = (β_m / scale_m) · Σᵢ αᵢyᵢ z_m(xᵢ). Report:
- mean pairwise cosine similarity across fits;
- per-feature sign consistency;
- for functional edges, mean signed and mean absolute weight aggregated to Yeo-7 network pairs;
- for structural features, weights per cortical region.

Correlated features swap weight between folds, so feature-level stability is never read as biological specificity.

**MKL weights.** The distribution of the selected β_s across folds, and how often β_s = 0 was chosen (MKL reduced to the functional model).

---

## 8. Confounds and sensitivity analyses

**Descriptive report, by group, before modelling:** age, sex, head motion (mean FD), retained scan time, eTIV and surface holes when available, medication if available, and acquisition metadata.

| Factor | Handling |
|---|---|
| Age, sex | Described by group; in the confound set for the confound-only baseline and residualization |
| Medication | Described if available. COBRE patients are predominantly medicated, so medication cannot be separated from diagnosis; stated as a limitation, not regressed out |
| Head motion | Already scrubbed by NIAK at FD > 0.5 mm; retained-time rule (D8); mean FD in the confound set; stringent sensitivity analysis at mean FD > 0.25 mm |
| Retained scan time | Differs by group and drives who is excluded; described by group and reported under both rules |
| Signal quality | tSNR unavailable with these derivatives; FreeSurfer surface holes enter the confound set once structural data exist |
| Missingness | No imputation; complete-case exclusion with reasons reported (§2.3) |
| Class imbalance | Stratified splits, `class_weight="balanced"`, PR-AUC reported with the test-fold prevalence. The 180 s rule leaves 31 patients vs 51 controls |
| Scanner/acquisition | Single scanner; parameters from the release README and the official parameter files |
| Diagnostic/recruitment differences | Strict-schizophrenia sensitivity analysis once subtypes are available; subgroups described |

**Sensitivity analyses.** All are reported regardless of outcome; none replaces the primary analysis.
1. **Confound-only baseline:** linear SVM on the confound set through the identical nested pipeline.
2. **Residualized imaging:** cross-validated confound regression (per-feature OLS slopes fit on training rows only) applied to `functional` and `mkl`.
3. **Motion rule:** the analysis repeated under the ≥ 120 s rule.
4. **Strict schizophrenia labels** (needs the official phenotype file).
5. **Stringent motion:** exclude participants with mean FD > 0.25 mm.
6. **Alternative MKL estimator:** centered kernel alignment weights (Cortes, Mohri & Rostamizadeh, 2012) instead of the inner-CV grid.
7. **Threshold:** metrics at a decision score of 0.

*Withdrawn from v1.0:* alternative denoising strategies, which the fixed preprocessing of the release makes impossible.

---

## 9. Interpretation of null results

- Failure of MKL to outperform the functional model is not a failed experiment. It is read as evidence that multimodal fusion gave no detectable incremental value under this dataset and validation design.
- If MKL selects β_s = 0 in most folds, the paper reports that learned fusion reduced to the functional model.
- Because earlier work reports inconsistent multimodal superiority, "no meaningful incremental value" (§7.3) is a prespecified, publishable result.

---

## 10. Limitations

- **Single site.** No leave-one-site-out validation is possible. The strongest supportable claim is generalization to unseen COBRE participants. A second public dataset (for example the UCLA CNP sample on OpenNeuro) is future work, not claimed here.
- **Sample size.** With 82 participants under the primary rule (31 patients) and 4,950 functional features, estimates are noisy and intervals wide.
- **Motion-driven selection.** The retained-time rule removes 57% of patients and 31% of controls, so the analysed sample favours participants who could stay still (§2.3).
- **Fixed preprocessing.** The functional preprocessing, including 6 mm smoothing and the absence of global signal regression, is inherited from the release and cannot be varied.
- **Atlas-space mismatch.** The Schaefer atlas is resampled from FSL MNI152 onto MNI152 2009a symmetric data, leaving small border misalignments.
- **Medication and chronicity** are confounded with diagnosis.
- **Residual motion effects** may persist after scrubbing and confound regression.
- **Approximate inference.** CV intervals are approximate, and the block-permutation null also breaks Xs–Xf dependence.
- **Weights are not biology.** Kernel and feature weights describe the fitted model only.

---

## 11. Computational plan

The primary run fits 10 repeats × 5 outer folds × 5 inner folds × (66 MKL configurations + 6 per other kernel model). Each permutation test reruns that pipeline 1,000 times in resumable chunks, every permutation using its own RNG stream (`SeedSequence([seed, index])`).

### 11.1 Measured runtime
Synthetic data at COBRE size (n = 146, 150 + 4,950 + 6 features), full grids, one core:

| Workload | Per repeat | Per run (10 repeats) | × 1,000 permutations |
|---|---|---|---|
| All 10 models | 8.0 s | ≈ 80 s | — |
| `mkl` + `functional` (structural-block null) | 2.7 s | ≈ 27 s | ≈ 7.5 core-hours (≈ 30 min on 16 cores) |
| 7 non-sensitivity models (label null) | 4.9 s | ≈ 49 s | ≈ 13.6 core-hours (≈ 1 h on 16 cores) |

Real data, functional arm (n = 82, 4,950 + 3 features, 16 cores):

| Step | Time |
|---|---|
| Connectivity extraction, 146 participants, streamed from the zip | 81 s |
| Nested CV, `functional` + `confounds_only`, 10 repeats | 20 s |
| 500 label permutations, full pipeline each | 641 s |

Structural preprocessing (FreeSurfer or FastSurfer) will dominate total compute and is excluded from these figures.

---

## 12. Deliverables and project structure

- This protocol, fixed before confirmatory results.
- The analysis notebooks, which are the analysis itself; `src/` mirrors them as importable modules with unit tests.
- A README covering the research question, data access, environment setup, structure and reproducibility.

```
mkl-project/
├── README.md  LICENSE  requirements.txt  environment.yml  CITATION.cff
├── protocol/protocol.md
├── configs/          model_config.yaml, cv_config.yaml, data_config.yaml, seeds.txt
├── notebooks/
│   ├── 00_protocol_overview.ipynb          objective, decisions, versions, lock hashes
│   ├── 01_dataset_audit.ipynb              sample rule, value handling, exclusions, motion
│   ├── 02_structural_features.ipynb        FreeSurfer stats → Xs (pending data)
│   ├── 03_functional_features.ipynb        fMRIPrep route (not used; helpers reused)
│   ├── 03b_niak_functional_features.ipynb  NIAK release → manifest + Xf
│   ├── 04_kernels_and_models.ipynb         SVM, kernels, MKL, stacking
│   ├── 05_nested_cross_validation.ipynb    leakage, nested CV, thresholds
│   ├── 06_metrics_and_statistical_inference.ipynb   metrics, ΔAUC, permutations, decision rule
│   ├── 07_stability_and_confounds.ipynb    stability, confounds, sensitivity analyses
│   ├── 08_visual_exploration.ipynb         results figures
│   └── 09_functional_only_analysis.ipynb   preliminary exploratory run (§15)
├── src/              same code as modules; tests/ exercises it
├── data/             raw/ derivatives/ qc/ participants_manifest*.csv (data not tracked)
├── results/          predictions/ metrics/ figures/ permutation_tests/
└── manuscript/       outline.md, methods.md, limitations.md
```

---

## 13. Deviations log

| Date | Section | Deviation | Reason | Decided before/after seeing outcome data |
|---|---|---|---|---|
| 2026-09-18 | §3.1, §3.3 (D3), D9 | Functional features taken from the verified NIAK 0.12.14 release (figshare 1160600) instead of fMRIPrep. Denoising sensitivity analyses and tSNR unavailable; the functional-arm confound set is age, sex, mean FD. | The raw COBRE T1 scans were not yet available, and fMRIPrep needs Docker and storage this machine lacks. | Before: no outcome data seen |
| 2026-09-18 | §1, §7 | Preliminary **functional-only** analysis before lock (notebook 09): functional model and confounds-only baseline, 500 label permutations, both motion rules. Labelled exploratory. | To validate the pipeline on real data while structural data are pending. | Before lock; results exploratory (§15) |
| 2026-09-26 | §2.3, §3.3 (D8) | Functional QC changed from an FD-threshold rule to a **retained-scan-time** rule, because the release is already scrubbed and ships no per-volume FD. | The derivative's format makes the original rule inapplicable. | Before lock; consequences reported in §2.3 |

---

## 14. Lock record

| Field | Value |
|---|---|
| Lock date | |
| SHA-256 of `protocol/protocol.md` | |
| SHA-256 of `configs/*.yaml`, `configs/seeds.txt` | |
| Git commit / tag | |
| Functional release + checksum (figshare 1160600) | |
| Structural pipeline + version (D3b) | |
| Motion rule confirmed (D8) | |
| δ confirmed (D2) | |
| Confirmed by | |

---

## 15. Preliminary exploratory results (disclosed, not confirmatory)

Run on 2026-09-18 with the functional data only, before the lock, and reported here so it cannot later be presented as a confirmatory finding.

**Primary motion rule (≥ 180 s): 82 participants, 31 patients, 51 controls**

| Model | ROC-AUC | PR-AUC (chance 0.38) | Balanced accuracy | Label-permutation p (B = 500) |
|---|---|---|---|---|
| `functional` | 0.688 (SD 0.132) | 0.655 | 0.598 | 0.006 |
| `confounds_only` | 0.487 (SD 0.155) | 0.461 | 0.490 | 0.55 |

- Functional minus confounds-only: ΔAUC +0.201, corrected 95% CI [−0.010, +0.412], p = 0.061.
- Sensitivity, ≥ 120 s rule (101 participants): `functional` AUC 0.752, `confounds_only` 0.600.
- Prediction stability: median score SD 0.30; 21 of 82 participants classified identically in all 10 repeats.

**Reading.** Connectivity beats chance; confounds alone do not, under the motion-matched primary sample. "Better than confounds alone" is suggestive but not established. These numbers do not test H1, and they do not license changes to §7 rules.

---

## Appendix A. Changes from the working draft

**v1.0 (initial revision of the draft):**
1. **Objective:** duplicated clause removed; patient definition made explicit (D1).
2. **Primary comparator:** fixed as the functional model; "best unimodal model" removed, since it would be chosen after the fact.
3. **Primary permutation test:** label permutation cannot test the null of no difference between models (§7.2); replaced with structural-block permutation. Label permutation kept for beats-chance tests.
4. **"Meaningful improvement":** defined as δ = 0.02 with a four-way decision rule (§7.3).
5. **Early-fusion dominance:** the exact dimensionality effect separated from redundancy, noise, correlation structure, scale and regularization (§4).
6. **Threshold:** Youden's J on inner out-of-fold scores, with 0 as a sensitivity analysis (§5.6).
7. **Feature selection:** none in the primary analysis; feature stability assessed on weight vectors (§5.5, §7.5).
8. **MKL specification:** software, versions, C grid, weight grid and constraint, normalization and centering all specified (§5.1–5.2).
9. **Metrics:** computed per fold, not pooled across folds (§6).
10. **ROC-AUC wording:** a good AUC does not guarantee good PPV under imbalance; PR-AUC reported with its chance level.
11. **Prediction stability:** 10 scores per participant (mean, SD, classification frequency), not thresholded labels alone.
12. **Data access:** `fetch_cobre` removed from nilearn; Pearson correlation computed directly rather than via `ConnectivityMeasure`'s shrinkage default.
13. **Secondary hypothesis:** the mechanism is described using β, not asserted.
14. **Missingness:** explicit no-imputation, complete-case rule.
15. **Stacking:** the remaining in-fold leak (base-model C selection) stated.

**v1.1 (this revision):**
16. **Data sources (§2.1):** split into the functional release now held and the structural scans still being obtained, with licences and status.
17. **Processing route (§3.1, D3/D3b):** functional from verified NIAK derivatives; structural pipeline pinned when the T1 scans arrive. The alternative-denoising sensitivity analyses are withdrawn as impossible.
18. **Sample rule (§2.3, D8):** now a retained-scan-time rule, with the observed counts (146 → 82 or 101) and an explicit statement of the motion-driven selection bias it causes.
19. **Structural features (§3.2, D5):** 150 features under FreeSurfer's DK atlas or 138 under FastSurfer's DKT atlas; multi-echo MPRAGE handling added.
20. **Value handling (§2.4):** missing-value tokens, case- and whitespace-insensitive matching, and the rule that unlisted values stop the audit.
21. **Runtime (§11.1) and structure (§12):** measured real-data timings added; the notebook-based structure documented.
22. **Preliminary results (§15):** the exploratory functional-only analysis disclosed in full.
