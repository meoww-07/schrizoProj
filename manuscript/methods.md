# Methods (draft)

This draft follows `protocol/protocol.md`. Where the two differ, the protocol and its deviations log (§13) take precedence. Fill in the bracketed values from the audit and the run logs.

## Participants
- **Data:** COBRE [release, access date].
- **Primary sample:** participants with a valid diagnostic label, age and sex, a usable T1w image, and a usable resting-state BOLD run (protocol §2.3).
- **Sample flow:** [initial N] → [final N] ([patients] patients, [controls] controls). Exclusions are in Table 1 and `sample_flow.csv`.

## Image processing
- **Preprocessing:** fMRIPrep [version] with FreeSurfer [version] (protocol §3.1).
- **Structural features:**
  - Desikan–Killiany cortical thickness and surface area, and 14 subcortical volumes.
  - Area and volume divided by eTIV.
  - [N] features in total.
- **Functional features:**
  - Denoising: nilearn `scrubbing` strategy, i.e. 24 motion parameters, WM/CSF signals, cosine high-pass, and censoring at FD > 0.5 mm or standardized DVARS > 1.5.
  - Atlas: Schaefer-100 (7 networks).
  - Connectivity: Pearson correlation, Fisher z, upper triangle ([N] edges).

## Classification
- **Kernels:** a linear kernel on z-scored features per modality. Kernels are trace-normalized using training-fold statistics.
- **Classifier:** `sklearn.svm.SVC` (scikit-learn 1.8.0) with a precomputed kernel and balanced class weights.
- **MKL:**
  - K = β_s K_s + (1 − β_s) K_f.
  - β_s ∈ {0, 0.1, …, 1} and C ∈ {10⁻³, …, 10²} are selected jointly by inner 5-fold CV AUC.
- **Comparators:**
  - unimodal models;
  - early concatenation;
  - equal-weight kernel fusion;
  - stacking (logistic regression on out-of-fold unimodal scores);
  - a confound-only model.

## Validation and inference
- **Cross-validation:** stratified 5-fold CV repeated 10 times, with an inner stratified 5-fold CV. All models use identical splits.
- **Metrics:** computed per outer fold, then averaged.
- **Decision threshold:** Youden's J on inner out-of-fold scores.
- **Primary test:**
  - Structural-block permutation (B = 1,000) of the mean paired ΔAUC (MKL − functional).
  - Nadeau–Bengio corrected 95% CI.
  - Decision rule with δ = 0.02 (protocol §7.3).
- **Secondary comparisons:** Holm-adjusted.
- **Beats-chance tests:** label permutation (B = 1,000), with the full nested pipeline rerun for every permutation.
