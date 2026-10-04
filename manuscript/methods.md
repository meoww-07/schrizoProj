# Methods

Written against `protocol/protocol.md` v1.1. Where the two differ, the protocol and its deviations log (§13) take precedence.

## Participants

Data come from the COBRE sample (Center for Biomedical Research Excellence, Mind Research Network), acquired at a single site on one 3 T Siemens Trio scanner.

Two components of that sample were used. Functional data came from the verified preprocessed release *COBRE preprocessed with NIAK* (figshare article 1160600, version 15, CC BY-NC), which covers 146 participants (72 patients with a schizophrenia-spectrum diagnosis, 74 healthy controls). Structural data came from the raw COBRE release distributed through NITRC/INDI (`COBRE_scan_data.tgz`), which contains T1-weighted scans for 148 participants. Checksums and access dates are recorded in `data/README.md`.

Participants entered the analysis when they had a valid diagnostic label, recorded age and sex, a T1 scan with a successful segmentation, and at least 180 s of resting-state data remaining after the release's motion scrubbing. Of the 146 participants with both modalities, **82 met the retained-time criterion: 31 patients and 51 controls**. This criterion removed 57% of patients but only 31% of controls, because patients moved more; the resulting sample is motion-matched (mean framewise displacement 0.203 mm in controls, 0.209 mm in patients) but favours participants able to remain still. A more permissive 120 s rule (101 participants) is reported as a sensitivity analysis.

No values were imputed at any point.

## Functional features

The NIAK release supplies resting-state images that have already been slice-time and motion corrected, normalized to MNI152 2009a symmetric space at 3 mm, scrubbed of volumes exceeding 0.5 mm framewise displacement, regressed against nuisance signals (slow drifts, motion principal components, white-matter and ventricle signals), and smoothed at 6 mm. No further denoising or censoring was applied, to avoid processing the data twice.

Regional time series were extracted with the Schaefer 2018 atlas (100 parcels, 7 networks), resampled onto the data grid, and z-scored over time. Pairwise Pearson correlations between parcels were computed with `numpy.corrcoef`, Fisher z transformed, and vectorized over the upper triangle excluding the diagonal, giving **4,950 connectivity features** per participant.

## Structural features

T1 scans were segmented with FastSurfer's VINN network (segmentation stream only; PyTorch 2.7.1+cu128 on an NVIDIA RTX 5060), taking about one minute per participant. Cortical surface reconstruction was not run, so cortical thickness and surface area were unavailable; this was a compute-driven decision recorded in the protocol before any structural feature was linked to a diagnostic label.

Regional volumes were computed from the DKT whole-brain segmentation: 62 cortical regions (31 per hemisphere) and the 14 subcortical structures (bilateral thalamus, caudate, putamen, pallidum, hippocampus, amygdala and nucleus accumbens), giving **76 structural features**. Each volume was divided by that participant's total segmented brain volume, an adjustment computed from the participant alone and therefore free of any cross-participant leakage. All 95 labels were present in every participant, and total brain volumes ranged from 0.85 to 1.58 litres.

## Classification models

Every model was an L2-regularized, hinge-loss support vector machine, implemented as `sklearn.svm.SVC` (scikit-learn 1.8.0) with a precomputed kernel and balanced class weights. Each modality was represented by a linear kernel on z-scored features, divided by the mean diagonal of the training kernel. This trace normalization removes the scale advantage the 4,950-feature functional block would otherwise hold over the 76-feature structural block. Kernels were not centred: features are already centred by training-fold standardization, and an SVM with an unregularized intercept is invariant to constant shifts.

Seven models were compared:

- **structural** and **functional**, the unimodal baselines, the latter being the prespecified comparator;
- **early fusion**, a single kernel over the concatenated standardized features, which implicitly weights each block by its feature count (0.015 for the structural block here);
- **equal-weight fusion**, K = 0.5·Ks + 0.5·Kf;
- **MKL**, K = βs·Ks + βf·Kf with βs chosen from {0, 0.1, …, 1} jointly with C by inner cross-validation, under the simplex constraint (non-negative weights summing to one);
- **stacking**, an L2 logistic regression on the two unimodal decision scores, trained only on out-of-fold scores generated within the inner loop;
- **confounds only**, a kernel on age, sex and mean framewise displacement.

The regularization strength C was selected from {10⁻³, 10⁻², 10⁻¹, 1, 10, 10²} for every kernel model. No feature selection was performed.

## Validation

All models were evaluated by stratified 5-fold cross-validation repeated 10 times, with an inner stratified 5-fold loop inside each outer training fold used to select C, β and the decision threshold. Every model received identical outer and inner splits, so all comparisons are paired. Split seeds are fixed in `configs/seeds.txt`, and each inner split's seed is derived from its repeat seed and fold index.

Every data-dependent operation (feature standardization, kernel normalization, hyperparameter selection, threshold selection, and confound regression where used) was fitted on training-fold data only and applied unchanged to held-out participants. Unit tests verify that an outer-test participant's prediction is unaffected by the features or labels of other test participants.

Decision thresholds were set by maximizing Youden's J on the pooled inner out-of-fold scores of the selected configuration, never on outer-test participants.

## Metrics and inference

ROC-AUC, PR-AUC, balanced accuracy, sensitivity and specificity were computed within each outer test fold and then averaged over the 50 folds; scores were never pooled across folds, because decision scores from different fold models lie on different scales.

The primary estimand was the mean paired difference in outer-fold AUC between MKL and the functional model. Its interval used the Nadeau–Bengio correction for repeated cross-validation, with variance factor 1/J + n_test/n_train (J = 50 folds, ratio = 1/4).

Two permutation nulls were computed, each re-running the complete nested pipeline (split generation, standardization, kernel normalization, hyperparameter and threshold selection) for every permutation:

1. **Structural-block null (primary, 1,000 permutations).** The rows of the structural block were permuted across participants while labels, connectivity and confounds were held fixed. This tests whether structural MRI carries information the fusion model can exploit. Permuting labels cannot answer this, because it drives both models to chance and centres their difference on zero regardless.
2. **Label null (1,000 permutations).** Diagnosis labels were permuted, testing each model against chance.

Monte Carlo p-values used the plus-one correction, (1 + #{null ≥ observed}) / (1 + B). Secondary comparisons between MKL and the other fusion strategies were Holm-corrected across four tests.

Superiority was declared only if all three prespecified conditions held: a permutation p below 0.05, a corrected interval excluding zero, and a mean difference of at least δ = 0.02 AUC.

## Software

Python 3.12.10, scikit-learn 1.8.0, numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, nilearn 0.14.1, nibabel 5.4.2, joblib 1.5.3. Upstream functional preprocessing used NIAK 0.12.14 under Octave 3.8.1 with the MINC toolkit 0.3.18. Multiple kernel learning was implemented directly in this repository rather than through a third-party package: with two kernels, an exact search over the weight simplex is transparent and reproducible. Analysis code, configuration and the preregistered protocol are available at the project repository.
