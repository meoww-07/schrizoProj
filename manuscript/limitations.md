# Limitations (draft, from protocol §10)

- **Single site.** COBRE does not allow leave-one-site-out validation. The claims extend to unseen COBRE participants, not to other scanners, sites or populations.
- **Sample size.** With p ≫ n, performance estimates are noisy: outer test folds hold about 30 participants. Repeated CV describes variability across partitions. It does not replicate the study.
- **Medication and chronicity.** Patients are expected to be predominantly medicated, so the classifier may pick up effects of treatment or illness duration as well as effects of diagnosis.
- **Motion.** Motion effects may remain after censoring and confound regression, and motion differs between groups. The confound-only baseline and the residualized sensitivity models address this, but only in part.
- **Inference.** CV-based intervals are approximate. The structural-block permutation null also breaks the dependence between Xs and Xf.
- **Analytic choices.** The atlas, denoising strategy and structural parcellation are fixed. Other defensible choices exist, and the sensitivity analyses cover only some of them.
- **Interpretation of weights.** Learned kernel weights and feature weights describe the fitted model. They do not show which modality or brain region is biologically more important.
