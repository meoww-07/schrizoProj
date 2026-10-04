# Limitations

## Statistical power is the binding constraint

With 82 participants (31 patients, 51 controls) and 4,950 functional features, the corrected interval around the primary difference spans roughly ±0.13 AUC, more than six times the 0.02 difference declared meaningful in advance. The study therefore could not have detected the effect it was designed to test, whatever that effect's true size. The inconclusive verdict reflects the design's resolution rather than evidence of absence, and any future version needs either a much larger sample or a far larger expected effect.

## Motion-driven selection

Requiring at least 180 s of usable resting-state data after scrubbing removed 57% of patients but only 31% of controls, because patients moved more. Two consequences follow. The analysed sample is no longer representative of COBRE's patients: it favours those able to lie still, who may differ in symptom severity and medication. And the class balance shifted from roughly even to 31 patients against 51 controls. The more permissive 120 s rule retains 101 participants but readmits a group difference in motion, trading one bias for another.

## Structural features are volumes, not thickness

Cortical thinning is the most replicated structural finding in schizophrenia, and it was the protocol's original target. Surface reconstruction proved infeasible here (1–10 hours per participant, with software that does not run natively on the available machine), so regional volumes from FastSurfer's segmentation were used instead. Volume confounds thickness with surface area, and is a blunter instrument for exactly the effect of interest. A structural block built from thickness might carry information this one does not.

## Fixed preprocessing

The functional data arrived already preprocessed: scrubbed at 0.5 mm, denoised with a specific nuisance model, smoothed at 6 mm, and without global signal regression. These choices could not be varied, so the planned sensitivity analyses over denoising strategy were withdrawn. The results are conditional on one preprocessing pipeline, and alternative choices might shift them.

## Atlas-space mismatch

The Schaefer atlas is distributed in FSL MNI152 space, while the functional data are in MNI152 2009a symmetric space. The atlas was resampled onto the data grid, which leaves small misalignments at parcel borders. These would blur signal slightly rather than create it, but a template-matched atlas would be preferable.

## Medication and chronicity

Patients in COBRE are predominantly medicated. Diagnosis, antipsychotic exposure and illness duration therefore cannot be separated, so any classifier that distinguishes the groups may be detecting treatment effects or chronicity as much as the disorder.

## Residual motion

Motion was scrubbed by the upstream pipeline, the retained-time rule excluded the worst cases, and mean framewise displacement entered the confound set. Residual motion effects on connectivity are nevertheless well documented and cannot be ruled out. The confound-only model reached chance (AUC 0.487, p = 0.58), which argues against motion alone driving the result in this motion-matched sample.

## Single site

All data come from one scanner at one site, so leave-one-site-out validation is impossible. The strongest supportable claim is generalization to unseen COBRE participants under this validation design, not transportability to other hospitals, scanners, populations or recruitment pathways. Replication in an independent sample, such as the UCLA Consortium for Neuropsychiatric Phenomics cohort, remains future work.

## Approximate inference

Cross-validation intervals are approximate even with the Nadeau–Bengio correction, and may still under-cover. The primary permutation null breaks the dependence between the structural and functional blocks as well as the structural-label relationship, so it tests whether structural data are usable at all rather than conditional independence given connectivity.

## Weights describe the model, not the brain

The learned kernel weights and the recovered feature weights characterise a fitted classifier. Correlated features exchange weight across folds, and linear-model weights partly serve to suppress noise rather than to mark signal. None of these quantities should be read as evidence that a particular region or connection differs between groups.

## Analytic choices made under constraint

Several decisions were driven by available compute rather than by methodological preference: segmentation instead of surface reconstruction, one hardware-limited GPU, and permutation counts bounded by runtime. Each is recorded in the protocol's deviations log with its reason and the date it was taken, all before the corresponding results were seen.
