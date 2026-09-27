# Manuscript outline

1. **Introduction**
   - The need for objective markers in psychosis.
   - Mixed evidence on whether multimodal MRI adds value.
   - Why kernel fusion suits p ≫ n data.
2. **Methods** (`methods.md`, mirroring the protocol)
3. **Results**
   1. Sample flow and descriptives (`results/metrics/sample_flow.csv`, `sample_description.csv`)
   2. Performance of each model: AUC, PR-AUC, balanced accuracy, sensitivity, specificity (`summary.csv`)
   3. Primary result: ΔAUC for MKL − functional, with CI, block-permutation p-value, and decision category (`primary_result.json`)
   4. Secondary comparisons, Holm-adjusted (`comparisons.csv`)
   5. Learned kernel weights β across folds, and the implicit weight in early fusion (`selections.csv`)
   6. Beats-chance permutation tests (`beats_chance.csv`)
   7. Stability of predictions and weights (`results/metrics/stability/`)
   8. Confound-only baseline and sensitivity analyses
4. **Discussion.** Interpret the results by the decision category fixed in protocol §7.3, then cover limitations (`limitations.md`).
5. **Data and code availability**
