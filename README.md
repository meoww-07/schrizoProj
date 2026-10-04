# Multimodal MRI Fusion for Psychosis Classification (COBRE, MKL)

**Research question.** Can participants with schizophrenia-spectrum disorders be told apart from healthy controls using structural and resting-state functional MRI? And does fusing the two modalities with **multiple kernel learning (MKL)** add discriminative value beyond functional connectivity alone?

The analysis is specified in advance in [`protocol/protocol.md`](protocol/protocol.md); the files in `configs/` are part of that specification. Changes made after the protocol is locked are recorded in its deviations log.

## Data

| Modality | Source | Status |
|---|---|---|
| Functional | *COBRE preprocessed with NIAK* (figshare 1160600), 146 participants, already denoised and scrubbed | held |
| Structural | Raw T1w scans from `COBRE_scan_data.tgz` (NITRC/INDI), 148 participants, 1 mm | held |

146 participants have both modalities. After the motion rule (at least 180 s of resting-state data retained after scrubbing) the analysis sample is **82 participants: 31 patients, 51 controls**. Provenance, checksums and licences are recorded in [`data/README.md`](data/README.md). No imaging data are tracked by git.

## The notebooks are the analysis

Each notebook explains its concepts in markdown, then implements them. Run any notebook top to bottom (*Restart & Run All*). Definition cells are tagged `library`, and later notebooks load earlier ones through `use_notebook(...)`, so nothing is duplicated.

| Notebook | What it does | Needs |
|---|---|---|
| `00_protocol_overview` | objective, hypotheses, prespecified settings, version check, lock fingerprints | — |
| `01_dataset_audit` | sample rule, value handling, exclusion flow, head-motion measures | — |
| `02_structural_features` | the FreeSurfer-surface route and its stats parsing (not used; see 02b) | — |
| `02b_fastsurfer_structural_features` | FastSurfer segmentation → regional volumes (Xs) | T1 scans, segmentation output |
| `03_functional_features` | connectivity concepts; helper functions reused by 03b | — |
| `03b_niak_functional_features` | NIAK release → manifest + 4,950 connectivity edges (Xf) | the NIAK archive |
| `04_kernels_and_models` | SVM, kernels, normalization, early fusion, MKL, stacking | — |
| `05_nested_cross_validation` | leakage, nested CV, repeats, seeds, decision thresholds | — |
| `06_metrics_and_statistical_inference` | ROC/PR-AUC, paired ΔAUC, corrected intervals, permutation tests, Holm | — |
| `07_stability_and_confounds` | prediction/feature stability, confounds, sensitivity analyses | — |
| `08_visual_exploration` | results figures and how to read them | results |
| `09_functional_only_analysis` | preliminary functional-only run (exploratory) | 03b output |
| `10_multimodal_mkl_analysis` | **the primary analysis**: all models, both permutation nulls, the prespecified decision | 02b + 03b output |

## Reproducing the analysis

Environment (Python 3.12):

```bash
pip install -r requirements.txt jupyterlab
```

Then, in order:

1. **Functional features** — notebook `03b`. Reads the NIAK archive straight from the zip (nothing is unzipped) and writes the manifest plus the connectivity table. About 90 seconds.
2. **Structural features** — segmentation first, then notebook `02b`. Segmentation uses FastSurfer with a CUDA-capable GPU:

```bash
python -m FastSurferCNN.run_prediction --in_dir <t1_dir> --tag "*_T1w.nii.gz" --remove_suffix "_T1w.nii.gz" --sd <out_dir> --device cuda
```

3. **The analysis** — notebook `10`. Runs every model through 10 × 5 × 5 nested cross-validation, then both permutation nulls, and applies the decision rule from protocol §7.3.
4. **Figures** — notebook `08`.

Paths live in `configs/data_config.yaml`; absolute paths are fine, which is how large data stay outside the synced project folder.

## Tests

```bash
python -m pytest
```

72 tests covering the guarantees that are invisible when broken: that a test participant's prediction never depends on other test participants, that kernel statistics come only from training rows, that stacking sees only out-of-fold scores, that MKL's grid endpoints reproduce the unimodal models, and that the decision rule classifies each case correctly. `src/` mirrors the notebooks as importable modules so the tests can exercise them.

## Reproducibility

- Versions are pinned in `requirements.txt` and recorded in `configs/model_config.yaml`; notebook 00 compares them with what is installed.
- Cross-validation uses the fixed seeds in `configs/seeds.txt`; each inner split's seed is derived from its repeat seed and fold index.
- Every permutation has its own RNG stream, so permutation chunks can be computed separately and still give the same null.
- Each run records package versions and SHA-256 fingerprints of the protocol and configs.

## Status

The functional arm is complete; its preliminary result is disclosed in protocol §15. The structural arm and the multimodal comparison run once segmentation finishes. The protocol's lock record (§14) is not yet completed, so current results are exploratory by the project's own rules.
