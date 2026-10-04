# Data

COBRE imaging and phenotype data are **not** tracked in this repository. `.gitignore` excludes `raw/` and `derivatives/`. Follow the terms of use that apply to each component.

## Provenance record

### 1. Functional data — held

| Field | Value |
|---|---|
| Title | COBRE preprocessed with NIAK 0.12.4 |
| Author | Pierre Bellec |
| Source | figshare article 1160600 — https://figshare.com/articles/dataset/COBRE_preprocessed_with_NIAK_0_12_4/1160600 |
| Version / published | version 15, published 2015-01-31, modified 2023-05-31 |
| Licence | Creative Commons Attribution–NonCommercial (inherited from the original COBRE release) |
| Downloaded as | `1160600.zip` |
| Size | 8,871,332,359 bytes (8.9 GB) |
| **SHA-256** | `b4fcb4a1572ecd3cd4434ff495cc13880f84222caf01b7cac3bdd7344e1ef816` |
| Checksum verified | 2026-09-27 |
| Contents | 146 participants (72 patients, 74 controls): `fmri_<id>_session1_run1.nii.gz`, `fmri_<id>_session1_run1_extra.mat`, `README.md`, `cobre_model_group.csv` |
| Upstream origin | Derivative of the COBRE sample from INDI: http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |

**Version discrepancy, recorded deliberately.** The figshare title says **NIAK 0.12.4**, while the `README.md` inside the archive says **NIAK version 0.12.14**. The archive README governs, since it ships with the data; both strings are recorded here so the paper can cite them without ambiguity.

**Do not confuse this with other releases.** A separate lightweight release preprocessed with **NIAK 0.17** exists at 6 mm resolution without smoothing, and the scripts in https://github.com/SIMEXP/cobre_preprocessed describe *that* pipeline. The data used here are 3 mm and smoothed at 6 mm.

**Preprocessing already applied** (so it is never repeated downstream): slice-timing and motion correction; normalization to MNI152 2009a symmetric, resampled to 3 mm; scrubbing of volumes with FD > 0.5 mm; regression of slow drifts, motion components, and white-matter and ventricle signals; 6 mm smoothing.

**Not used:** the BASC connectome archive (figshare article 1450804, `cobre_resolution_*.mat`). It uses a different parcellation and would replace the prespecified feature definition.

### 2. Structural data — access requested

| Field | Value |
|---|---|
| Source | COBRE raw release via NITRC/INDI (`COBRE_scan_data.tgz`), or the same data via SchizConnect/COINS |
| Access | NITRC account plus membership of the 1000 Functional Connectomes Project |
| Licence | Creative Commons Attribution–NonCommercial |
| Needed for | T1-weighted multi-echo MPRAGE (5 echoes, 1 mm) → structural features; `COBRE_phenotypic_data.csv` for diagnostic subtypes |
| Status | requested; fill in the date, files and checksums on arrival |

| Field | Value |
|---|---|
| Downloaded | `COBRE_scan_data.tgz`, 2026-10-04, from NITRC file id 5066 |
| Archive size | 4,084,403,718 bytes (3.9 GB) |
| **Archive SHA-256** | `bf0e7a0fb44bc5ad9a4339561024fcc1f2e05e6d82da85a9889cacf36a3881dc` |
| Archive contents | 148 subjects × {`anat_1/mprage.nii.gz`, `rest_1/rest.nii.gz`}, 296 files |
| Files retained | the 148 T1 scans only, extracted to `data/raw/cobre_t1/sub-<ID>_T1w.nii.gz` (774 MB). The raw resting-state runs were not kept: the functional features come from the NIAK release above. |
| **Echo handling** | Not needed. Each `mprage.nii.gz` is a single 3-D volume, 192 × 256 × 256 at 1 mm, so the five MPRAGE echoes were already combined in this release. |
| Overlap with the functional release | 146 of 146 — every participant with functional data has a T1. Two subjects (`0040070`, `0040083`) have a T1 but no functional data and fall out of the intersection. |

## Expected layout
```
data/
├── participants_manifest_niak.csv      written by notebook 03b (functional arm)
├── participants_manifest.csv           written by notebook 01 (multimodal arm)
├── derivatives/
│   ├── atlases/                        Schaefer 2018 (downloaded by nilearn)
│   ├── features/                       functional_features_niak.csv, functional_edges.csv, …
│   ├── freesurfer/                     structural reconstructions (pending)
│   └── fmriprep/                       only if the raw route is ever used
├── raw/                                raw archives / BIDS data (not tracked)
└── qc/structural_visual_qc.csv         participant_id, qc_pass (1/0), rater, notes
```
The functional archive itself stays outside the repository (currently `C:/Users/diksh/Downloads/1160600.zip`, path set in `configs/data_config.yaml`). Nothing is unzipped: notebook 03b decompresses each participant in memory.

## Audit checklist
- [x] Functional release checksummed and recorded.
- [ ] Map raw diagnosis values to `label_map` in `configs/data_config.yaml` once the official phenotype file arrives (the NIAK table carries no subtype).
- [ ] Confirm the motion rule (D8): ≥ 180 s or ≥ 120 s retained.
- [ ] Record the structural pipeline and version (D3b), and the resulting feature count (D5).
- [ ] Complete the structural provenance table above.
- [ ] Review `results/metrics/sample_flow.csv` before any confirmatory model is fitted.
