"""Functional connectivity features Xf from fMRIPrep derivatives (protocol §3.3).

    python -m src.extract_functional_features                    # primary denoising
    python -m src.extract_functional_features --strategy simple  # sensitivity variants

For every included participant: denoise with nilearn's load_confounds_strategy,
extract Schaefer ROI mean time series, compute Pearson correlations with
np.corrcoef, Fisher-z transform, and keep the upper triangle (k = 1). Writes
data/derivatives/features/functional_features_<strategy>.csv,
functional_edges.csv and functional_qc_<strategy>.csv (median tSNR).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import edge_table, fisher_z, upper_triangle

ROOT = Path(__file__).resolve().parents[1]


def load_atlas(atlas_cfg):
    """(labels image, ROI names) for the configured atlas."""
    if atlas_cfg.get("labels_img"):
        labels = pd.read_csv(ROOT / atlas_cfg["labels_tsv"], sep="\t").sort_values("index")
        image, names = str(ROOT / atlas_cfg["labels_img"]), labels["name"].astype(str).tolist()
    else:
        from nilearn.datasets import fetch_atlas_schaefer_2018

        atlas = fetch_atlas_schaefer_2018(
            n_rois=atlas_cfg["n_rois"], yeo_networks=atlas_cfg["yeo_networks"],
            resolution_mm=atlas_cfg["resolution_mm"], data_dir=str(ROOT / "data" / "derivatives" / "atlases"))
        image = atlas.maps
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in atlas.labels]
    names = [n for n in names if n != "Background"]
    if len(names) != atlas_cfg["n_rois"]:
        raise ValueError(f"Atlas has {len(names)} ROI names, expected {atlas_cfg['n_rois']}")
    return image, names


def roi_network(name):
    """Yeo network from a Schaefer label such as '7Networks_LH_Vis_1'."""
    parts = name.split("_")
    return parts[2] if len(parts) > 3 and parts[0].endswith("Networks") else "NA"


def participant_connectome(bold, atlas_image, denoise_kwargs):
    """Upper-triangle Fisher-z Pearson connectivity and the number of retained volumes."""
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    from nilearn.maskers import NiftiLabelsMasker

    confounds, sample_mask = load_confounds_strategy(str(bold), **denoise_kwargs)
    masker = NiftiLabelsMasker(labels_img=atlas_image, standardize="zscore_sample")
    series = masker.fit_transform(str(bold), confounds=confounds, sample_mask=sample_mask)
    if np.any(series.std(axis=0) == 0):
        raise ValueError(f"{bold.name}: an ROI has a constant time series (atlas coverage).")
    # Plain Pearson correlation, as specified; ConnectivityMeasure would apply its covariance estimator.
    r = np.corrcoef(series, rowvar=False)
    return upper_triangle(fisher_z(r)), series.shape


def temporal_snr(bold, mask):
    """Median voxelwise mean/SD over time within the brain mask (before denoising)."""
    from nilearn.masking import apply_mask

    data = apply_mask(str(bold), str(mask))
    mean, sd = data.mean(axis=0), data.std(axis=0)
    keep = sd > 0
    return float(np.median(mean[keep] / sd[keep]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_config.yaml"))
    parser.add_argument("--strategy", default="primary", help="key under functional.denoise")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    func = cfg["functional"]
    denoise_kwargs = func["denoise"][args.strategy]

    manifest = pd.read_csv(ROOT / cfg["manifest"], dtype={"participant_id": str})
    participants = manifest.loc[manifest["included"] == 1, "participant_id"]
    atlas_image, names = load_atlas(func["atlas"])
    edges = edge_table(names, [roi_network(n) for n in names])

    fmriprep = ROOT / cfg["fmriprep_dir"]
    feature_rows, quality_rows = [], []
    for pid in participants:
        (bold,) = sorted((fmriprep / pid).glob(
            f"**/func/*task-{func['task']}*space-{func['space']}*res-{func['resolution']}"
            "*desc-preproc_bold.nii.gz"))
        vector, shape = participant_connectome(bold, atlas_image, denoise_kwargs)
        if shape[1] != len(names):
            raise ValueError(f"{pid}: {shape[1]} ROI time series, expected {len(names)}")
        feature_rows.append(pd.Series(vector, index=edges["feature"], name=pid))
        mask = Path(str(bold).replace("desc-preproc_bold", "desc-brain_mask"))
        quality_rows.append({"participant_id": pid, "tsnr_median": temporal_snr(bold, mask)})
        print(f"{pid}: {shape[0]} volumes retained")

    features = pd.DataFrame(feature_rows).rename_axis("participant_id").reset_index()
    out = ROOT / cfg["features_dir"]
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / f"functional_features_{args.strategy}.csv", index=False)
    edges.to_csv(out / "functional_edges.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(out / f"functional_qc_{args.strategy}.csv", index=False)
    print(f"{len(features)} participants x {len(edges)} edges ({args.strategy})")


if __name__ == "__main__":
    main()
