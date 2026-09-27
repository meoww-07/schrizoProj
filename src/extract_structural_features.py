"""Structural features Xs from FreeSurfer stats files (protocol §3.2).

    python -m src.extract_structural_features

For every participant with a complete reconstruction in the manifest:
Desikan-Killiany thickness (68) and surface area (68) plus 14 aseg subcortical
volumes; area and volume are divided by eTIV. Writes
data/derivatives/features/structural_features.csv and structural_qc.csv
(eTIV, surface holes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import normalize_by_etiv

ROOT = Path(__file__).resolve().parents[1]
N_DK_REGIONS = 34
SUBCORTICAL = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens-area")
ALIASES = {"Thalamus": ("Thalamus", "Thalamus-Proper")}  # FreeSurfer < 7 uses Thalamus-Proper


def read_stats(path):
    """``# Measure`` values and the region table of a FreeSurfer ``*.stats`` file."""
    measures, header, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# Measure "):
            parts = [p.strip() for p in line[len("# Measure "):].split(",")]
            measures[parts[1]] = float(parts[-2])
        elif line.startswith("# ColHeaders"):
            header = line.split()[2:]
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split())
    table = pd.DataFrame(rows, columns=header)
    for column in table.columns:
        if column != "StructName":
            table[column] = pd.to_numeric(table[column])
    return measures, table


def participant_features(subject_dir):
    stats = Path(subject_dir) / "stats"
    features = {}
    for hemi in ("lh", "rh"):
        _, table = read_stats(stats / f"{hemi}.aparc.stats")
        if len(table) != N_DK_REGIONS:
            raise ValueError(f"{stats / f'{hemi}.aparc.stats'}: {len(table)} regions, expected {N_DK_REGIONS}")
        for _, region in table.iterrows():
            features[f"thick_{hemi}_{region['StructName']}"] = float(region["ThickAvg"])
            features[f"area_{hemi}_{region['StructName']}"] = float(region["SurfArea"])

    measures, aseg = read_stats(stats / "aseg.stats")
    volumes = dict(zip(aseg["StructName"], aseg["Volume_mm3"].astype(float)))
    for side in ("Left", "Right"):
        for structure in SUBCORTICAL:
            name = next((f"{side}-{alias}" for alias in ALIASES.get(structure, (structure,))
                         if f"{side}-{alias}" in volumes), None)
            if name is None:
                raise ValueError(f"{stats / 'aseg.stats'}: {side}-{structure} not found")
            features[f"vol_{side}-{structure}"] = volumes[name]
    quality = {"eTIV": measures["eTIV"], "surface_holes": measures.get("SurfaceHoles", np.nan)}
    return features, quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_config.yaml"))
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    manifest = pd.read_csv(ROOT / cfg["manifest"], dtype={"participant_id": str})
    participants = manifest.loc[manifest["fs_complete"].eq(True), "participant_id"]
    freesurfer = ROOT / cfg["freesurfer_dir"]
    feature_rows, quality_rows = [], []
    for pid in participants:
        features, quality = participant_features(freesurfer / pid)
        feature_rows.append({"participant_id": pid, **features})
        quality_rows.append({"participant_id": pid, **quality})

    features = pd.DataFrame(feature_rows)
    if features.drop(columns="participant_id").isna().any().any():
        raise ValueError("Participants differ in their FreeSurfer region lists.")
    quality = pd.DataFrame(quality_rows)
    scaled = [c for c in features.columns if c.startswith(("area_", "vol_"))]
    features = normalize_by_etiv(features, scaled, quality["eTIV"])

    out = ROOT / cfg["features_dir"]
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "structural_features.csv", index=False)
    quality.to_csv(out / "structural_qc.csv", index=False)
    counts = {prefix: sum(c.startswith(prefix) for c in features.columns) for prefix in ("thick_", "area_", "vol_")}
    print(f"{len(features)} participants; features: {counts} = {sum(counts.values())} total")


if __name__ == "__main__":
    main()
