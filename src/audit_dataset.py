"""Dataset audit (protocol §2): participant manifest, sample flow, descriptives.

Run once the raw BIDS data and fMRIPrep/FreeSurfer derivatives are in place,
and before any model is fitted:

    python -m src.audit_dataset

Writes
    data/participants_manifest.csv      one row per phenotype participant, all exclusion reasons
    results/metrics/sample_flow.csv     initial N -> exclusions -> final N, patients, controls
    results/metrics/sample_description.csv
    results/metrics/acquisition_summary.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Exclusion reasons, in the order used by the sequential flow table (protocol §2.3).
REASONS = (
    "missing_label",
    "excluded_label",
    "missing_covariate",
    "missing_structural",
    "missing_functional",
    "failed_structural_qc",
    "failed_functional_qc",
)
FS_STATS = ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats")
ACQ_FIELDS = ("Manufacturer", "ManufacturersModelName", "MagneticFieldStrength",
              "SoftwareVersions", "RepetitionTime", "EchoTime")
# Used when configs/data_config.yaml has no ``missing_values`` entry.
DEFAULT_MISSING_VALUES = ("", "na", "n/a", "nan", "none", "null", "unknown", "not reported", "missing", "-", "?")
LABEL_CATEGORIES = {"patient", "control", "exclude"}
SEX_CATEGORIES = {"male", "female"}


class UnmappedValueError(ValueError):
    """A non-missing phenotype value that the config does not recognize."""

    def __init__(self, field, raw, problem="not listed in the config"):
        super().__init__(f"{field}: {raw!r} is {problem}")
        self.field, self.raw = field, raw


def _canonical(value):
    """Comparable text: integral numbers without decimals, whitespace collapsed, case folded."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).casefold()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return str(int(value)) if value.is_integer() else repr(value)
    return " ".join(str(value).split()).casefold()


def normalize_value(raw, missing_values=DEFAULT_MISSING_VALUES):
    """Canonical form of a raw phenotype value, or None if it is missing.

    Missing means None, NaN/NA, an empty or whitespace-only string, or any token in
    ``missing_values`` (compared case- and whitespace-insensitively).
    """
    if raw is None or (np.ndim(raw) == 0 and pd.isna(raw)):
        return None
    text = _canonical(raw)
    return None if text in {_canonical(v) for v in missing_values} else text


def value_lookup(mapping, missing_values=DEFAULT_MISSING_VALUES):
    """{canonical raw value: category}; rejects values listed twice or listed as missing tokens."""
    lookup = {}
    for category, values in mapping.items():
        for value in values or []:
            key = normalize_value(value, missing_values)
            if key is None:
                raise ValueError(f"{value!r} (under {category!r}) is a missing-value token and cannot be a category.")
            if lookup.get(key, category) != category:
                raise ValueError(f"{value!r} is mapped to both {lookup[key]!r} and {category!r}.")
            lookup[key] = category
    return lookup


def map_value(raw, mapping, field="value", missing_values=DEFAULT_MISSING_VALUES):
    """Category of a raw phenotype value (protocol §2.4).

    Returns None if the value is missing, and raises UnmappedValueError if it is
    non-missing but not listed in ``mapping``: unexpected values are never silently
    treated as missing.
    """
    key = normalize_value(raw, missing_values)
    if key is None:
        return None
    lookup = value_lookup(mapping, missing_values)
    if key not in lookup:
        raise UnmappedValueError(field, raw)
    return lookup[key]


def parse_number(raw, field="value", missing_values=DEFAULT_MISSING_VALUES):
    """Float value, None if missing; UnmappedValueError if the value is not a number."""
    if normalize_value(raw, missing_values) is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise UnmappedValueError(field, raw, "not a number") from None


def bids_id(raw):
    raw = str(raw).strip()
    return raw if raw.startswith("sub-") else f"sub-{raw}"


def _stem(nifti):
    name = nifti.name
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def sidecar(nifti, bids_dir, suffix):
    """Acquisition fields from the BIDS JSON sidecar, with top-level inheritance."""
    fields = {}
    for path in (bids_dir / f"task-rest_{suffix}.json" if suffix == "bold" else None,
                 nifti.parent / f"{_stem(nifti)}.json"):
        if path is not None and path.exists():
            fields.update(json.loads(path.read_text()))
    return {key: fields.get(key) for key in ACQ_FIELDS}


def motion_summary(preproc_bold, denoise_kwargs, qc):
    """Mean/max FD, retained volumes after censoring, and the functional QC decision."""
    import nibabel as nib
    from nilearn.interfaces.fmriprep import load_confounds_strategy

    prefix = preproc_bold.name.split("_space-")[0]
    confounds = pd.read_csv(preproc_bold.parent / f"{prefix}_desc-confounds_timeseries.tsv", sep="\t")
    fd = confounds["framewise_displacement"].to_numpy(float)
    _, sample_mask = load_confounds_strategy(str(preproc_bold), **denoise_kwargs)
    n_volumes = len(confounds)
    n_retained = n_volumes if sample_mask is None else len(sample_mask)
    tr = float(nib.load(str(preproc_bold)).header.get_zooms()[3])
    summary = {
        "mean_fd": float(np.nanmean(fd)),
        "max_fd": float(np.nanmax(fd)),
        "n_volumes": n_volumes,
        "n_retained": n_retained,
        "tr": tr,
        "retained_seconds": n_retained * tr,
    }
    summary["func_qc_pass"] = bool(summary["mean_fd"] <= qc["max_mean_fd_mm"]
                                   and summary["max_fd"] <= qc["max_fd_mm"]
                                   and summary["retained_seconds"] >= qc["min_retained_seconds"])
    return summary


def load_visual_qc(path):
    """{participant_id: 1, 0 or NaN}; qc_pass must be 1 (pass), 0 (fail) or empty (not rated)."""
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype={"participant_id": str})
    ratings = [normalize_value(v) for v in table["qc_pass"]]
    bad = [(pid, raw) for pid, raw, r in zip(table["participant_id"], table["qc_pass"], ratings)
           if r is not None and r not in ("0", "1")]
    if bad:
        raise ValueError(f"qc_pass must be 1, 0 or empty; found: {bad}")
    return {bids_id(pid): (np.nan if r is None else int(r)) for pid, r in zip(table["participant_id"], ratings)}


def build_manifest(cfg):
    """One row per phenotype participant; stops on any unrecognized phenotype value."""
    ph = cfg["phenotype"]
    missing_values = cfg.get("missing_values", DEFAULT_MISSING_VALUES)
    label_map, sex_map = cfg["label_map"], cfg["sex_map"]
    pheno = pd.read_csv(ROOT / ph["path"], sep=ph.get("sep", ","), dtype={ph["id_column"]: str})
    pheno.columns = [str(c).strip() for c in pheno.columns]
    if not label_map.get("patient") or not label_map.get("control"):
        print(pheno[ph["diagnosis_column"]].value_counts(dropna=False).to_string())
        raise SystemExit("label_map in configs/data_config.yaml is empty. Map the raw diagnosis "
                         "values above to patient/control (protocol §2.4), then rerun.")
    for name, mapping, allowed in [("label_map", label_map, LABEL_CATEGORIES), ("sex_map", sex_map, SEX_CATEGORIES)]:
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"{name} has unknown categories {sorted(unknown)}; allowed: {sorted(allowed)}")
        value_lookup(mapping, missing_values)

    bids = ROOT / cfg["bids_dir"]
    fmriprep = ROOT / cfg["fmriprep_dir"]
    freesurfer = ROOT / cfg["freesurfer_dir"]
    func, qc = cfg["functional"], cfg["qc"]
    visual_qc = load_visual_qc(ROOT / qc["structural_visual_qc"])
    subtype_column = ph.get("subtype_column") or ph["diagnosis_column"]
    strict_keys = {normalize_value(v, missing_values) for v in cfg.get("strict_patient_values") or []} - {None}

    rows, problems = [], Counter()
    seen = {"diagnosis": set(), "sex": set(), "subtype": set()}
    for _, record in pheno.iterrows():
        pid = bids_id(record[ph["id_column"]])
        mapped = {}
        for field, column, mapping in [("diagnosis", ph["diagnosis_column"], label_map), ("sex", ph["sex_column"], sex_map)]:
            seen[field].add(normalize_value(record[column], missing_values))
            try:
                mapped[field] = map_value(record[column], mapping, field, missing_values)
            except UnmappedValueError as err:
                problems[(err.field, err.raw)] += 1
                mapped[field] = None
        try:
            age = parse_number(record[ph["age_column"]], "age", missing_values)
        except UnmappedValueError as err:
            problems[(err.field, err.raw)] += 1
            age = None
        subtype = normalize_value(record[subtype_column], missing_values)
        seen["subtype"].add(subtype)

        label_status = {"patient": "patient", "control": "control", "exclude": "excluded", None: "missing"}[mapped["diagnosis"]]
        if label_status == "patient":
            strict = np.nan if subtype is None else int(subtype in strict_keys)
        else:
            strict = 0
        row = {
            "participant_id": pid,
            "diagnosis_raw": record[ph["diagnosis_column"]],
            "label_status": label_status,
            "label": {"patient": 1, "control": 0}.get(label_status, np.nan),
            "strict_patient": strict,
            "age": np.nan if age is None else age,
            "sex_raw": record[ph["sex_column"]],
            "sex_male": {"male": 1, "female": 0}.get(mapped["sex"], np.nan),
            "mean_fd": np.nan, "max_fd": np.nan, "n_volumes": np.nan, "n_retained": np.nan,
            "tr": np.nan, "retained_seconds": np.nan, "func_qc_pass": False,
        }
        if ph.get("medication_column"):
            row["medication_raw"] = record[ph["medication_column"]]

        t1 = sorted((bids / pid).glob("**/anat/*_T1w.nii*"))
        bold = sorted((bids / pid).glob(f"**/func/*task-{func['task']}*_bold.nii*"))
        preproc = sorted((fmriprep / pid).glob(
            f"**/func/*task-{func['task']}*space-{func['space']}*res-{func['resolution']}"
            "*desc-preproc_bold.nii.gz"))
        row.update(n_t1=len(t1), n_bold=len(bold), n_preproc_bold=len(preproc))
        if t1:
            row.update({f"t1_{k}": v for k, v in sidecar(t1[0], bids, "T1w").items()})
        if bold:
            row.update({f"bold_{k}": v for k, v in sidecar(bold[0], bids, "bold").items()})
        row["fs_complete"] = all((freesurfer / pid / "stats" / f).exists() for f in FS_STATS)
        row["struct_visual_qc"] = visual_qc.get(pid, np.nan)
        if len(preproc) == 1:
            row.update(motion_summary(preproc[0], func["denoise"]["primary"], qc))
        rows.append(row)

    if problems:
        listing = "\n".join(f"  {field}: {raw!r}  ({n} participants)"
                            for (field, raw), n in sorted(problems.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))))
        raise ValueError("Unrecognized phenotype values. Map each one in configs/data_config.yaml "
                         "(label_map / sex_map), add it to missing_values, or list it under "
                         "label_map.exclude, then rerun:\n" + listing)
    for field, mapping in [("diagnosis", label_map), ("sex", sex_map)]:
        unused = [v for values in mapping.values() for v in values or []
                  if normalize_value(v, missing_values) not in seen[field]]
        if unused:
            print(f"Warning: configured {field} values never found in the data (typo?): {unused}")
    if strict_keys - seen["subtype"]:
        print(f"Warning: strict_patient_values never found in the data (typo?): {sorted(strict_keys - seen['subtype'])}")
    return add_exclusions(pd.DataFrame(rows))


def add_exclusions(manifest):
    """Boolean ``excl_*`` column per reason, all reasons joined, and ``included``."""
    m = manifest.copy()
    if "label_status" in m:
        status = m["label_status"]
    else:
        status = pd.Series(np.where(m["label"].isna(), "missing", "labelled"), index=m.index)
    reasons = {
        "missing_label": status.eq("missing"),
        "excluded_label": status.eq("excluded"),
        "missing_covariate": m["age"].isna() | m["sex_male"].isna(),
        "missing_structural": m["n_t1"] < 1,
        "missing_functional": m["n_bold"] < 1,
        "failed_structural_qc": ~m["fs_complete"].eq(True) | ~m["struct_visual_qc"].eq(1),
        "failed_functional_qc": m["n_preproc_bold"].ne(1) | ~m["func_qc_pass"].eq(True),
    }
    for reason in REASONS:
        m[f"excl_{reason}"] = reasons[reason].to_numpy(bool)
    m["exclusion_reasons"] = [";".join(r for r in REASONS if m.at[i, f"excl_{r}"]) for i in m.index]
    m["included"] = (m["exclusion_reasons"] == "").astype(int)
    return m


def sample_flow(manifest):
    """Sequential flow table: each participant counted under their first reason."""
    remaining = np.ones(len(manifest), dtype=bool)
    rows = [("initial_N", len(manifest))]
    for reason in REASONS:
        hit = manifest[f"excl_{reason}"].to_numpy(bool)
        rows.append((f"excluded_{reason}", int((remaining & hit).sum())))
        remaining &= ~hit
    final = manifest.loc[remaining]
    rows += [("final_N", len(final)),
             ("patients", int((final["label"] == 1).sum())),
             ("controls", int((final["label"] == 0).sum())),
             ("strict_patients", int(final["strict_patient"].sum()))]
    return pd.DataFrame(rows, columns=["step", "n"])


def describe_sample(manifest, stringent_fd):
    included = manifest[manifest["included"] == 1]
    rows = []
    for label, group in included.groupby("label"):
        rows.append({
            "group": "patient" if label == 1 else "control",
            "n": len(group),
            "age_mean": group["age"].mean(), "age_sd": group["age"].std(),
            "male_n": int(group["sex_male"].sum()), "male_pct": 100 * group["sex_male"].mean(),
            "mean_fd_mean": group["mean_fd"].mean(), "mean_fd_sd": group["mean_fd"].std(),
            "retained_seconds_mean": group["retained_seconds"].mean(),
            f"n_mean_fd_above_{stringent_fd}": int((group["mean_fd"] > stringent_fd).sum()),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_config.yaml"))
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    manifest = build_manifest(cfg)
    manifest.to_csv(ROOT / cfg["manifest"], index=False)
    metrics = ROOT / "results" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    flow = sample_flow(manifest)
    flow.to_csv(metrics / "sample_flow.csv", index=False)
    describe_sample(manifest, cfg["qc"]["stringent_mean_fd_mm"]).to_csv(
        metrics / "sample_description.csv", index=False)
    acq = [c for c in manifest.columns if c.startswith(("t1_", "bold_"))]
    if acq:
        (manifest.loc[manifest["included"] == 1, acq].astype(str).value_counts()
         .rename("n").reset_index().to_csv(metrics / "acquisition_summary.csv", index=False))
    print(flow.to_string(index=False))
    if manifest["struct_visual_qc"].isna().any():
        print(f"\n{int(manifest['struct_visual_qc'].isna().sum())} participants have no structural "
              f"visual QC rating in {cfg['qc']['structural_visual_qc']} and are counted as failed.")


if __name__ == "__main__":
    main()
