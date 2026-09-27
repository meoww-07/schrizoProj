"""Repeated nested cross-validation and permutation testing (protocol §5.3, §7).

    python -m src.validation run                        # all models, observed labels
    python -m src.validation permute-structural         # primary incremental-value null
    python -m src.validation permute-labels             # each model vs chance
    python -m src.validation permute-labels --start 0 --n-permutations 100   # one chunk

Every model sees identical outer and inner splits, so outer-fold predictions
are paired across models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from functools import partial
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from src.metrics import mean_auc_by_model, mean_paired_auc_difference
from src.models import build_models

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "nilearn", "nibabel", "joblib", "PyYAML")


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_seeds(path):
    lines = (line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
    return [int(line) for line in lines if line and not line.startswith("#")]


def outer_splits(y, n_splits, seeds):
    """(repeat, seed, fold, train_idx, test_idx) for every repeat; shared by all models."""
    splits = []
    for repeat, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
            splits.append((repeat, seed, fold, train, test))
    return splits


def inner_splits(y_train, n_splits, seed, fold):
    """Stratified splits of one outer training fold (indices relative to that fold)."""
    inner_seed = int(np.random.SeedSequence([seed, fold]).generate_state(1)[0])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=inner_seed)
    return list(cv.split(np.zeros(len(y_train)), y_train))


def run_outer_fold(X, y, ids, split, models, n_inner, collect_weights=False):
    repeat, seed, fold, train, test = split
    inner = inner_splits(y[train], n_inner, seed, fold)
    predictions, selections, weights = [], [], {}
    for model in models:
        model.fit(X[train], y[train], inner)
        score = model.decision_function(X[test])
        predictions.append(pd.DataFrame({
            "model": model.name,
            "repeat": repeat,
            "seed": seed,
            "fold": fold,
            "participant_id": ids[test],
            "y_true": y[test],
            "score": score,
            "threshold": model.threshold_,
            "y_pred": (score >= model.threshold_).astype(int),
        }))
        selections.append({"model": model.name, "repeat": repeat, "fold": fold,
                           "n_train": len(train), "n_test": len(test), **model.selection()})
        if collect_weights and hasattr(model, "primal_weights"):
            for modality, w in model.primal_weights().items():
                weights[f"{model.name}__r{repeat}__f{fold}__{modality}"] = w
    return pd.concat(predictions, ignore_index=True), selections, weights


def run_nested_cv(X, y, ids, model_factory, seeds, n_outer=5, n_inner=5, n_jobs=1,
                  collect_weights=False):
    """Run every model from ``model_factory()`` through repeated nested CV.

    Returns long-format outer-test predictions, per-fold selections (C, beta,
    threshold, inner AUC) and, optionally, primal weight vectors.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    ids = np.asarray(ids)
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_outer_fold)(X, y, ids, split, model_factory(), n_inner, collect_weights)
        for split in outer_splits(y, n_outer, seeds)
    )
    predictions = pd.concat([r[0] for r in results], ignore_index=True)
    selections = pd.DataFrame([row for r in results for row in r[1]])
    weights = {key: w for r in results for key, w in r[2].items()}
    return predictions, selections, weights


def permuted_dataset(X, y, kind, index, seed, block=None):
    """Permutation ``index`` of the data, from its own RNG stream.

    kind="labels": permute diagnosis labels (null: no label-imaging association).
    kind="block": permute the rows of one feature block across participants,
    keeping labels and every other block fixed (null: the block is uninformative).
    """
    perm = np.random.default_rng([seed, index]).permutation(len(y))
    if kind == "labels":
        return X, y[perm]
    if kind == "block":
        X_perm = X.copy()
        X_perm[:, block] = X[np.ix_(perm, block)]
        return X_perm, y
    raise ValueError(f"Unknown permutation kind {kind!r}")


def run_permutations(X, y, ids, model_factory, seeds, statistic, kind, indices, seed,
                     block=None, n_outer=5, n_inner=5, n_jobs=1):
    """Null distribution of ``statistic(predictions)``, rerunning the full nested pipeline."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    def one(index):
        X_perm, y_perm = permuted_dataset(X, y, kind, index, seed, block)
        predictions, _, _ = run_nested_cv(X_perm, y_perm, ids, model_factory, seeds, n_outer, n_inner)
        return statistic(predictions)

    indices = list(indices)
    values = Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in indices)
    rows = [{"permutation": i, **(v.to_dict() if isinstance(v, pd.Series) else {"statistic": float(v)})}
            for i, v in zip(indices, values)]
    return pd.DataFrame(rows)


def load_design(data_config, functional_strategy="primary", strict_patients=False):
    """Design matrix with column blocks structural | functional | confounds.

    Only participants with ``included == 1`` in the manifest are used. Any
    missing value is an error: the protocol forbids imputation (§2.5).
    """
    features = ROOT / data_config["features_dir"]
    manifest = pd.read_csv(ROOT / data_config["manifest"], dtype={"participant_id": str})
    manifest = manifest[manifest["included"] == 1]
    if strict_patients:
        manifest = manifest[(manifest["label"] == 0) | (manifest["strict_patient"] == 1)]

    structural = pd.read_csv(features / "structural_features.csv", dtype={"participant_id": str})
    functional = pd.read_csv(features / f"functional_features_{functional_strategy}.csv",
                             dtype={"participant_id": str})
    tables = [structural, functional,
              pd.read_csv(features / "structural_qc.csv", dtype={"participant_id": str}),
              pd.read_csv(features / "functional_qc_primary.csv", dtype={"participant_id": str})]
    design = manifest
    for table in tables:
        design = design.merge(table, on="participant_id", how="left", validate="one_to_one")

    columns = {
        "structural": [c for c in structural.columns if c != "participant_id"],
        "functional": [c for c in functional.columns if c != "participant_id"],
        "confounds": list(data_config["confound_columns"]),
    }
    ordered = columns["structural"] + columns["functional"] + columns["confounds"]
    X = design[ordered].to_numpy(dtype=float)
    if np.isnan(X).any():
        bad = design.loc[np.isnan(X).any(axis=1), "participant_id"].tolist()
        raise ValueError(f"Missing values for included participants (no imputation allowed): {bad}")

    blocks, start = {}, 0
    for name in ("structural", "functional", "confounds"):
        blocks[name] = np.arange(start, start + len(columns[name]))
        start += len(columns[name])
    return X, design["label"].to_numpy(int), design["participant_id"].to_numpy(), blocks, columns


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "permute-structural", "permute-labels"])
    parser.add_argument("--models", nargs="+", help="subset of model names (run only)")
    parser.add_argument("--functional-strategy", default="primary",
                        help="denoising variant of the functional features (protocol §8)")
    parser.add_argument("--strict-patients", action="store_true",
                        help="controls + strict schizophrenia only (protocol §8)")
    parser.add_argument("--tag", default="", help="suffix for output files of sensitivity runs")
    parser.add_argument("--start", type=int, default=0, help="first permutation index")
    parser.add_argument("--n-permutations", type=int, help="permutations in this chunk")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    configs = {name: ROOT / "configs" / f"{name}.yaml" for name in ("model_config", "cv_config", "data_config")}
    model_cfg, cv_cfg, data_cfg = (load_yaml(configs[n]) for n in ("model_config", "cv_config", "data_config"))
    seeds = read_seeds(ROOT / cv_cfg["seeds_file"])
    if len(seeds) != cv_cfg["outer"]["n_repeats"]:
        raise ValueError("Number of seeds does not match outer.n_repeats.")
    n_outer, n_inner = cv_cfg["outer"]["n_splits"], cv_cfg["inner"]["n_splits"]
    X, y, ids, blocks, columns = load_design(data_cfg, args.functional_strategy, args.strict_patients)
    suffix = f"_{args.tag}" if args.tag else ""

    if args.command == "run":
        names = args.models or [m["name"] for m in model_cfg["models"]]
        factory = partial(build_models, model_cfg, blocks, include=names)
        start = time.perf_counter()
        predictions, selections, weights = run_nested_cv(
            X, y, ids, factory, seeds, n_outer, n_inner, n_jobs=args.n_jobs, collect_weights=True)
        elapsed = time.perf_counter() - start

        (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
        (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
        predictions.to_csv(RESULTS / "predictions" / f"predictions{suffix}.csv", index=False)
        selections.to_csv(RESULTS / "metrics" / f"selections{suffix}.csv", index=False)
        np.savez_compressed(RESULTS / "predictions" / f"primal_weights{suffix}.npz", **weights)
        run_info = {
            "command": "run", "tag": args.tag, "models": names,
            "functional_strategy": args.functional_strategy, "strict_patients": args.strict_patients,
            "n_participants": int(len(y)), "n_patients": int(y.sum()), "n_controls": int((y == 0).sum()),
            "n_features": {name: len(cols) for name, cols in columns.items()},
            "seeds": seeds, "n_outer": n_outer, "n_inner": n_inner,
            "elapsed_seconds": round(elapsed, 1),
            "sha256": {p.name: file_sha256(p) for p in [*configs.values(), ROOT / cv_cfg["seeds_file"],
                                                         ROOT / "protocol" / "protocol.md"]},
            "packages": package_versions(), "python": platform.python_version(),
            "platform": platform.platform(),
        }
        (RESULTS / "metrics" / f"run_info{suffix}.json").write_text(json.dumps(run_info, indent=2))
        print(f"{len(names)} models, {len(y)} participants, {elapsed:.0f} s. "
              f"Next: python -m src.metrics{' --tag ' + args.tag if args.tag else ''}")
        return

    perm_cfg = cv_cfg["permutation"]
    n_perm = args.n_permutations or perm_cfg["n_permutations"]
    indices = range(args.start, args.start + n_perm)
    if args.command == "permute-structural":
        a, b = model_cfg["primary_comparison"]["model"], model_cfg["primary_comparison"]["comparator"]
        factory = partial(build_models, model_cfg, blocks, include=[a, b])
        statistic = partial(mean_paired_auc_difference, model_a=a, model_b=b)
        kind, block, null_name = "block", blocks["structural"], "structural_block"
    else:
        names = [m["name"] for m in model_cfg["models"] if m["role"] != "sensitivity"]
        factory = partial(build_models, model_cfg, blocks, include=names)
        statistic, kind, block, null_name = mean_auc_by_model, "labels", None, "labels"

    start = time.perf_counter()
    null = run_permutations(X, y, ids, factory, seeds, statistic, kind, indices, perm_cfg["seed"],
                            block=block, n_outer=n_outer, n_inner=n_inner, n_jobs=args.n_jobs)
    out_dir = RESULTS / "permutation_tests" / f"{null_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chunk_{indices.start:04d}-{indices.stop - 1:04d}.csv"
    null.to_csv(out, index=False)
    print(f"{len(null)} permutations in {time.perf_counter() - start:.0f} s -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
