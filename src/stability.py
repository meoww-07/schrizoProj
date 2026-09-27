"""Prediction and feature stability (protocol §7.5).

Prediction stability: every participant is tested once per repeat, so each has
one out-of-sample score per repeat; we report the mean and SD of that score and
how often the participant is classified as a patient.

Feature stability: there is no feature-selection step, so stability is assessed
on the primal weight vectors (cosine similarity, sign consistency) and on
network-level aggregates. Correlated features can trade weight between folds,
so none of this is read as biological specificity.

    python -m src.stability          # after `python -m src.validation run`
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def prediction_stability(predictions, model):
    subset = predictions[predictions["model"] == model]
    return (subset.groupby("participant_id")
            .agg(y_true=("y_true", "first"),
                 mean_score=("score", "mean"),
                 sd_score=("score", "std"),
                 n_scores=("score", "size"),
                 frac_predicted_patient=("y_pred", "mean"))
            .reset_index())


def load_weights(path):
    """{(model, repeat, fold, modality): vector} from ``primal_weights*.npz``."""
    out = {}
    with np.load(path) as data:
        for key in data.files:
            model, repeat, fold, modality = key.split("__")
            out[(model, int(repeat[1:]), int(fold[1:]), modality)] = data[key]
    return out


def weight_matrix(weights, model, modality):
    keys = sorted(k for k in weights if k[0] == model and k[3] == modality)
    return np.vstack([weights[k] for k in keys]) if keys else np.empty((0, 0))


def cosine_similarity_matrix(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    unit = W / np.where(norms > 0, norms, 1.0)
    return unit @ unit.T


def mean_pairwise_similarity(W):
    S = cosine_similarity_matrix(W)
    return float(S[np.triu_indices(len(W), k=1)].mean())


def sign_consistency(W):
    """Per feature: fraction of fits whose weight has the sign of the mean weight."""
    reference = np.sign(W.mean(axis=0))
    return (np.sign(W) == reference).mean(axis=0)


def selection_frequency(masks):
    """Per-feature selection frequency, for exploratory analyses that select features."""
    return np.asarray(masks, dtype=float).mean(axis=0)


def network_aggregate(w, edges):
    """Mean signed and mean absolute edge weight per (unordered) network pair."""
    pairs = [tuple(sorted(p)) for p in zip(edges["network_i"], edges["network_j"])]
    table = pd.DataFrame({"network_a": [p[0] for p in pairs], "network_b": [p[1] for p in pairs],
                          "weight": w, "abs_weight": np.abs(w)})
    return (table.groupby(["network_a", "network_b"])
            .agg(n_edges=("weight", "size"), mean_weight=("weight", "mean"),
                 mean_abs_weight=("abs_weight", "mean"))
            .reset_index())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS / "metrics" / f"stability{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(RESULTS / "predictions" / f"predictions{suffix}.csv")
    for model in predictions["model"].unique():
        prediction_stability(predictions, model).to_csv(out / f"prediction_{model}.csv", index=False)

    weights = load_weights(RESULTS / "predictions" / f"primal_weights{suffix}.npz")
    features_dir = ROOT / "data" / "derivatives" / "features"
    names = {
        "structural": pd.read_csv(features_dir / "structural_features.csv", nrows=0).columns[1:],
        "functional": pd.read_csv(features_dir / "functional_features_primary.csv", nrows=0).columns[1:],
    }
    edges = pd.read_csv(features_dir / "functional_edges.csv")
    if list(edges["feature"]) != list(names["functional"]):
        raise ValueError("functional_edges.csv does not match the functional feature columns.")

    summary = []
    for model, modality in sorted({(k[0], k[3]) for k in weights}):
        W = weight_matrix(weights, model, modality)
        if len(W) < 2 or not np.any(W):
            continue
        summary.append({"model": model, "modality": modality, "n_fits": len(W),
                        "mean_pairwise_cosine": mean_pairwise_similarity(W)})
        if modality in names:
            pd.DataFrame({"feature": names[modality], "mean_weight": W.mean(axis=0),
                          "sd_weight": W.std(axis=0, ddof=1),
                          "sign_consistency": sign_consistency(W)}).to_csv(
                out / f"weights_{model}_{modality}.csv", index=False)
        if modality == "functional":
            network_aggregate(W.mean(axis=0), edges).to_csv(out / f"networks_{model}.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "weight_similarity.csv", index=False)
    print(f"Stability tables -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
