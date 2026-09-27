"""Outer-fold performance metrics and inference (protocol §6–§7).

Metrics are computed within each outer test fold and then averaged: decision
scores from different fold models are on different scales, so pooling them
across folds would distort discrimination.

    python -m src.metrics            # after `python -m src.validation run`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FOLD_KEYS = ["model", "repeat", "fold"]
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]


def fold_metrics(predictions, zero_threshold=False):
    """One row per (model, repeat, fold) with every protocol §6 metric.

    ``zero_threshold=True`` scores thresholded metrics at decision score 0
    instead of the inner-CV Youden threshold (sensitivity analysis 7).
    """
    rows = []
    for (model, repeat, fold), g in predictions.groupby(FOLD_KEYS, sort=False):
        y = g["y_true"].to_numpy(int)
        score = g["score"].to_numpy(float)
        y_pred = (score >= 0).astype(int) if zero_threshold else g["y_pred"].to_numpy(int)
        sensitivity = float(np.mean(y_pred[y == 1] == 1))
        specificity = float(np.mean(y_pred[y == 0] == 0))
        rows.append({
            "model": model, "repeat": repeat, "fold": fold,
            "n_test": len(y), "prevalence": float(y.mean()),
            "auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })
    return pd.DataFrame(rows)


def summarize(folds):
    """Mean and SD across outer folds (SD describes partition variability, not an SE)."""
    summary = folds.groupby("model", sort=False)[METRICS + ["prevalence"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def fold_auc(predictions):
    """Outer-fold AUC indexed by (model, repeat, fold); the fast path used by permutations."""
    return pd.Series({key: roc_auc_score(g["y_true"], g["score"])
                      for key, g in predictions.groupby(FOLD_KEYS, sort=False)}, name="auc")


def mean_auc_by_model(predictions):
    return fold_auc(predictions).groupby(level=0).mean()


def _paired(a, b):
    if a.empty or b.empty or set(a.index) != set(b.index):
        raise ValueError("Paired comparison requires both models on identical outer folds.")
    return (a - b.reindex(a.index)).to_numpy(float)


def paired_differences(folds, model_a, model_b, metric="auc"):
    """Fold-level ``metric(model_a) - metric(model_b)`` on identical test participants."""
    def per_model(name):
        return folds[folds["model"] == name].set_index(["repeat", "fold"])[metric]
    return _paired(per_model(model_a), per_model(model_b))


def mean_paired_auc_difference(predictions, model_a, model_b):
    auc = fold_auc(predictions)
    return float(np.mean(_paired(auc.xs(model_a, level=0), auc.xs(model_b, level=0))))


def corrected_resampled_ci(diffs, test_train_ratio, alpha=0.05):
    """Nadeau & Bengio (2003) corrected repeated-CV interval for a mean paired difference.

    The variance factor 1/J + n_test/n_train accounts for overlapping training
    sets across folds. The correction is approximate (protocol §7.2).
    """
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / J + test_train_ratio) * d.var(ddof=1)))
    t_crit = stats.t.ppf(1 - alpha / 2, J - 1)
    if se > 0:
        p = float(2 * stats.t.sf(abs(mean) / se, J - 1))
    else:
        p = 0.0 if mean != 0 else 1.0
    return {"mean": mean, "se": se, "ci_low": mean - t_crit * se, "ci_high": mean + t_crit * se,
            "p_two_sided": p, "n_folds": J}


def permutation_p_value(observed, null, alternative="greater"):
    """Monte Carlo p-value with the plus-one correction: (1 + #extreme) / (1 + B)."""
    null = np.asarray(null, dtype=float)
    if alternative == "greater":
        extreme = np.sum(null >= observed)
    elif alternative == "two-sided":
        extreme = np.sum(np.abs(null) >= abs(observed))
    else:
        raise ValueError(f"Unknown alternative {alternative!r}")
    return float((1 + extreme) / (1 + len(null)))


def holm(pvalues):
    """Holm step-down adjusted p-values for a {name: p} mapping."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def classify_primary_result(mean, ci_low, ci_high, p_perm, delta, alpha=0.05):
    """Prespecified decision rule of protocol §7.3."""
    if p_perm < alpha and ci_low > 0 and mean >= delta:
        return "superiority supported"
    if p_perm < alpha and ci_low > 0:
        return "detectable but below the minimum meaningful difference"
    if ci_high < delta:
        return "no meaningful incremental value"
    return "inconclusive"


def load_null(name, suffix=""):
    """Concatenate permutation chunks written by ``src.validation``; None if absent."""
    chunks = sorted((RESULTS / "permutation_tests" / f"{name}{suffix}").glob("chunk_*.csv"))
    if not chunks:
        return None
    null = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    if null["permutation"].duplicated().any():
        raise ValueError(f"Duplicate permutation indices in {name}{suffix} chunks.")
    return null


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    suffix = f"_{args.tag}" if args.tag else ""

    with open(ROOT / "configs" / "model_config.yaml", encoding="utf-8") as fh:
        model_cfg = yaml.safe_load(fh)
    with open(ROOT / "configs" / "cv_config.yaml", encoding="utf-8") as fh:
        cv_cfg = yaml.safe_load(fh)
    alpha = cv_cfg["inference"]["alpha"]
    delta = cv_cfg["inference"]["min_meaningful_delta_auc"]
    ratio = 1.0 / (cv_cfg["outer"]["n_splits"] - 1)
    out = RESULTS / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(RESULTS / "predictions" / f"predictions{suffix}.csv")
    folds = fold_metrics(predictions)
    folds.to_csv(out / f"fold_metrics{suffix}.csv", index=False)
    summarize(folds).to_csv(out / f"summary{suffix}.csv", index=False)
    summarize(fold_metrics(predictions, zero_threshold=True)).to_csv(
        out / f"summary_threshold0{suffix}.csv", index=False)
    present = set(predictions["model"])

    a, b = model_cfg["primary_comparison"]["model"], model_cfg["primary_comparison"]["comparator"]
    pairs = [("primary", a, b)] + [("secondary", x, z) for x, z in model_cfg["secondary_comparisons"]]
    comparisons = []
    for role, x, z in pairs:
        if {x, z} <= present:
            ci = corrected_resampled_ci(paired_differences(folds, x, z), ratio, alpha)
            comparisons.append({"role": role, "model": x, "comparator": z, **ci})
    comparisons = pd.DataFrame(comparisons)
    secondary = comparisons[comparisons["role"] == "secondary"]
    adjusted = holm(dict(zip(secondary["comparator"], secondary["p_two_sided"])))
    comparisons["p_holm"] = [adjusted.get(z) if r == "secondary" else None
                             for r, z in zip(comparisons["role"], comparisons["comparator"])]
    comparisons.to_csv(out / f"comparisons{suffix}.csv", index=False)

    primary = comparisons[comparisons["role"] == "primary"].iloc[0]
    result = {"comparison": f"{a} - {b}", "mean_delta_auc": primary["mean"],
              "ci_low": primary["ci_low"], "ci_high": primary["ci_high"],
              "delta": delta, "alpha": alpha}
    block_null = load_null("structural_block", suffix)
    if block_null is None:
        result["result"] = "pending: run `python -m src.validation permute-structural`"
    else:
        result["n_permutations"] = len(block_null)
        result["p_block_permutation"] = permutation_p_value(primary["mean"], block_null["statistic"])
        result["result"] = classify_primary_result(primary["mean"], primary["ci_low"], primary["ci_high"],
                                                   result["p_block_permutation"], delta, alpha)
    (out / f"primary_result{suffix}.json").write_text(json.dumps(result, indent=2, default=float))

    label_null = load_null("labels", suffix)
    if label_null is not None:
        observed = mean_auc_by_model(predictions)
        chance = [{"model": m, "mean_auc": observed[m], "n_permutations": len(label_null),
                   "p_permutation": permutation_p_value(observed[m], label_null[m])}
                  for m in observed.index if m in label_null.columns]
        pd.DataFrame(chance).to_csv(out / f"beats_chance{suffix}.csv", index=False)

    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
