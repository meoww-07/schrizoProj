import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.metrics import (classify_primary_result, corrected_resampled_ci, fold_metrics, holm,
                         mean_paired_auc_difference, paired_differences, permutation_p_value)


def predictions_frame():
    rows = []
    for model, shift in (("a", 1.0), ("b", 0.0)):
        for fold in range(2):
            y = np.array([0, 0, 1, 1])
            score = np.array([-1.0, 0.5, 0.2, 1.0]) + shift * y
            rows.append(pd.DataFrame({"model": model, "repeat": 0, "fold": fold, "participant_id": range(4),
                                      "y_true": y, "score": score, "y_pred": (score >= 0.6).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_fold_metrics_values():
    folds = fold_metrics(predictions_frame())
    b = folds[folds["model"] == "b"].iloc[0]
    assert b["auc"] == pytest.approx(0.75)
    assert b["sensitivity"] == pytest.approx(0.5)   # scores 0.2, 1.0 at threshold 0.6
    assert b["specificity"] == pytest.approx(1.0)
    zero = fold_metrics(predictions_frame(), zero_threshold=True)
    assert zero[zero["model"] == "b"].iloc[0]["specificity"] == pytest.approx(0.5)


def test_paired_differences_align_on_folds():
    preds = predictions_frame()
    diffs = paired_differences(fold_metrics(preds), "a", "b")
    np.testing.assert_allclose(diffs, [0.25, 0.25])
    assert mean_paired_auc_difference(preds, "a", "b") == pytest.approx(0.25)


def test_paired_differences_reject_mismatched_folds():
    folds = fold_metrics(predictions_frame())
    folds = folds[~((folds["model"] == "b") & (folds["fold"] == 1))]
    with pytest.raises(ValueError):
        paired_differences(folds, "a", "b")


def test_plus_one_permutation_p_value():
    assert permutation_p_value(1.0, np.zeros(999)) == pytest.approx(1 / 1000)
    assert permutation_p_value(0.0, np.zeros(9)) == pytest.approx(1.0)


def test_corrected_interval_is_wider_than_naive():
    d = np.random.default_rng(0).normal(0.02, 0.05, size=50)
    ci = corrected_resampled_ci(d, test_train_ratio=0.25)
    naive_half_width = stats.t.ppf(0.975, 49) * d.std(ddof=1) / np.sqrt(50)
    assert (ci["ci_high"] - ci["ci_low"]) / 2 > naive_half_width
    assert ci["mean"] == pytest.approx(d.mean())


def test_holm_adjustment():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


@pytest.mark.parametrize("mean, low, high, p, expected", [
    (0.05, 0.01, 0.09, 0.01, "superiority supported"),
    (0.01, 0.002, 0.018, 0.01, "detectable but below the minimum meaningful difference"),
    (-0.01, -0.03, 0.01, 0.40, "no meaningful incremental value"),
    (0.02, -0.01, 0.05, 0.08, "inconclusive"),
])
def test_primary_decision_rule(mean, low, high, p, expected):
    assert classify_primary_result(mean, low, high, p, delta=0.02) == expected
