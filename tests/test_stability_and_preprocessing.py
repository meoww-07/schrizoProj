import numpy as np
import pandas as pd
import pytest

from src.preprocessing import ConfoundRegressor, edge_table, fisher_z, normalize_by_etiv, upper_triangle
from src.stability import (cosine_similarity_matrix, mean_pairwise_similarity, network_aggregate,
                           prediction_stability, sign_consistency)


def test_confound_regressor_removes_confounds_with_training_slopes():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(100, 2))
    X = C @ np.array([[2.0, 0.0, 1.0], [0.0, -1.0, 0.5]]) + rng.normal(size=(100, 3))
    train = slice(0, 70)
    reg = ConfoundRegressor().fit(X[train], C[train])
    resid = reg.transform(X[train], C[train])
    centered = C[train] - C[train].mean(axis=0)
    np.testing.assert_allclose(centered.T @ (resid - resid.mean(axis=0)), 0, atol=1e-8)
    np.testing.assert_allclose(resid.mean(axis=0), X[train].mean(axis=0))
    # Held-out rows use the training slopes, not slopes refit on themselves.
    held_out = reg.transform(X[70:], C[70:])
    np.testing.assert_allclose(held_out, X[70:] - (C[70:] - reg.confound_mean_) @ reg.coef_)


def test_fisher_z_is_finite_on_the_diagonal():
    assert np.isfinite(fisher_z(np.eye(3))).all()


def test_upper_triangle_order_matches_edge_table():
    matrix = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(upper_triangle(matrix), [1, 2, 3, 6, 7, 11])
    table = edge_table(["a", "b", "c", "d"], ["N1", "N1", "N2", "N2"])
    assert list(table["feature"][:2]) == ["fc_000_001", "fc_000_002"]
    assert len(table) == 6 and table.iloc[-1][["name_i", "name_j"]].tolist() == ["c", "d"]


def test_normalize_by_etiv():
    df = pd.DataFrame({"area_x": [10.0, 20.0], "thick_x": [2.5, 2.6]})
    out = normalize_by_etiv(df, ["area_x"], pd.Series([2.0, 4.0]))
    assert out["area_x"].tolist() == [5.0, 5.0]
    assert out["thick_x"].tolist() == [2.5, 2.6]


def test_prediction_stability_summaries():
    preds = pd.DataFrame({"model": "m", "participant_id": ["p1", "p1", "p2", "p2"],
                          "y_true": [1, 1, 0, 0], "score": [1.0, 3.0, -1.0, -1.0], "y_pred": [1, 0, 0, 0]})
    out = prediction_stability(preds, "m").set_index("participant_id")
    assert out.loc["p1", "mean_score"] == 2.0
    assert out.loc["p1", "frac_predicted_patient"] == 0.5
    assert out.loc["p2", "sd_score"] == 0.0


def test_weight_similarity_and_sign_consistency():
    W = np.array([[1.0, -1.0, 0.5], [2.0, -2.0, 1.0], [1.0, -1.0, -0.5]])
    np.testing.assert_allclose(np.diag(cosine_similarity_matrix(W)), 1.0)
    assert cosine_similarity_matrix(W)[0, 1] == pytest.approx(1.0)
    assert 0 < mean_pairwise_similarity(W) < 1
    np.testing.assert_allclose(sign_consistency(W), [1.0, 1.0, 2 / 3])


def test_network_aggregate_pools_both_edge_orientations():
    edges = pd.DataFrame({"network_i": ["A", "B", "A"], "network_j": ["B", "A", "A"]})
    out = network_aggregate(np.array([1.0, -3.0, 2.0]), edges).set_index(["network_a", "network_b"])
    assert out.loc[("A", "B"), "n_edges"] == 2
    assert out.loc[("A", "B"), "mean_weight"] == -1.0
    assert out.loc[("A", "B"), "mean_abs_weight"] == 2.0
