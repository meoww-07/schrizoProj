import numpy as np
import pytest

from src.models import KernelSVM, Stacking, youden_threshold
from src.validation import inner_splits

MODALITIES = ["structural", "functional"]


def test_youden_threshold_separates_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    threshold = youden_threshold(y, scores)
    np.testing.assert_array_equal((scores >= threshold).astype(int), y)


def test_youden_ties_break_toward_zero():
    # J = 0.5 at thresholds 2 and -1; -1 is closer to 0.
    assert youden_threshold(np.array([0, 1, 0, 1]), np.array([-2.0, -1.0, 1.0, 2.0])) == -1.0


@pytest.mark.parametrize("weighting, extra", [
    ("fixed", {}), ("concat", {}), ("grid", {"beta_grid": [0.3]}), ("alignment", {}),
])
def test_primal_weights_reproduce_decision_function(synthetic, weighting, extra):
    X, y, blocks = synthetic(signal="both")
    train, test = np.arange(40), np.arange(40, 60)
    model = KernelSVM("m", blocks, MODALITIES, weighting=weighting, C_grid=[1.0], **extra)
    model.fit(X[train], y[train], inner_splits(y[train], 3, seed=0, fold=0))
    w = model.primal_weights()
    if weighting == "concat":
        cols = np.concatenate([blocks["structural"], blocks["functional"]])
        Z = model.kernels_["concat"].transform_features(X[test][:, cols])
        p_s = len(blocks["structural"])
        Z = {"structural": Z[:, :p_s], "functional": Z[:, p_s:]}
    else:
        Z = {m: model.kernels_[m].transform_features(X[test][:, blocks[m]]) for m in MODALITIES}
    reconstructed = sum(Z[m] @ w[m] for m in MODALITIES) + model.svm_.intercept_[0]
    np.testing.assert_allclose(reconstructed, model.decision_function(X[test]), atol=1e-8)


def test_mkl_grid_endpoint_reproduces_unimodal_model(synthetic):
    X, y, blocks = synthetic()
    train, test = np.arange(40), np.arange(40, 60)
    inner = inner_splits(y[train], 3, seed=0, fold=0)
    mkl = KernelSVM("mkl", blocks, MODALITIES, weighting="grid", beta_grid=[1.0], C_grid=[0.1, 1.0])
    uni = KernelSVM("s", blocks, ["structural"], C_grid=[0.1, 1.0])
    mkl.fit(X[train], y[train], inner)
    uni.fit(X[train], y[train], inner)
    np.testing.assert_allclose(mkl.decision_function(X[test]), uni.decision_function(X[test]), atol=1e-10)


def test_early_fusion_reports_dimension_proportional_weights(synthetic):
    X, y, blocks = synthetic(p_s=10, p_f=60)
    model = KernelSVM("early", blocks, MODALITIES, weighting="concat", C_grid=[1.0])
    model.fit(X, y, inner_splits(y, 3, seed=0, fold=0))
    assert model.modality_weights()["structural"] == pytest.approx(10 / 70)


def test_mkl_prefers_the_informative_modality(synthetic):
    X, y, blocks = synthetic(n=80, signal="structural", effect=1.0)
    model = KernelSVM("mkl", blocks, MODALITIES, weighting="grid",
                      beta_grid=[0.0, 0.25, 0.5, 0.75, 1.0], C_grid=[0.01, 1.0])
    model.fit(X, y, inner_splits(y, 5, seed=0, fold=0))
    assert model.weights_["structural"] >= 0.5


def test_fixed_weights_must_lie_on_simplex(synthetic):
    _, _, blocks = synthetic()
    with pytest.raises(ValueError):
        KernelSVM("bad", blocks, MODALITIES, weights={"structural": 0.7, "functional": 0.7})


def test_stacking_meta_features_are_out_of_fold(synthetic):
    X, y, blocks = synthetic(signal="both")
    inner = inner_splits(y, 3, seed=0, fold=0)
    base = [KernelSVM("s", blocks, ["structural"], C_grid=[0.1, 1.0]),
            KernelSVM("f", blocks, ["functional"], C_grid=[0.1, 1.0])]
    model = Stacking("stack", base).fit(X, y, inner)
    train, val = inner[0]
    expected = (KernelSVM("s", blocks, ["structural"])
                .fit_config(X[train], y[train], base[0].config_)
                .decision_function(X[val]))
    np.testing.assert_allclose(model.oof_scores_[val, 0], expected)
    assert np.isfinite(model.decision_function(X)).all()
