import numpy as np
import pytest

from src.kernels import ModalityKernel, alignment_weights, center_kernel, combine_kernels


def test_trace_normalization_gives_unit_mean_diagonal():
    X = np.random.default_rng(0).normal(size=(30, 50)) * 5 + 3
    kernel = ModalityKernel().fit(X)
    assert np.mean(np.diag(kernel.K_train_)) == pytest.approx(1.0)
    assert kernel.scale_ == pytest.approx(50)


def test_constant_features_do_not_count_toward_scale():
    X = np.random.default_rng(0).normal(size=(30, 50))
    X[:, :5] = 2.0
    assert ModalityKernel().fit(X).scale_ == pytest.approx(45)


def test_kernel_statistics_come_from_training_rows_only():
    X = np.random.default_rng(1).normal(size=(40, 20))
    train, test = X[:30], X[30:]
    kernel = ModalityKernel().fit(train)
    np.testing.assert_allclose(kernel.scaler_.mean_, train.mean(axis=0))
    cross = kernel.cross_kernel(test)
    assert cross.shape == (10, 30)
    # A held-out row's kernel values depend only on that row and the training data.
    np.testing.assert_allclose(kernel.cross_kernel(test[:1]), cross[:1])


def test_concatenated_kernel_is_dimension_weighted_sum_of_modality_kernels():
    rng = np.random.default_rng(2)
    Xs, Xf = rng.normal(size=(25, 8)), rng.normal(size=(25, 40))
    ks, kf = ModalityKernel().fit(Xs), ModalityKernel().fit(Xf)
    kc = ModalityKernel().fit(np.hstack([Xs, Xf]))
    expected = (8 * ks.K_train_ + 40 * kf.K_train_) / 48
    np.testing.assert_allclose(kc.K_train_, expected)


def test_center_kernel_has_zero_row_and_column_means():
    A = np.random.default_rng(3).normal(size=(10, 4))
    Kc = center_kernel(A @ A.T)
    np.testing.assert_allclose(Kc.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Kc.mean(axis=1), 0, atol=1e-12)


def test_alignment_weights_lie_on_simplex_and_favor_informative_kernel():
    rng = np.random.default_rng(4)
    y = np.array([0, 1] * 20)
    informative = rng.normal(size=(40, 5)) + 2 * y[:, None]
    noise = rng.normal(size=(40, 50))
    kernels = {"a": ModalityKernel().fit(informative).K_train_, "b": ModalityKernel().fit(noise).K_train_}
    weights = alignment_weights(kernels, y)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0
    assert weights["a"] > weights["b"]


def test_combine_kernels_skips_zero_weights_and_rejects_all_zero():
    A, B = np.eye(3), np.ones((3, 3))
    np.testing.assert_array_equal(combine_kernels({"a": A, "b": B}, {"a": 1.0, "b": 0.0}), A)
    with pytest.raises(ValueError):
        combine_kernels({"a": A}, {"a": 0.0})


def test_residualized_kernel_requires_confounds():
    X = np.random.default_rng(5).normal(size=(10, 3))
    with pytest.raises(ValueError):
        ModalityKernel(residualize=True).fit(X)
