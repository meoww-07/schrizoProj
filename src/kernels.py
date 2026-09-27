"""Fold-safe modality kernels (protocol §5.2).

Each modality is represented by a linear kernel on z-scored features, divided
by the mean diagonal of the training kernel. Feature means and SDs, confound
slopes and the kernel scale are estimated from training participants only and
applied unchanged to held-out participants.

With z-scored features the mean training diagonal equals the number of
non-constant features, so trace normalization removes the head start that a
4,950-edge block would otherwise have over a 150-feature block. It does not
equalize redundancy, noise or correlation structure (protocol §4).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler

from src.preprocessing import ConfoundRegressor


class ModalityKernel:
    """Standardize -> (optional confound regression) -> linear kernel -> trace normalization.

    Parameters
    ----------
    normalize : {"trace", "none"}
    residualize : bool
        Regress confounds out of every feature (slopes fit on training rows)
        before standardization. Used only by sensitivity models.
    """

    def __init__(self, normalize="trace", residualize=False):
        if normalize not in ("trace", "none"):
            raise ValueError(f"Unknown kernel normalization {normalize!r}")
        self.normalize = normalize
        self.residualize = residualize

    def fit(self, X, confounds=None):
        X = np.asarray(X, dtype=float)
        if self.residualize:
            if confounds is None:
                raise ValueError("residualize=True requires confounds")
            self.residualizer_ = ConfoundRegressor().fit(X, confounds)
            X = self.residualizer_.transform(X, confounds)
        self.scaler_ = StandardScaler().fit(X)
        self.Z_train_ = self.scaler_.transform(X)
        gram = self.Z_train_ @ self.Z_train_.T
        self.scale_ = float(np.mean(np.diag(gram))) if self.normalize == "trace" else 1.0
        if not self.scale_ > 0:
            raise ValueError("Training kernel has zero trace: every feature is constant in this fold.")
        self.K_train_ = gram / self.scale_
        return self

    def transform_features(self, X, confounds=None):
        """Standardized (and, if configured, residualized) features using training statistics."""
        X = np.asarray(X, dtype=float)
        if self.residualize:
            X = self.residualizer_.transform(X, confounds)
        return self.scaler_.transform(X)

    def cross_kernel(self, X, confounds=None):
        """Kernel between new rows and the training rows, shape (n_new, n_train)."""
        return self.transform_features(X, confounds) @ self.Z_train_.T / self.scale_


def combine_kernels(kernels, weights):
    """``sum_m weights[m] * kernels[m]`` over the modalities named in ``weights``."""
    total = None
    for name, weight in weights.items():
        if weight == 0:
            continue
        term = weight * kernels[name]
        total = term if total is None else total + term
    if total is None:
        raise ValueError("All kernel weights are zero.")
    return total


def center_kernel(K):
    """Double-center a square training kernel: H K H with H = I - 11'/n."""
    K = np.asarray(K, dtype=float)
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def alignment_weights(kernels, y):
    """Centered kernel alignment weights ("alignf"; Cortes, Mohri & Rostamizadeh, 2012).

    Solves min_{v >= 0} v'Mv - 2v'a with M_kl = <Kc_k, Kc_l>_F and
    a_k = <Kc_k, yy'>_F, then rescales v onto the simplex. Sensitivity MKL
    estimator only; the kernels passed in must be training kernels.
    """
    names = list(kernels)
    y_pm = np.where(np.asarray(y) == 1, 1.0, -1.0)
    target = np.outer(y_pm, y_pm)
    centered = [center_kernel(kernels[name]) for name in names]
    M = np.array([[np.sum(a * b) for b in centered] for a in centered])
    a = np.array([np.sum(Kc * target) for Kc in centered])
    L = cholesky(M + 1e-10 * np.trace(M) * np.eye(len(names)), lower=True)
    v, _ = nnls(L.T, solve_triangular(L, a, lower=True))
    if v.sum() <= 0:
        v = np.ones(len(names))
    return dict(zip(names, v / v.sum()))
