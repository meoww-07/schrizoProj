"""Leakage-safe preprocessing helpers (protocol §3 and §5.4).

``ConfoundRegressor`` learns from data, so it is fit on training rows only and
applied unchanged to held-out rows. The other helpers act on one participant
at a time and cannot leak information across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConfoundRegressor:
    """Cross-validated confound regression (Snoek, Miletić & Scholte, 2019).

    Per-feature OLS slopes on the confounds are estimated from training rows;
    ``transform`` removes the confound-explained part from any rows using those
    slopes and keeps each feature's training mean.
    """

    def fit(self, X, confounds):
        X = np.asarray(X, dtype=float)
        C = _as_2d(confounds)
        self.confound_mean_ = C.mean(axis=0)
        self.coef_, *_ = np.linalg.lstsq(
            C - self.confound_mean_, X - X.mean(axis=0), rcond=None
        )
        return self

    def transform(self, X, confounds):
        C = _as_2d(confounds)
        return np.asarray(X, dtype=float) - (C - self.confound_mean_) @ self.coef_


def _as_2d(values):
    values = np.asarray(values, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def fisher_z(r, eps=1e-7):
    """Fisher r-to-z; |r| is clipped below 1 so perfect correlations stay finite."""
    return np.arctanh(np.clip(r, -1.0 + eps, 1.0 - eps))


def upper_triangle(matrix):
    """Row-major upper triangle of a square matrix, diagonal excluded."""
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def edge_table(roi_names, networks=None):
    """One row per edge, in the order produced by ``upper_triangle``."""
    rows, cols = np.triu_indices(len(roi_names), k=1)
    names = np.asarray(roi_names)
    table = pd.DataFrame({
        "feature": [f"fc_{i:03d}_{j:03d}" for i, j in zip(rows, cols)],
        "roi_i": rows,
        "roi_j": cols,
        "name_i": names[rows],
        "name_j": names[cols],
    })
    if networks is not None:
        networks = np.asarray(networks)
        table["network_i"] = networks[rows]
        table["network_j"] = networks[cols]
    return table


def normalize_by_etiv(features, columns, etiv):
    """Proportional head-size adjustment, one participant at a time."""
    out = features.copy()
    out[columns] = out[columns].div(etiv, axis=0)
    return out
