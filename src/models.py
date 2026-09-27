"""Classifiers compared in protocol §4.

Every kernel model is ``sklearn.svm.SVC(kernel="precomputed")`` on a convex
combination of trace-normalized linear modality kernels, so the unimodal,
early-fusion, equal-weight and MKL models differ only in how the kernel is
assembled. An SVC on a linear kernel is the L2-regularized hinge-loss linear
SVM specified in protocol §5.

Every model is fit with ``fit(X, y, inner_splits)``: hyperparameters, kernel
weights and the decision threshold are chosen from the outer training fold
``X, y`` and the inner splits supplied by ``validation.py``. Held-out
participants are only ever passed to ``decision_function``.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.kernels import ModalityKernel, alignment_weights, combine_kernels

TIE_TOL = 1e-12


def youden_threshold(y, scores):
    """Score threshold maximizing sensitivity + specificity - 1 (protocol §5.6).

    Participants with ``score >= threshold`` are classified as patients. Ties
    are broken toward the threshold closest to 0, the SVM default.
    """
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    j = tpr[finite] - fpr[finite]
    thresholds = thresholds[finite]
    tied = np.flatnonzero(j >= j.max() - TIE_TOL)
    return float(thresholds[tied[np.argmin(np.abs(thresholds[tied]))]])


class KernelSVM:
    """SVM on a fixed or learned convex combination of modality kernels.

    Parameters
    ----------
    name : str
    blocks : dict[str, array of int]
        Column indices of each modality in the design matrix.
    modalities : list[str]
        Modalities entering the kernel.
    weighting : {"fixed", "grid", "alignment", "concat"}
        fixed: ``weights`` given (default: equal). grid: MKL; the weight of the
        first modality is chosen jointly with C by inner CV from ``beta_grid``
        and the second gets 1 - beta. alignment: centered kernel alignment on
        the training kernels. concat: early fusion, one kernel over the
        concatenated columns.
    residualize_on : str or None
        Block of confounds regressed out of every feature within training folds.
    """

    WEIGHTINGS = ("fixed", "grid", "alignment", "concat")

    def __init__(self, name, blocks, modalities, weighting="fixed", weights=None,
                 C_grid=(1.0,), beta_grid=None, residualize_on=None,
                 normalize="trace", class_weight="balanced"):
        if weighting not in self.WEIGHTINGS:
            raise ValueError(f"Unknown weighting {weighting!r}")
        if weighting == "grid" and (len(modalities) != 2 or not beta_grid):
            raise ValueError("Grid MKL needs exactly two modalities and a beta_grid.")
        if weighting in ("concat", "alignment") and len(modalities) < 2:
            raise ValueError(f"{weighting!r} weighting needs at least two modalities.")
        if weighting == "fixed":
            weights = weights or {m: 1.0 / len(modalities) for m in modalities}
            if (set(weights) != set(modalities) or min(weights.values()) < 0
                    or not np.isclose(sum(weights.values()), 1.0)):
                raise ValueError("Fixed weights must cover the modalities, be nonnegative and sum to 1.")
        needed = list(modalities) + ([residualize_on] if residualize_on else [])
        self.name = name
        self.blocks = {m: np.asarray(blocks[m]) for m in needed}
        self.modalities = list(modalities)
        self.weighting = weighting
        self.weights = weights
        self.C_grid = [float(c) for c in C_grid]
        self.beta_grid = None if beta_grid is None else [float(b) for b in beta_grid]
        self.residualize_on = residualize_on
        self.normalize = normalize
        self.class_weight = class_weight

    # -- kernel construction -------------------------------------------------

    def _kernel_names(self):
        return ["concat"] if self.weighting == "concat" else self.modalities

    def _columns(self, name):
        if name == "concat":
            return np.concatenate([self.blocks[m] for m in self.modalities])
        return self.blocks[name]

    def _confounds(self, X):
        return X[:, self.blocks[self.residualize_on]] if self.residualize_on else None

    def _fit_kernels(self, X):
        confounds = self._confounds(X)
        return {
            name: ModalityKernel(self.normalize, residualize=confounds is not None)
            .fit(X[:, self._columns(name)], confounds)
            for name in self._kernel_names()
        }

    def _cross_kernels(self, kernels, X):
        confounds = self._confounds(X)
        return {name: k.cross_kernel(X[:, self._columns(name)], confounds)
                for name, k in kernels.items()}

    def _configs(self):
        betas = self.beta_grid if self.weighting == "grid" else [None]
        return [{"C": C, "beta": beta} for beta in betas for C in self.C_grid]

    def _weights(self, config, train_kernels, y):
        if self.weighting == "concat":
            return {"concat": 1.0}
        if self.weighting == "fixed":
            return dict(self.weights)
        if self.weighting == "grid":
            first, second = self.modalities
            return {first: config["beta"], second: 1.0 - config["beta"]}
        return alignment_weights({m: train_kernels[m] for m in self.modalities}, y)

    def _svm(self, C):
        return SVC(kernel="precomputed", C=C, class_weight=self.class_weight)

    # -- fitting -------------------------------------------------------------

    def fit(self, X, y, inner_splits):
        """Select (C, beta) and the threshold by inner CV, then refit on all of ``X``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        configs = self._configs()
        oof = np.full((len(configs), len(y)), np.nan)
        fold_auc = np.full((len(configs), len(inner_splits)), np.nan)
        for f, (train, val) in enumerate(inner_splits):
            kernels = self._fit_kernels(X[train])
            K_train = {name: k.K_train_ for name, k in kernels.items()}
            K_val = self._cross_kernels(kernels, X[val])
            for c, config in enumerate(configs):
                weights = self._weights(config, K_train, y[train])
                svm = self._svm(config["C"]).fit(combine_kernels(K_train, weights), y[train])
                scores = svm.decision_function(combine_kernels(K_val, weights))
                oof[c, val] = scores
                fold_auc[c, f] = roc_auc_score(y[val], scores)
        if np.isnan(oof).any():
            raise ValueError("Inner splits must place every training participant in exactly one validation fold.")

        mean_auc = fold_auc.mean(axis=1)
        tied = np.flatnonzero(mean_auc >= mean_auc.max() - TIE_TOL)
        best = min(tied, key=lambda c: (configs[c]["C"],
                                        0.0 if configs[c]["beta"] is None else abs(configs[c]["beta"] - 0.5)))
        self.config_ = configs[best]
        self.inner_auc_ = float(mean_auc[best])
        self.threshold_ = youden_threshold(y, oof[best])
        return self._refit(X, y, self.config_)

    def fit_config(self, X, y, config):
        """Fit with a fixed configuration (no tuning); used to build stacking features."""
        return self._refit(np.asarray(X, dtype=float), np.asarray(y), config)

    def _refit(self, X, y, config):
        self.kernels_ = self._fit_kernels(X)
        K_train = {name: k.K_train_ for name, k in self.kernels_.items()}
        self.weights_ = self._weights(config, K_train, y)
        self.svm_ = self._svm(config["C"]).fit(combine_kernels(K_train, self.weights_), y)
        return self

    def decision_function(self, X):
        K = self._cross_kernels(self.kernels_, np.asarray(X, dtype=float))
        return self.svm_.decision_function(combine_kernels(K, self.weights_))

    # -- reporting -----------------------------------------------------------

    def modality_weights(self):
        """Weight of each trace-normalized modality kernel in the fitted model.

        For early fusion the weights are implicit: each block's share of the
        non-constant standardized features (protocol §4).
        """
        if self.weighting != "concat":
            return {m: float(self.weights_.get(m, 0.0)) for m in self.modalities}
        active = self.kernels_["concat"].scaler_.var_ > 0
        counts, start = {}, 0
        for m in self.modalities:
            size = len(self.blocks[m])
            counts[m] = int(active[start:start + size].sum())
            start += size
        total = sum(counts.values())
        return {m: counts[m] / total for m in self.modalities}

    def selection(self):
        out = {"C": self.config_["C"], "inner_auc": self.inner_auc_, "threshold": self.threshold_}
        out.update({f"beta_{m}": w for m, w in self.modality_weights().items()})
        return out

    def primal_weights(self):
        """Weights on standardized features per modality (protocol §7.5).

        w_m = (beta_m / scale_m) * sum_i dual_i * z_m(x_i), so that
        decision = sum_m z_m(x) . w_m + intercept.
        """
        dual = self.svm_.dual_coef_.ravel()
        support = self.svm_.support_
        out = {}
        for name, kernel in self.kernels_.items():
            w = self.weights_.get(name, 0.0) / kernel.scale_ * (dual @ kernel.Z_train_[support])
            if name != "concat":
                out[name] = w
                continue
            start = 0
            for m in self.modalities:
                size = len(self.blocks[m])
                out[m] = w[start:start + size]
                start += size
        return out


class Stacking:
    """Late fusion: unimodal kernel-SVM scores combined by L2 logistic regression.

    Base models are tuned on the outer training fold. The meta-model is trained
    only on out-of-fold base scores from the inner splits, so it never sees a
    score produced by a base model fit on the same participant (protocol §4).
    """

    def __init__(self, name, base_models, meta_C=1.0):
        self.name = name
        self.base_models = list(base_models)
        self.meta_C = meta_C

    def _meta(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.meta_C))

    def fit(self, X, y, inner_splits):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for base in self.base_models:
            base.fit(X, y, inner_splits)

        oof = np.full((len(y), len(self.base_models)), np.nan)
        for train, val in inner_splits:
            for b, base in enumerate(self.base_models):
                fold_model = copy.copy(base).fit_config(X[train], y[train], base.config_)
                oof[val, b] = fold_model.decision_function(X[val])
        if np.isnan(oof).any():
            raise ValueError("Inner splits must place every training participant in exactly one validation fold.")
        self.oof_scores_ = oof

        meta_oof = np.full(len(y), np.nan)
        aucs = []
        for train, val in inner_splits:
            meta_oof[val] = self._meta().fit(oof[train], y[train]).decision_function(oof[val])
            aucs.append(roc_auc_score(y[val], meta_oof[val]))
        self.inner_auc_ = float(np.mean(aucs))
        self.threshold_ = youden_threshold(y, meta_oof)
        self.meta_ = self._meta().fit(oof, y)
        return self

    def decision_function(self, X):
        base_scores = np.column_stack([m.decision_function(X) for m in self.base_models])
        return self.meta_.decision_function(base_scores)

    def selection(self):
        out = {"inner_auc": self.inner_auc_, "threshold": self.threshold_}
        for base, coef in zip(self.base_models, self.meta_[-1].coef_.ravel()):
            out[f"C_{base.name}"] = base.config_["C"]
            out[f"meta_coef_{base.name}"] = float(coef)
        return out


def build_models(model_config, blocks, include=None):
    """Fresh, unfitted models for one outer fold (config order unless ``include`` is given)."""
    if model_config["kernel"].get("centering"):
        raise NotImplementedError("Kernel centering is not part of the protocol (§5.2).")
    specs = {spec["name"]: spec for spec in model_config["models"]}
    names = list(specs) if include is None else list(include)
    unknown = set(names) - set(specs)
    if unknown:
        raise KeyError(f"Models not in model_config: {sorted(unknown)}")

    def kernel_model(spec):
        weighting = spec.get("weighting", "fixed")
        return KernelSVM(
            name=spec["name"],
            blocks=blocks,
            modalities=spec["modalities"],
            weighting=weighting,
            weights=spec.get("weights"),
            C_grid=model_config["svm"]["C_grid"],
            beta_grid=model_config["mkl"]["beta_grid"] if weighting == "grid" else None,
            residualize_on=spec.get("residualize_on"),
            normalize=model_config["kernel"]["normalization"],
            class_weight=model_config["svm"]["class_weight"],
        )

    models = []
    for name in names:
        spec = specs[name]
        if spec["type"] == "stacking":
            base = [kernel_model(specs[b]) for b in spec["base_models"]]
            models.append(Stacking(name, base, meta_C=model_config["stacking"]["meta_C"]))
        else:
            models.append(kernel_model(spec))
    return models
