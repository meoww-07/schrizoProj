from functools import partial

import numpy as np
import pytest

from src.metrics import mean_auc_by_model
from src.models import build_models
from src.validation import (ROOT, inner_splits, load_yaml, outer_splits, permuted_dataset,
                            read_seeds, run_nested_cv, run_outer_fold, run_permutations)


def small_config():
    """The real model config with smaller grids so the tests run quickly."""
    cfg = load_yaml(ROOT / "configs" / "model_config.yaml")
    cfg["svm"]["C_grid"] = [0.1, 1.0]
    cfg["mkl"]["beta_grid"] = [0.0, 0.5, 1.0]
    return cfg


def test_every_participant_is_tested_once_per_repeat():
    y = np.array([0, 1] * 25)
    splits = outer_splits(y, 5, seeds=[1, 2])
    for repeat in (0, 1):
        tested = np.concatenate([s[4] for s in splits if s[0] == repeat])
        np.testing.assert_array_equal(np.sort(tested), np.arange(len(y)))


def test_outer_splits_are_deterministic_and_stratified():
    y = np.array([0] * 30 + [1] * 20)
    first, second = outer_splits(y, 5, [7]), outer_splits(y, 5, [7])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a[4], b[4])
        assert y[a[4]].sum() == 4


def test_inner_splits_partition_the_outer_training_fold():
    y_train = np.array([0, 1] * 20)
    inner = inner_splits(y_train, 5, seed=3, fold=2)
    validation = np.sort(np.concatenate([val for _, val in inner]))
    np.testing.assert_array_equal(validation, np.arange(len(y_train)))
    again = inner_splits(y_train, 5, seed=3, fold=2)
    for (_, a), (_, b) in zip(inner, again):
        np.testing.assert_array_equal(a, b)


def test_all_configured_models_run_on_identical_folds(synthetic):
    X, y, blocks = synthetic(n=48, signal="both")
    cfg = small_config()
    predictions, selections, weights = run_nested_cv(
        X, y, np.arange(len(y)), partial(build_models, cfg, blocks), seeds=[0],
        n_outer=3, n_inner=3, collect_weights=True)
    names = {m["name"] for m in cfg["models"]}
    assert set(predictions["model"]) == names
    assert len(predictions) == len(names) * len(y)
    folds = {name: frozenset(map(tuple, g[["repeat", "fold", "participant_id"]].to_numpy()))
             for name, g in predictions.groupby("model")}
    assert len(set(folds.values())) == 1
    assert set(selections["model"]) == names
    assert weights


def test_outer_predictions_ignore_other_test_participants(synthetic):
    X, y, blocks = synthetic(n=48, signal="both")
    split = outer_splits(y, 4, [0])[0]
    test = split[4]
    names = ["mkl", "stacking", "mkl_residualized"]
    ids = np.arange(len(y))
    base, _, _ = run_outer_fold(X, y, ids, split, build_models(small_config(), blocks, names), 3)

    X_changed, y_changed = X.copy(), y.copy()
    X_changed[test[1:]] = np.random.default_rng(1).normal(size=(len(test) - 1, X.shape[1])) * 10
    y_changed[test] = 1 - y_changed[test]
    changed, _, _ = run_outer_fold(X_changed, y_changed, ids, split,
                                   build_models(small_config(), blocks, names), 3)
    for name in names:
        before = base[(base["model"] == name) & (base["participant_id"] == test[0])]["score"].iloc[0]
        after = changed[(changed["model"] == name) & (changed["participant_id"] == test[0])]["score"].iloc[0]
        assert before == pytest.approx(after, abs=1e-10)


def test_block_permutation_moves_whole_rows_of_one_block(synthetic):
    X, y, blocks = synthetic()
    X_perm, y_perm = permuted_dataset(X, y, "block", index=3, seed=11, block=blocks["structural"])
    np.testing.assert_array_equal(y_perm, y)
    other = np.setdiff1d(np.arange(X.shape[1]), blocks["structural"])
    np.testing.assert_array_equal(X_perm[:, other], X[:, other])
    perm = np.random.default_rng([11, 3]).permutation(len(y))
    np.testing.assert_array_equal(X_perm[:, blocks["structural"]], X[perm][:, blocks["structural"]])


def test_label_permutation_is_reproducible_per_index(synthetic):
    X, y, _ = synthetic()
    _, a = permuted_dataset(X, y, "labels", 5, 11)
    _, b = permuted_dataset(X, y, "labels", 5, 11)
    _, c = permuted_dataset(X, y, "labels", 6, 11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.sum() == y.sum()


def test_run_permutations_returns_one_row_per_index(synthetic):
    X, y, blocks = synthetic(n=40)
    factory = partial(build_models, small_config(), blocks, include=["structural", "functional"])
    null = run_permutations(X, y, np.arange(len(y)), factory, [0], mean_auc_by_model, "labels",
                            indices=[0, 1], seed=5, n_outer=3, n_inner=3)
    assert list(null["permutation"]) == [0, 1]
    assert {"structural", "functional"} <= set(null.columns)


def test_seed_file_matches_configured_repeats():
    cv = load_yaml(ROOT / "configs" / "cv_config.yaml")
    seeds = read_seeds(ROOT / cv["seeds_file"])
    assert len(seeds) == cv["outer"]["n_repeats"]
    assert len(set(seeds)) == len(seeds)


def test_comparisons_reference_defined_models_and_grid_nests_baselines():
    cfg = load_yaml(ROOT / "configs" / "model_config.yaml")
    names = [m["name"] for m in cfg["models"]]
    assert len(names) == len(set(names))
    pairs = [(cfg["primary_comparison"]["model"], cfg["primary_comparison"]["comparator"]),
             *map(tuple, cfg["secondary_comparisons"])]
    for a, b in pairs:
        assert a in names and b in names
    grid = cfg["mkl"]["beta_grid"]
    assert {0.0, 0.5, 1.0} <= set(grid) and min(grid) >= 0 and max(grid) <= 1
