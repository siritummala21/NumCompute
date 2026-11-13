import warnings

import numpy as np
import pytest

from _helpers import make_blobs
from numcompute_stream import NotFittedError
from numcompute_stream.tree import (DecisionTree, DecisionTreeClassifier, best_split_loop,
                                    best_split_vectorised, entropy, gini)


def test_gini_and_entropy_values():
    assert gini([5, 5]) == pytest.approx(0.5)
    assert gini([10, 0]) == 0.0
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([4, 0]) == 0.0
    assert gini([0, 0]) == 0.0                           # empty node is not NaN


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("min_leaf", [1, 3])
def test_vectorised_split_matches_loop(rng, criterion, min_leaf):
    # small integer features give many ties, NaN rows test the missing-value rule
    for _ in range(40):
        n, d = int(rng.integers(2, 50)), int(rng.integers(1, 4))
        X = rng.integers(0, 5, size=(n, d)).astype(float)
        X[rng.random(X.shape) < 0.1] = np.nan
        y = rng.integers(0, 3, size=n)
        a = best_split_vectorised(X, y, 3, np.arange(d), criterion, min_leaf)
        b = best_split_loop(X, y, 3, np.arange(d), criterion, min_leaf)
        assert (a is None) == (b is None)
        if a is not None:
            assert (a["feature"], a["threshold"], a["nan_left"]) == \
                   (b["feature"], b["threshold"], b["nan_left"])
            assert a["gain"] == pytest.approx(b["gain"])


def test_split_never_separates_equal_values():
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
    split = best_split_vectorised(X, np.array([0, 1, 0, 1]), 2, [0])
    assert split is None                                 # no threshold helps here
    split = best_split_vectorised(X, np.array([0, 0, 1, 1]), 2, [0])
    assert 1.0 <= split["threshold"] < 2.0


def test_tree_learns_easy_problem(rng):
    X, y = make_blobs(600, rng)
    tree = DecisionTreeClassifier(max_depth=5).fit(X[:400], y[:400])
    assert tree.score(X[400:], y[400:]) > 0.9
    assert tree.get_depth() <= 5
    proba = tree.predict_proba(X[400:])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_buffer_retrains_only_when_full(rng):
    X, y = make_blobs(250, rng)
    tree = DecisionTree(max_depth=4, buffer_size=100)
    tree.partial_fit(X[:60], y[:60])
    assert not tree.is_fitted and tree.buffer_count == 60
    with pytest.raises(NotFittedError):
        tree.predict(X[:5])
    tree.partial_fit(X[60:120], y[60:120])             # 120 >= 100: retrain
    assert tree.last_partial_fit_retrained_ and tree.buffer_count == 0
    assert tree.n_retrains_ == 1 and tree.retrain_history_[0]["n_samples"] == 120
    tree.partial_fit(X[120:150], y[120:150])
    assert not tree.last_partial_fit_retrained_ and tree.n_retrains_ == 1


def test_streamed_tree_equals_batch_tree_on_the_same_buffer(rng):
    X, y = make_blobs(300, rng)
    streamed = DecisionTreeClassifier(max_depth=4, buffer_size=300)
    for start in range(0, 300, 50):
        streamed.partial_fit(X[start:start + 50], y[start:start + 50])
    batch = DecisionTreeClassifier(max_depth=4).fit(X, y)
    np.testing.assert_array_equal(streamed.predict(X), batch.predict(X))


def test_single_class_and_constant_features_give_one_leaf():
    tree = DecisionTreeClassifier().fit(np.ones((10, 2)), np.r_[np.zeros(5), np.ones(5)])
    assert tree.get_n_leaves() == 1
    tree = DecisionTreeClassifier().fit(np.random.default_rng(0).random((10, 2)), np.zeros(10))
    assert tree.get_n_leaves() == 1 and set(tree.predict([[0.5, 0.5]])) == {0}


def test_nan_features_follow_the_larger_child():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [np.nan]])
    tree = DecisionTreeClassifier().fit(X, [0, 0, 0, 1, 0])
    assert tree.predict([[np.nan]])[0] == 0


def test_new_class_after_retrain_has_zero_probability(rng):
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(rng.random((10, 2)), np.array([0, 1] * 5))
    tree.partial_fit(rng.random((3, 2)), np.array([2, 2, 2]))   # buffered, no retrain
    assert list(tree.classes_) == [0, 1, 2]
    assert tree.predict_proba(rng.random((4, 2)))[:, 2].tolist() == [0, 0, 0, 0]


def test_wider_chunk_pads_older_rows(rng):
    # when the one-hot encoder meets a new category the chunk gets a new column
    tree = DecisionTreeClassifier(buffer_size=20)
    tree.partial_fit(rng.random((10, 2)), np.zeros(10, dtype=int))
    tree.partial_fit(rng.random((10, 3)), np.ones(10, dtype=int))
    assert tree.n_features_in_ == 3
    with pytest.raises(ValueError):
        tree.partial_fit(rng.random((5, 2)), np.zeros(5, dtype=int))


def test_flush_keeps_only_last_rows(rng):
    X, y = make_blobs(80, rng)
    tree = DecisionTreeClassifier(buffer_size=1000)
    tree.partial_fit(X, y)
    tree.flush(min_samples=10, last_n=30)
    assert tree.n_retrains_ == 1 and tree.retrain_history_[-1]["n_samples"] == 30
    tree.flush(min_samples=10)                          # buffer now empty: nothing happens
    assert tree.n_retrains_ == 1


def test_max_features_and_random_state(rng):
    X, y = make_blobs(300, rng, n_features=9)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert DecisionTreeClassifier(max_features=0.5).fit(X, y).score(X, y) > 0.8


def test_loop_predict_and_inspection(rng):
    X, y = make_blobs(200, rng)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    np.testing.assert_array_equal(tree.predict_loop(X), tree.predict(X))
    assert tree.feature_importances_.sum() == pytest.approx(1.0)
    assert "class:" in tree.export_text(["a", "b", "c", "d"])
    assert tree.memory_bytes() > 0
    assert tree.predict(np.empty((0, 4))).shape == (0,)


@pytest.mark.parametrize("kwargs", [{"max_depth": -1}, {"buffer_size": 0},
                                    {"min_samples_split": 1}, {"criterion": "mse"},
                                    {"splitter": "fast"}, {"max_features": "half"},
                                    {"max_features": 1.5}, {"max_depth": 2.5},
                                    {"buffer_size": True}])
def test_bad_parameters_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DecisionTreeClassifier(**kwargs)


def test_bad_data_is_rejected():
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        tree.fit([[1.0], [2.0]], [0])                   # lengths differ
    with pytest.raises(ValueError):
        tree.fit([[1.0], [np.inf]], [0, 1])
    with pytest.raises(ValueError):
        tree.fit([[1.0], [2.0]], [0.0, np.nan])
    tree.fit([[1.0, 2.0], [2.0, 1.0]], [0, 1])
    with pytest.raises(ValueError):
        tree.predict([[1.0]])                           # too few features
    with pytest.raises(TypeError):
        tree.partial_fit([[1.0, 2.0]], ["a"])          # label type changed


def test_split_between_huge_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # hi - lo used to overflow here
        tree = DecisionTreeClassifier().fit([[-1e308], [1e308]], [0, 1])
    np.testing.assert_array_equal(tree.predict([[-1e308], [1e308]]), [0, 1])
