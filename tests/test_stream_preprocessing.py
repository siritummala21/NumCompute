import warnings

import numpy as np
import pytest

from _helpers import random_chunks
from numcompute_stream import NotFittedError
from numcompute_stream.preprocessing import (MinMaxScaler, OneHotEncoder, SimpleImputer,
                                             StandardScaler)


def test_streamed_scaler_equals_one_big_fit(rng):
    X = rng.normal(10, 3, size=(300, 3))
    streamed = StandardScaler()
    for sl in random_chunks(len(X), rng):
        streamed.partial_fit(X[sl])
    Z = streamed.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(streamed.inverse_transform(Z), X, rtol=1e-12)


def test_tiny_but_not_constant_column_is_scaled(rng):
    # 2.1 feedback: this column was treated as constant by a std threshold
    X = np.array([[1e-16], [2e-16], [3e-16]])
    Z = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:]).transform(X)
    np.testing.assert_allclose(Z.ravel(), [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], rtol=1e-10)


def test_constant_column_becomes_zero():
    X = np.array([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]])
    sc = StandardScaler().partial_fit(X)
    assert sc.scale_[0] == 1.0
    np.testing.assert_array_equal(sc.transform(X)[:, 0], 0.0)


def test_fit_resets_but_partial_fit_accumulates(rng):
    sc = StandardScaler().partial_fit([[0.0], [2.0]])
    sc.partial_fit([[4.0]])
    assert sc.mean_[0] == 2.0
    sc.fit([[10.0], [10.0]])
    assert sc.mean_[0] == 10.0


def test_scaler_errors():
    with pytest.raises(NotFittedError):
        StandardScaler().transform([[1.0]])
    sc = StandardScaler().partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        sc.transform([[1.0]])
    assert sc.transform(np.empty((0, 2))).shape == (0, 2)


def test_minmax_running_range_and_clip():
    mm = MinMaxScaler().partial_fit([[0.0], [5.0]]).partial_fit([[10.0]])
    np.testing.assert_allclose(mm.transform([[0.0], [5.0], [10.0]]).ravel(), [0, 0.5, 1])
    assert mm.transform([[20.0]])[0, 0] == 2.0
    clipped = MinMaxScaler(feature_range=(-1, 1), clip=True).partial_fit([[0.0], [10.0]])
    assert clipped.transform([[20.0]])[0, 0] == 1.0
    np.testing.assert_allclose(mm.inverse_transform(mm.transform([[3.0]])), [[3.0]])


def test_minmax_constant_column_and_bad_range():
    mm = MinMaxScaler().partial_fit([[4.0], [4.0]])
    assert mm.transform([[4.0]])[0, 0] == 0.0
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


@pytest.mark.parametrize("strategy, expected", [("mean", 3.0), ("median", 2.0),
                                                ("most_frequent", 2.0), ("constant", -1.0)])
def test_imputer_strategies(strategy, expected):
    X = np.array([[2.0], [2.0], [np.nan], [5.0]])
    imp = SimpleImputer(strategy=strategy, fill_value=-1.0)
    imp.partial_fit(X[:2]).partial_fit(X[2:])
    out = imp.transform(X)
    assert out[2, 0] == pytest.approx(expected)
    assert not np.isnan(out).any()


def test_imputer_categorical_most_frequent_and_ties():
    X = np.array([["a"], ["b"], ["b"], [None]], dtype=object)
    imp = SimpleImputer(strategy="most_frequent").partial_fit(X)
    assert imp.transform(X)[3, 0] == "b"
    tie = SimpleImputer(strategy="most_frequent").partial_fit([[2.0], [1.0]])
    assert tie.statistics_[0] == 1.0          # tie goes to the smaller value


def test_imputer_mean_on_strings_is_rejected():
    with pytest.raises(TypeError):
        SimpleImputer(strategy="mean").partial_fit(np.array([["a"]], dtype=object))
    with pytest.raises(ValueError):
        SimpleImputer(strategy="mode")


def test_imputer_column_with_no_values_warns():
    imp = SimpleImputer(fill_value=0.0).partial_fit([[np.nan, 1.0]])
    with pytest.warns(RuntimeWarning):
        out = imp.transform([[np.nan, np.nan]])
    np.testing.assert_array_equal(out, [[0.0, 1.0]])


def test_onehot_new_categories_are_appended_last():
    enc = OneHotEncoder().partial_fit(np.array([["b"], ["a"]], dtype=object))
    first = enc.transform(np.array([["a"]], dtype=object))
    enc.partial_fit(np.array([["c"]], dtype=object))
    later = enc.transform(np.array([["a"], ["c"]], dtype=object))
    assert enc.get_feature_names_out(["site"]) == ["site=b", "site=a", "site=c"]
    np.testing.assert_array_equal(later[0, :2], first[0])     # old columns did not move
    np.testing.assert_array_equal(later[1], [0, 0, 1])


def test_onehot_unknown_missing_and_inverse():
    enc = OneHotEncoder().partial_fit(np.array([["x", "p"], ["y", "q"]], dtype=object))
    out = enc.transform(np.array([["z", None]], dtype=object))
    np.testing.assert_array_equal(out, 0)                      # unknown and missing -> zeros
    X = np.array([["y", "p"]], dtype=object)
    np.testing.assert_array_equal(enc.inverse_transform(enc.transform(X)), X)
    strict = OneHotEncoder(handle_unknown="error").partial_fit(np.array([["x"]], dtype=object))
    with pytest.raises(ValueError):
        strict.transform(np.array([["z"]], dtype=object))


def test_huge_values_are_scaled_correctly():
    # found by probing like the 2.1 marker: the squares overflowed near 1e308
    X = np.array([[-1e308], [0.0], [1e308]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Z = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:]).transform(X)
        mm = MinMaxScaler().partial_fit(X)
        M = mm.transform(X)
    np.testing.assert_allclose(Z.ravel(), [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], rtol=1e-12)
    np.testing.assert_array_equal(M.ravel(), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(mm.inverse_transform(M), X)
