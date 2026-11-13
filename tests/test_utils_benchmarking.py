import numpy as np
import pytest

from numcompute import benchmarking
from numcompute.utils import (batch_iterator, check_array, log_softmax, logsumexp,
                              pairwise_distances, relu, sigmoid, softmax, topk_mask)


def test_logsumexp_stability_and_edge_cases():
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000 + np.log(2))
    assert logsumexp([-1000.0, -1000.0]) == pytest.approx(-1000 + np.log(2))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([1.0, np.inf]) == np.inf
    assert np.isnan(logsumexp([1.0, np.nan]))
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(logsumexp(X, axis=1), np.log(np.exp(X).sum(1)))
    assert logsumexp(X, axis=0, keepdims=True).shape == (1, 2)


def test_softmax_overflow_and_sum():
    p = softmax([1000.0, 0.0, -1000.0])
    assert np.all(np.isfinite(p)) and p[0] == pytest.approx(1.0)
    P = softmax(np.random.default_rng(0).normal(size=(5, 4)) * 100, axis=1)
    np.testing.assert_allclose(P.sum(1), 1)
    np.testing.assert_allclose(np.exp(log_softmax([1.0, 2.0, 3.0])), softmax([1.0, 2.0, 3.0]))


def test_sigmoid_relu():
    s = sigmoid(np.array([-800.0, 0.0, 800.0]))
    np.testing.assert_allclose(s, [0, 0.5, 1])
    assert relu([-1.0, 2.0]).tolist() == [0, 2]


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "manhattan", "cosine"])
def test_pairwise_distances_against_loop(rng, metric):
    A, B = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    D = pairwise_distances(A, B, metric)
    ref = np.empty((6, 4))
    for i in range(6):
        for j in range(4):
            d = A[i] - B[j]
            ref[i, j] = {"euclidean": np.sqrt(d @ d), "sqeuclidean": d @ d,
                         "manhattan": np.abs(d).sum(),
                         "cosine": 1 - A[i] @ B[j] / np.linalg.norm(A[i]) / np.linalg.norm(B[j])}[metric]
    np.testing.assert_allclose(D, ref, atol=1e-10)


def test_pairwise_self_distance_zero_and_errors():
    A = np.array([[1e8, 1e8], [1e8, 1e8 + 1]])
    D = pairwise_distances(A)
    # exact expected values, not just finiteness (Assignment 2.1 feedback)
    np.testing.assert_array_equal(D, [[0.0, 1.0], [1.0, 0.0]])
    assert pairwise_distances([[1e9]], [[1e9 + 1]])[0, 0] == 1.0
    with pytest.raises(ValueError):
        pairwise_distances(np.ones((2, 3)), np.ones((2, 2)))


def test_check_array_messages():
    with pytest.raises(ValueError, match="reshape"):
        check_array([1, 2, 3])
    with pytest.raises(ValueError, match="NaN"):
        check_array([[np.nan]], allow_nan=False)


def test_topk_mask_and_batches():
    m = topk_mask(np.array([[3, 1, 2], [0, 5, 4]]), 2)
    assert m.sum(1).tolist() == [2, 2] and m[0].tolist() == [True, False, True]
    X, y = np.arange(10).reshape(5, 2), np.arange(5)
    batches = list(batch_iterator(X, y, batch_size=2))
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]
    assert len(list(batch_iterator(X, batch_size=2, drop_last=True))) == 2
    shuffled = np.concatenate([b for b in batch_iterator(y, batch_size=2, shuffle=True, seed=1)])
    assert sorted(shuffled.tolist()) == y.tolist()


def test_benchmark_harness():
    rows = benchmarking.compare({"loop": lambda x: sum(x), "np": np.sum}, args=([1.0, 2.0, 3.0],),
                                repeat=2, task="sum")
    assert [r["name"] for r in rows] == ["loop", "np"] and rows[0]["speedup"] == 1.0
    assert "| sum |" in benchmarking.format_table(rows)
    assert "numpy" in benchmarking.environment_info()
    with pytest.raises(AssertionError):
        benchmarking.compare({"a": lambda: 1.0, "b": lambda: 2.0}, repeat=1)
