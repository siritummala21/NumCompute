import numpy as np
import pytest

from _helpers import random_chunks
from numcompute_stream.stats import (RunningStats, StreamingHistogram, StreamingQuantile,
                                     StreamStats, update_stats, welford_loop)


def test_running_stats_equal_numpy_for_any_chunking(rng):
    X = rng.normal(3.0, 2.0, size=(500, 3))
    for _ in range(5):
        rs = RunningStats()
        for sl in random_chunks(len(X), rng):
            rs.update(X[sl])
        np.testing.assert_allclose(rs.mean(), X.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(rs.var(), X.var(axis=0), rtol=1e-10)
        np.testing.assert_allclose(rs.std(ddof=1), X.std(axis=0, ddof=1), rtol=1e-10)
        np.testing.assert_array_equal(rs.min_, X.min(axis=0))
        np.testing.assert_array_equal(rs.max_, X.max(axis=0))


def test_large_offset_does_not_lose_the_variance(rng):
    # sum(x^2) - n*mean^2 would cancel here; the merge formula does not
    X = 1e9 + rng.normal(0, 1, size=(2000, 1))
    rs = RunningStats()
    for sl in random_chunks(len(X), rng):
        rs.update(X[sl])
    np.testing.assert_allclose(rs.var(), X.var(axis=0), rtol=1e-6)


def test_nan_values_are_skipped_and_counted(rng):
    X = rng.normal(size=(200, 2))
    X[rng.random(X.shape) < 0.2] = np.nan
    rs = RunningStats().update(X[:50]).update(X[50:])
    np.testing.assert_allclose(rs.mean(), np.nanmean(X, axis=0), rtol=1e-12)
    np.testing.assert_allclose(rs.var(), np.nanvar(X, axis=0), rtol=1e-10)
    np.testing.assert_array_equal(rs.result()["n_missing"], np.isnan(X).sum(axis=0))


def test_all_nan_column_gives_nan_not_an_error():
    rs = RunningStats().update(np.array([[1.0, np.nan], [2.0, np.nan]]))
    assert rs.mean()[0] == 1.5 and np.isnan(rs.mean()[1])
    assert np.isnan(rs.var()[1])


def test_window_only_uses_the_last_chunks(rng):
    chunks = [rng.normal(i, 1, size=(30, 2)) for i in range(6)]
    rs = RunningStats(window=2)
    for c in chunks:
        rs.update(c)
    last_two = np.vstack(chunks[-2:])
    np.testing.assert_allclose(rs.mean(), last_two.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(rs.var(), last_two.var(axis=0), rtol=1e-10)


def test_empty_chunk_is_ignored_and_bad_input_raises():
    rs = RunningStats().update(np.empty((0, 2))).update([[1.0, 2.0]])
    assert rs.count_.tolist() == [1, 1]
    with pytest.raises(ValueError):
        rs.update([[1.0, 2.0, 3.0]])           # wrong number of features
    with pytest.raises(ValueError):
        rs.update([1.0, 2.0])                  # 1-D is ambiguous
    with pytest.raises(ValueError):
        rs.update([[np.inf, 1.0]])
    with pytest.raises(RuntimeError):
        RunningStats().mean()


def test_welford_loop_matches_vectorised_version(rng):
    X = rng.normal(size=(300, 2))
    X[5, 1] = np.nan
    mean, var = welford_loop(X)
    rs = RunningStats().update(X)
    np.testing.assert_allclose(mean, rs.mean(), rtol=1e-12)
    np.testing.assert_allclose(var, rs.var(), rtol=1e-10)


def test_quantile_is_exact_when_everything_is_kept(rng):
    X = rng.normal(size=(400, 2))
    q = StreamingQuantile(capacity=None)
    for sl in random_chunks(len(X), rng):
        q.update(X[sl])
    np.testing.assert_allclose(q.result([0.1, 0.5, 0.9]),
                               np.quantile(X, [0.1, 0.5, 0.9], axis=0))


def test_reservoir_has_fixed_size_and_is_close(rng):
    X = rng.uniform(0, 1, size=(20000, 1))
    q = StreamingQuantile(capacity=1000, random_state=0)
    for sl in random_chunks(len(X), rng, max_size=500):
        q.update(X[sl])
    assert q._sample.shape == (1000, 1)
    assert q.n_seen_ == 20000
    assert abs(q.result(0.5)[0] - 0.5) < 0.05


def test_quantile_rejects_bad_q():
    q = StreamingQuantile().update([[1.0], [2.0]])
    with pytest.raises(ValueError):
        q.result(1.5)


def test_histogram_counts_match_numpy_with_fixed_range(rng):
    X = rng.random((600, 2))
    h = StreamingHistogram(bins=5, range=(0.0, 1.0))
    for sl in random_chunks(len(X), rng):
        h.update(X[sl])
    counts, edges = h.result()
    for j in range(2):
        np.testing.assert_array_equal(counts[j], np.histogram(X[:, j], 5, (0.0, 1.0))[0])
    assert edges.shape == (2, 6)


def test_histogram_keeps_out_of_range_values_in_end_bins():
    h = StreamingHistogram(bins=2).update([[0.0], [1.0]])     # range taken from chunk 1
    h.update([[-10.0], [10.0], [np.nan]])
    counts, _ = h.result()
    assert counts.sum() == 4                                  # NaN skipped, nothing lost
    assert counts[0].tolist() == [2, 2]


def test_histogram_constant_first_chunk_and_window():
    h = StreamingHistogram(bins=4, window=1).update([[5.0], [5.0]])
    counts, edges = h.result()
    assert edges[0, 0] < 5.0 < edges[0, -1] and counts.sum() == 2
    h.update([[5.0]])
    assert h.result()[0].sum() == 1                           # only the last chunk


def test_update_stats_collects_everything(rng):
    X = rng.normal(size=(300, 2))
    state = None
    for sl in random_chunks(len(X), rng):
        state = update_stats(X[sl], state)
    out = state.result()
    np.testing.assert_allclose(out["mean"], X.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(out["median"], np.median(X, axis=0))   # capacity 1000 > 300
    assert out["hist_counts"].sum() == X.size
    assert isinstance(state, StreamStats)
