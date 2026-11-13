"""numcompute_stream builds on the group's numcompute package.

The streaming classes subclass the numcompute classes, and every streamed
result must equal what numcompute computes on all of the data at once.
"""

import numpy as np

import numcompute.metrics as nc_metrics
import numcompute.pipeline as nc_pipeline
import numcompute.preprocessing as nc_pre
import numcompute.stats as nc_stats
from _helpers import random_chunks
from numcompute import io as nc_io
from numcompute_stream import io as st_io
from numcompute_stream import metrics as st_metrics
from numcompute_stream.pipeline import ColumnTransformer, Pipeline
from numcompute_stream.preprocessing import (MinMaxScaler, OneHotEncoder, SimpleImputer,
                                             StandardScaler)
from numcompute_stream.stats import RunningStats, StreamingHistogram
from numcompute_stream.tree import DecisionTreeClassifier


def test_streaming_classes_subclass_the_group_classes():
    pairs = [(StandardScaler, nc_pre.StandardScaler), (MinMaxScaler, nc_pre.MinMaxScaler),
             (SimpleImputer, nc_pre.SimpleImputer), (OneHotEncoder, nc_pre.OneHotEncoder),
             (Pipeline, nc_pipeline.Pipeline), (ColumnTransformer, nc_pipeline.ColumnTransformer),
             (DecisionTreeClassifier, nc_pipeline.BaseEstimator)]
    for streaming, group in pairs:
        assert issubclass(streaming, group), streaming.__name__


def test_get_params_comes_from_the_group_base_class():
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=50)
    assert tree.get_params()["buffer_size"] == 50
    tree.set_params(max_depth=4)
    assert tree.max_depth == 4


def test_missing_tokens_are_the_group_tokens():
    assert set(st_io.DEFAULT_MISSING) == {t.lower() for t in nc_io.DEFAULT_MISSING}


def test_running_stats_equal_numcompute_batch(rng):
    for _ in range(10):
        X = rng.normal(5.0, 3.0, size=(int(rng.integers(5, 400)), 4))
        X[rng.random(X.shape) < 0.1] = np.nan
        rs = RunningStats()
        for sl in random_chunks(len(X), rng):
            rs.update(X[sl])
        np.testing.assert_allclose(rs.mean(), nc_stats.mean(X, axis=0, skipna=True), rtol=1e-12)
        np.testing.assert_allclose(rs.var(ddof=1), nc_stats.var(X, axis=0, ddof=1, skipna=True),
                                   rtol=1e-10)


def test_histogram_counts_equal_numcompute(rng):
    X = rng.random((500, 2))
    h = StreamingHistogram(bins=8, range=(0.0, 1.0))
    for sl in random_chunks(len(X), rng):
        h.update(X[sl])
    counts, edges = h.result()
    for j in range(2):
        nc_counts, nc_edges = nc_stats.histogram(X[:, j], bins=8, range=(0.0, 1.0))
        np.testing.assert_array_equal(counts[j], nc_counts)
        np.testing.assert_allclose(edges[j], nc_edges)


def test_streamed_scalers_equal_numcompute_fit(rng):
    X = rng.normal(10.0, 2.0, size=(250, 3))
    for streaming, group in ((StandardScaler(), nc_pre.StandardScaler()),
                             (MinMaxScaler(), nc_pre.MinMaxScaler())):
        for sl in random_chunks(len(X), rng):
            streaming.partial_fit(X[sl])
        np.testing.assert_allclose(streaming.transform(X), group.fit(X).transform(X), atol=1e-12)


def test_metrics_equal_numcompute(rng):
    y_true = rng.integers(0, 3, size=600)
    y_pred = np.where(rng.random(600) < 0.7, y_true, rng.integers(0, 3, size=600))
    cm = st_metrics.StreamingConfusionMatrix()
    acc = st_metrics.Accuracy()
    f1 = st_metrics.F1Score("macro")
    for sl in random_chunks(600, rng):
        for m in (cm, acc, f1):
            m.update(y_true[sl], y_pred[sl])
    np.testing.assert_array_equal(cm.result(), nc_metrics.confusion_matrix(y_true, y_pred))
    assert np.isclose(acc.result(), nc_metrics.accuracy(y_true, y_pred))
    assert np.isclose(f1.result(), nc_metrics.f1(y_true, y_pred, average="macro"))


def test_streaming_auc_equals_numcompute(rng):
    y = rng.integers(0, 2, size=300)
    scores = np.round(rng.random(300) + 0.3 * y, 2)      # rounding gives ties
    auc = st_metrics.StreamingAUC()
    for sl in random_chunks(300, rng):
        auc.update(y[sl], scores[sl])
    assert np.isclose(auc.result(), nc_metrics.roc_auc_score(y, scores))
