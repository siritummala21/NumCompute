import numpy as np
import pytest

from _helpers import random_chunks
from numcompute_stream.metrics import (Accuracy, F1Score, Precision, Recall, RollingMetric,
                                       StreamingAUC, StreamingConfusionMatrix,
                                       confusion_matrix_loop)


@pytest.fixture
def labels(rng):
    y_true = rng.integers(0, 3, size=500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 3, size=500))
    return y_true, y_pred


def stream(metric, y_true, y_pred, rng):
    for sl in random_chunks(len(y_true), rng):
        metric.update(y_true[sl], y_pred[sl])
    return metric.result()


def test_accuracy_streamed_equals_batch(labels, rng):
    y_true, y_pred = labels
    assert stream(Accuracy(), y_true, y_pred, rng) == pytest.approx(np.mean(y_true == y_pred))


def test_confusion_matrix_grows_with_new_labels():
    cm = StreamingConfusionMatrix().update([0, 1], [0, 1])
    cm.update([2], [0])                                 # label 2 appears later
    np.testing.assert_array_equal(cm.labels_, [0, 1, 2])
    np.testing.assert_array_equal(cm.result(), [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert cm.n_samples_ == 3


def test_confusion_matrix_label_type_change_is_rejected():
    cm = StreamingConfusionMatrix().update([0, 1], [0, 1])
    with pytest.raises(TypeError):
        cm.update(["a"], ["a"])


def test_loop_confusion_matrix_matches(labels, rng):
    y_true, y_pred = labels
    cm = stream(StreamingConfusionMatrix(), y_true, y_pred, rng)
    np.testing.assert_array_equal(cm, confusion_matrix_loop(y_true, y_pred, [0, 1, 2]))


def manual_scores(y_true, y_pred, c):
    tp = np.sum((y_true == c) & (y_pred == c))
    p = tp / max(np.sum(y_pred == c), 1)
    r = tp / max(np.sum(y_true == c), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def test_macro_and_weighted_averages(labels, rng):
    y_true, y_pred = labels
    per_class = np.array([manual_scores(y_true, y_pred, c) for c in range(3)])
    support = np.bincount(y_true)
    for k, cls in enumerate((Precision, Recall, F1Score)):
        assert stream(cls("macro"), y_true, y_pred, rng) == pytest.approx(per_class[:, k].mean())
        weighted = np.sum(per_class[:, k] * support) / support.sum()
        assert stream(cls("weighted"), y_true, y_pred, rng) == pytest.approx(weighted)


def test_binary_and_micro():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    assert Precision("binary").update(y_true, y_pred).result() == pytest.approx(2 / 3)
    assert Recall("binary").update(y_true, y_pred).result() == pytest.approx(2 / 3)
    # micro-averaged F1 equals accuracy for single-label problems
    assert F1Score("micro").update(y_true, y_pred).result() == pytest.approx(0.6)


def test_class_never_predicted_gives_zero_not_nan():
    p = Precision("macro").update([0, 1], [0, 0]).result()
    assert p == pytest.approx(0.25)
    assert np.isnan(Accuracy().result())                 # nothing seen yet


def test_auc_perfect_random_and_ties():
    y = np.array([0, 0, 1, 1])
    assert StreamingAUC().update(y, [0.1, 0.2, 0.8, 0.9]).result() == 1.0
    assert StreamingAUC().update(y, [0.5, 0.5, 0.5, 0.5]).result() == 0.5
    one_class = StreamingAUC().update([1, 1], [0.3, 0.4])
    assert np.isnan(one_class.result())


def test_auc_multiclass_one_vs_rest(rng):
    y = rng.integers(0, 3, size=200)
    scores = np.eye(3)[y] * 0.6 + rng.random((200, 3)) * 0.5
    auc = StreamingAUC(classes=[0, 1, 2])
    for sl in random_chunks(200, rng):
        auc.update(y[sl], scores[sl])
    assert 0.5 < auc.result() <= 1.0


def test_auc_rejects_nan_scores():
    with pytest.raises(ValueError):
        StreamingAUC().update([0, 1], [0.1, np.nan])


def test_rolling_metric_forgets_old_samples():
    roll = RollingMetric(Accuracy(), window=4)
    roll.update([0, 0, 0, 0], [1, 1, 1, 1])              # all wrong
    roll.update([0, 0], [0, 0])
    roll.update([0, 0], [0, 0])                          # last 4 are all right
    assert roll.result() == 1.0


def test_rolling_metric_partial_chunk_window():
    roll = RollingMetric(Accuracy(), window=3)
    roll.update([0, 0], [1, 1]).update([0, 0], [0, 0])
    assert roll.result() == pytest.approx(2 / 3)          # 1 wrong + 2 right


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        Accuracy().update([0, 1], [0])
    with pytest.raises(ValueError):
        F1Score(average="sum")
