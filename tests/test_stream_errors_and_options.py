"""Error messages and less common options.

Written after measuring which lines the other tests never ran: every error
path should raise the documented exception with a useful message, and every
option should give the expected result.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from _helpers import make_blobs
from numcompute_stream import (DDM, ColumnTransformer, DecisionTreeClassifier, F1Score,
                               MinMaxScaler, NotFittedError, OneHotEncoder, Pipeline,
                               Precision, Recall, RollingMetric, SimpleImputer, StandardScaler,
                               StreamingAUC, StreamingHistogram, StreamingQuantile,
                               StreamTrainer, chunk_stream, read_csv, write_csv)
from numcompute_stream import visualise as vis
from numcompute_stream._validation import (check_array, check_labels, check_random_state,
                                           check_scalar)


# ------------------------------------------------------------- input checks
@pytest.mark.parametrize("call, error", [
    (lambda: check_array(None), TypeError),
    (lambda: check_array([["a", "b"]]), TypeError),
    (lambda: check_array(np.empty((0, 3))), ValueError),
    (lambda: check_labels(None), TypeError),
    (lambda: check_labels(np.zeros((3, 2))), ValueError),
    (lambda: check_labels([]), ValueError),
    (lambda: check_scalar("3", "x"), TypeError),
    (lambda: check_scalar(float("inf"), "x"), ValueError),
    (lambda: check_scalar(5, "x", max_val=4), ValueError),
    (lambda: check_random_state("seed"), TypeError),
])
def test_input_checks_raise(call, error):
    with pytest.raises(error):
        call()


def test_column_vector_labels_and_generator_seed():
    assert check_labels(np.array([[1], [2]])).shape == (2,)
    rng = np.random.default_rng(3)
    assert check_random_state(rng) is rng


def test_set_params_is_checked_and_all_or_nothing():
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=50)
    assert tree.set_params(max_depth=4).max_depth == 4
    with pytest.raises(ValueError):
        tree.set_params(max_depth=2, buffer_size=0)    # second value is bad
    assert tree.max_depth == 4                          # nothing changed
    with pytest.raises(ValueError):
        tree.set_params(not_a_parameter=1)
    with pytest.raises(ValueError):
        MinMaxScaler().set_params(feature_range=(1, 0))


# ------------------------------------------------------------- io
def test_csv_file_problems(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_csv(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_csv(header_only)
    with pytest.raises(ValueError):
        list(chunk_stream(np.ones((3, 2)), np.ones(2)))
    with pytest.raises(ValueError):
        write_csv(tmp_path / "x.csv", np.ones((2, 2)), ["only_one_name"])


def test_float_labels_stay_float(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("a,label\n1,0.5\n2,1.5\n")
    assert read_csv(p, target="label").y.tolist() == [0.5, 1.5]


# ------------------------------------------------------------- preprocessing
def test_preprocessing_not_fitted_and_bad_options():
    for est in (SimpleImputer(), OneHotEncoder()):
        with pytest.raises(NotFittedError):
            est.transform([[1.0]])
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="drop")
    with pytest.raises(ValueError):
        SimpleImputer(fill_value=float("nan"))
    with pytest.raises(ValueError):
        OneHotEncoder().partial_fit(np.array(["a", "b"], dtype=object))      # 1-D


def test_empty_chunks_in_preprocessing():
    imp = SimpleImputer().partial_fit([[1.0, 2.0]])
    assert imp.partial_fit(np.empty((0, 2))).transform(np.empty((0, 2))).shape == (0, 2)
    enc = OneHotEncoder().partial_fit(np.array([["a"]], dtype=object))
    assert enc.partial_fit(np.empty((0, 1), dtype=object)).transform(
        np.empty((0, 1), dtype=object)).shape == (0, 1)


def test_scaler_options():
    X = np.array([[1.0], [3.0]])
    no_mean = StandardScaler(with_mean=False).partial_fit(X)
    np.testing.assert_allclose(no_mean.transform(X), [[1.0], [3.0]])        # std is 1
    no_std = StandardScaler(with_std=False).partial_fit(X)
    np.testing.assert_allclose(no_std.transform(X), [[-1.0], [1.0]])
    np.testing.assert_allclose(no_std.inverse_transform(no_std.transform(X)), X)


# ------------------------------------------------------------- statistics
def test_statistics_before_data_and_bad_range():
    with pytest.raises(RuntimeError):
        StreamingQuantile().result()
    with pytest.raises(RuntimeError):
        StreamingHistogram().result()
    with pytest.raises(ValueError):
        StreamingHistogram(range=(1.0, 1.0))
    h = StreamingHistogram(bins=2).update([[0.0], [1.0]])
    assert h.update(np.empty((0, 1))).result()[0].sum() == 2


# ------------------------------------------------------------- metrics
def test_micro_averages_and_missing_positive_label():
    y_true, y_pred = [0, 1, 2, 2], [0, 2, 2, 1]
    assert Precision("micro").update(y_true, y_pred).result() == pytest.approx(0.5)
    assert Recall("micro").update(y_true, y_pred).result() == pytest.approx(0.5)
    assert Precision("binary", pos_label=7).update(y_true, y_pred).result() == 0.0
    assert F1Score().update([], []).result() == 0.0                         # empty chunk


def test_auc_and_rolling_errors():
    with pytest.raises(ValueError):
        StreamingAUC().update([0, 1], [0.5])
    with pytest.raises(ValueError, match="classes"):
        StreamingAUC().update([0, 1, 2], np.eye(3)).result()
    assert np.isnan(RollingMetric().result())


# ------------------------------------------------------------- tree
@pytest.mark.parametrize("max_features, expected", [("log2", 3), (2, 2), (20, 9)])
def test_number_of_features_tried(max_features, expected):
    assert DecisionTreeClassifier(max_features=max_features)._n_try(9) == expected


def test_constant_feature_is_skipped_by_loop_splitter():
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    y = (np.arange(20) >= 10).astype(int)
    loop = DecisionTreeClassifier(splitter="loop").fit(X, y)
    vec = DecisionTreeClassifier().fit(X, y)
    assert loop.score(X, y) == vec.score(X, y) == 1.0


def test_empty_chunk_does_not_change_the_tree(rng):
    X, y = make_blobs(50, rng)
    tree = DecisionTreeClassifier(buffer_size=10).partial_fit(X, y)
    before = tree.n_retrains_
    tree.partial_fit(np.empty((0, 4)), np.empty(0))
    assert tree.n_retrains_ == before and not tree.last_partial_fit_retrained_


# ------------------------------------------------------------- pipeline
def test_pipeline_and_column_transformer_errors():
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", DecisionTreeClassifier())])
    with pytest.raises(TypeError):
        Pipeline([("scale", StandardScaler()), ("last", object())])
    with pytest.raises(TypeError):
        ColumnTransformer([("bad", object(), [0])])
    with pytest.raises(ValueError):
        ColumnTransformer([("s", StandardScaler(), [0])]).partial_fit(np.ones(3))


def test_column_transformer_batch_fit_and_passthrough():
    X = np.array([[1.0, "a", 5.0], [2.0, "b", 6.0], [3.0, "a", 7.0]], dtype=object)
    ct = ColumnTransformer([("num", StandardScaler(), [0]), ("cat", OneHotEncoder(), [1])],
                           remainder="passthrough")
    out = ct.fit_transform(X)
    np.testing.assert_allclose(out[:, 3], [5.0, 6.0, 7.0])                 # passed through
    assert ct.get_feature_names_out(["n", "c", "z"]) == ["n", "c=a", "c=b", "z"]
    ct.fit(X[:, :2])                                                        # fit starts again
    assert ct.transform(X[:, :2]).shape == (3, 3)
    pipe = Pipeline([("ct", ct), ("tree", DecisionTreeClassifier())]).fit(X[:, :2], [0, 1, 0])
    np.testing.assert_array_equal(pipe.predict(X[:, :2]), [0, 1, 0])


def test_partial_fit_transform_and_empty_chunk():
    sc = StandardScaler()
    out = sc.partial_fit_transform(np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(out.ravel(), [-1.0, 1.0])
    ct = ColumnTransformer([("s", StandardScaler(), [0])]).partial_fit([[1.0], [2.0]])
    assert ct.partial_fit(np.empty((0, 1))).transform([[1.5]]).shape == (1, 1)


# ------------------------------------------------------------- trainer, DDM, plots
def test_trainer_and_ddm_errors(rng):
    with pytest.raises(TypeError, match="flush"):
        StreamTrainer(Pipeline([("s", StandardScaler()), ("x", _NoFlush())]),
                      drift_detector=DDM())
    with pytest.raises(ValueError):
        DDM().update([0, 1], [0])
    ddm = DDM(min_samples=1000).update([0, 1], [1, 1])                    # too few samples yet
    assert ddm.state == "stable" and DDM().update_counts(0, 0).n_ == 0
    assert StreamTrainer(DecisionTreeClassifier()).history_arrays() == {}


class _NoFlush:
    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_trainer_handles_a_model_that_is_not_fitted():
    class Lazy(_NoFlush):
        def predict(self, X):
            raise NotFittedError("not yet")

    record = StreamTrainer(Lazy()).process_chunk(np.ones((5, 2)), np.zeros(5))
    assert np.isnan(record["chunk_accuracy"])


def test_more_plots_and_plot_errors(tmp_path):
    fig = vis.plot_buffer_tradeoff([100, 500], [0.7, 0.73], [0.2, 0.1],
                                   save_path=tmp_path / "b.png", show=False)
    assert (tmp_path / "b.png").stat().st_size > 0 and len(fig.axes) == 2
    vis.compare_models([[0.5, 0.6], [0.4, 0.7]], labels=["a", "b"], show=False)   # list form
    vis.plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1], show=False)        # with errors
    with pytest.raises(ValueError):
        vis.compare_models({str(i): [0.5] for i in range(9)}, show=False)
    with pytest.raises(ValueError):
        vis.plot_predictions_vs_ground_truth([], [], show=False)
    with pytest.raises(ValueError):
        vis.plot_confusion_matrix(np.ones((2, 3)), show=False)
