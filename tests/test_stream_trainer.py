import logging

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from _helpers import make_blobs
from numcompute_stream import DecisionTreeClassifier, Pipeline, StandardScaler, StreamTrainer
from numcompute_stream import visualise as vis
from numcompute_stream.io import chunk_stream


def make_trainer(**kwargs):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=200,
                                                     random_state=0))])
    return StreamTrainer(pipe, rolling_window=200, **kwargs)


def test_prequential_log_has_one_record_per_chunk(rng):
    X, y = make_blobs(1000, rng)
    trainer = make_trainer()
    trainer.run(chunk_stream(X, y, chunk_size=100))
    h = trainer.history_arrays()
    assert len(trainer.history) == 10
    assert h["samples_seen"][-1] == 1000
    # the first chunks come before the first retrain, so they cannot be scored
    assert np.isnan(h["chunk_accuracy"][0]) and np.isnan(h["chunk_accuracy"][1])
    assert h["retrained"].sum() == 5
    assert trainer.summary()["final_cumulative_accuracy"] > 0.85


def test_test_then_train_order(rng):
    # if the chunk were trained on before scoring, a deep tree would score 100%
    X = rng.random((400, 3))
    y = rng.integers(0, 2, size=400)              # pure noise
    pipe = DecisionTreeClassifier(buffer_size=100)
    trainer = StreamTrainer(pipe)
    trainer.run(chunk_stream(X, y, chunk_size=100))
    assert trainer.history[-1]["chunk_accuracy"] < 0.8


def test_binary_problem_logs_auc_and_csv(rng, tmp_path):
    X, y = make_blobs(600, rng, n_classes=2)
    trainer = make_trainer()
    trainer.run(chunk_stream(X, y, chunk_size=100))
    assert 0.5 < trainer.summary()["auc"] <= 1.0
    path = tmp_path / "log.csv"
    trainer.to_csv(path)
    assert path.read_text().splitlines()[0].startswith("chunk,n_samples")
    assert len(path.read_text().splitlines()) == 7


def test_empty_chunk_is_skipped_and_max_chunks(rng):
    X, y = make_blobs(500, rng)
    trainer = make_trainer()
    assert trainer.process_chunk(np.empty((0, 4)), np.empty(0)) is None
    trainer.run(chunk_stream(X, y, chunk_size=100), max_chunks=3)
    assert len(trainer.history) == 3


def test_log_lines_are_written(rng, caplog):
    X, y = make_blobs(400, rng)
    trainer = make_trainer(log_every=2, name="demo")
    with caplog.at_level(logging.INFO, logger="numcompute_stream"):
        trainer.run(chunk_stream(X, y, chunk_size=100))
    assert sum("[demo] chunk" in m for m in caplog.messages) == 2


def test_trainer_argument_checks():
    with pytest.raises(TypeError):
        StreamTrainer(object())
    with pytest.raises(TypeError):
        StreamTrainer(StandardScaler())                # no predict
    with pytest.raises(ValueError):
        make_trainer().to_csv("unused.csv")            # empty history


def test_plots_return_figures(rng, tmp_path):
    X, y = make_blobs(800, rng)
    trainer = make_trainer()
    trainer.run(chunk_stream(X, y, chunk_size=100))
    h = trainer.history_arrays()
    retrains = np.flatnonzero(h["retrained"])
    figures = [
        vis.plot_metric_over_time(h["rolling_accuracy"], retrain_points=retrains, smooth=3,
                                  save_path=tmp_path / "m.png", show=False),
        vis.compare_models(h["chunk_accuracy"], h["cumulative_accuracy"], show=False),
        vis.compare_models({"a": h["chunk_accuracy"], "b": h["rolling_accuracy"]}, show=False),
        vis.plot_predictions_vs_ground_truth(trainer.last_truth, trainer.last_predictions,
                                             show=False),
        vis.plot_confusion_matrix(trainer.confusion.result(), trainer.confusion.labels_,
                                  normalise=True, show=False),
        vis.plot_feature_importances(trainer.model["tree"].feature_importances_, show=False),
        vis.plot_stream_dashboard(h, show=False),
    ]
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figures)
    assert (tmp_path / "m.png").stat().st_size > 0
    with pytest.raises(ValueError):
        vis.plot_metric_over_time([], show=False)
    with pytest.raises(ValueError):
        vis.plot_predictions_vs_ground_truth([0, 1], [0], show=False)
