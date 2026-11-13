import numpy as np
import pytest

from _helpers import make_blobs
from numcompute_stream import (ColumnTransformer, DecisionTreeClassifier, OneHotEncoder, Pipeline,
                               SimpleImputer, StandardScaler, make_pipeline)


def mixed_data(rng, n=200):
    """Two numeric columns (with NaN) and one categorical column, as an object array."""
    X, y = make_blobs(n, rng, n_features=2, n_classes=2)
    X[rng.random(X.shape) < 0.05] = np.nan
    site = np.where(y == 1, "A", "B").astype(object)
    return np.column_stack([X.astype(object), site]), y


def make_model(buffer_size=100):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), [0, 1]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder())]), [2]),
    ])
    return Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(max_depth=4,
                                                                   buffer_size=buffer_size))])


def test_pipeline_partial_fit_updates_every_step(rng):
    X, y = make_blobs(300, rng)
    pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=100))])
    for start in range(0, 300, 50):
        pipe.partial_fit(X[start:start + 50], y[start:start + 50])
    np.testing.assert_allclose(pipe["sc"].mean_, X.mean(axis=0), rtol=1e-12)
    assert pipe["tree"].n_retrains_ == 3
    assert pipe.score(X, y) > 0.8


def test_column_transformer_with_mixed_types(rng):
    X, y = mixed_data(rng)
    model = make_model()
    model.partial_fit(X[:100], y[:100]).partial_fit(X[100:], y[100:])
    out = model["pre"].transform(X)
    assert out.shape == (200, 4) and not np.isnan(out).any()
    assert model["pre"].get_feature_names_out(["t", "v", "site"]) == \
        ["t", "v", "site=B", "site=A"] or \
        model["pre"].get_feature_names_out(["t", "v", "site"]) == ["t", "v", "site=A", "site=B"]
    assert model.predict(X).shape == (200,)
    assert model.predict_proba(X).shape == (200, 2)


def test_new_category_mid_stream_does_not_break_the_tree(rng):
    X, y = mixed_data(rng, n=300)
    X[200:, 2] = "C"                                   # a new site appears
    model = make_model(buffer_size=100)
    for start in range(0, 300, 50):
        model.partial_fit(X[start:start + 50], y[start:start + 50])
    assert model["pre"].transform(X).shape[1] == 5
    assert model.predict(X[250:]).shape == (50,)


def test_fit_only_external_step_is_fitted_once(rng):
    class Shift:                                      # like a step from another library
        def __init__(self):
            self.calls = 0

        def fit(self, X):                             # takes X only and returns None
            self.calls += 1
            self.mean = np.mean(X, axis=0)

        def transform(self, X):
            return np.asarray(X) - self.mean

    X, y = make_blobs(200, rng)
    shift = Shift()
    pipe = Pipeline([("shift", shift), ("tree", DecisionTreeClassifier(buffer_size=50))])
    for start in range(0, 200, 50):
        pipe.partial_fit(X[start:start + 50], y[start:start + 50])
    assert shift.calls == 1                           # frozen after the first chunk
    pipe.reset()
    pipe.partial_fit(X[:50], y[:50])
    assert shift.calls == 2


def test_batch_fit_uses_the_group_pipeline(rng):
    X, y = make_blobs(200, rng)
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3))
    pipe.fit(X, y)
    assert [name for name, _ in pipe.steps] == ["standardscaler", "decisiontreeclassifier"]
    assert pipe.score(X, y) > 0.8


def test_passthrough_and_empty_chunk(rng):
    X, y = make_blobs(100, rng)
    pipe = Pipeline([("skip", "passthrough"), ("tree", DecisionTreeClassifier(buffer_size=50))])
    pipe.partial_fit(np.empty((0, 4)), np.empty(0))
    assert pipe["tree"].buffer_count == 0
    pipe.partial_fit(X, y)
    assert pipe.is_fitted and pipe.memory_bytes() > 0


def test_pipeline_errors():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(TypeError):
        Pipeline([("bad", object()), ("tree", DecisionTreeClassifier())])

    class FitOnly:
        def fit(self, X, y):
            return self

    with pytest.raises(TypeError):
        Pipeline([("model", FitOnly())]).partial_fit([[1.0]], [0])
    ct = ColumnTransformer([("sc", StandardScaler(), [0])])
    ct.partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        ct.transform([[1.0]])


def test_transform_only_step_is_allowed(rng):
    class Double:                                      # no fit at all
        def transform(self, X):
            return np.asarray(X, dtype=float) * 2

    X, y = make_blobs(100, rng)
    streamed = Pipeline([("double", Double()), ("tree", DecisionTreeClassifier(buffer_size=50))])
    streamed.partial_fit(X[:50], y[:50]).partial_fit(X[50:], y[50:])
    batch = Pipeline([("double", Double()), ("tree", DecisionTreeClassifier())]).fit(X, y)
    assert streamed.score(X, y) > 0.8 and batch.score(X, y) == 1.0
