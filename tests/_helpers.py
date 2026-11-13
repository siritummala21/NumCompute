"""Small helpers shared by the stream tests."""

import numpy as np


def random_chunks(n, rng, max_size=40):
    """Split range(n) into consecutive slices of random size (some may be empty).

    Streaming results must not depend on how the data is cut into chunks, so
    most tests feed the data through these random cuts.
    """
    n_cuts = int(rng.integers(1, max(2, n // max_size + 3)))
    cuts = np.sort(rng.integers(0, n + 1, size=n_cuts))
    edges = np.concatenate([[0], cuts, [n]])
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def make_blobs(n, rng, n_features=4, n_classes=3, spread=1.0):
    """Gaussian clusters: an easy classification problem for the tree tests."""
    centres = rng.normal(0, 4, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n)
    X = centres[y] + rng.normal(0, spread, size=(n, n_features))
    return X, y
