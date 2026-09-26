import numpy as np

from hdcompass.topology import h1_persistence, is_ring, shuffle_null


def _walk(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.3, n))


def test_noisy_circle_is_ring():
    rng = np.random.default_rng(0)
    th = _walk(1500, 0)
    X = np.column_stack([np.cos(th), np.sin(th)]) + 0.1 * rng.standard_normal((1500, 2))
    null = shuffle_null(X, n_shuffles=10, n_landmarks=200)
    ring, p = is_ring(X, null, n_landmarks=200)
    assert ring and p <= 1 / 11 + 1e-12
    h1 = h1_persistence(X, n_landmarks=200)
    assert np.all(np.diff(h1["lifetimes"]) <= 0) and h1["ring_score"] > 3


def test_gaussian_blob_is_not_ring():
    X = np.random.default_rng(1).standard_normal((1500, 3))
    null = shuffle_null(X, n_shuffles=10, n_landmarks=200)
    ring, p = is_ring(X, null, n_landmarks=200)
    assert not ring and p > 0.05
