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


def test_is_ring_needs_to_beat_every_shuffle_when_few():
    null = [1.0, 1.0, 1.0, 1.4, 1.6]
    assert is_ring(1.58, null) == (False, 2 / 6)
    assert is_ring(1.61, null)[0]


def test_parallel_null_matches_serial_and_density_filter():
    from hdcompass.topology import density_filter

    th = _walk(600, 2)
    X = np.column_stack(
        [np.cos(th), np.sin(th), 0.05 * np.random.default_rng(3).standard_normal(600)]
    )
    serial = shuffle_null(X, n_shuffles=3, n_landmarks=150)
    parallel = shuffle_null(X, n_shuffles=3, n_landmarks=150, n_jobs=2)
    np.testing.assert_array_equal(serial, parallel)
    # density filter keeps the requested fraction and drops scattered off-ring points
    rng = np.random.default_rng(4)
    ring = np.column_stack([np.cos(th), np.sin(th)]) + 0.02 * rng.standard_normal((600, 2))
    noise = rng.uniform(-0.6, 0.6, (60, 2))  # sparse points inside the hole
    kept = density_filter(np.vstack([ring, noise]), keep=0.8)
    assert len(kept) == int(np.ceil(0.8 * 660)) or abs(len(kept) - 0.8 * 660) <= 1
    assert np.mean(np.linalg.norm(kept, axis=1) < 0.7) < 0.02
