import numpy as np

from hdcompass.align import angle_error
from hdcompass.decode import decode, fit_sigma, forward_backward, transition_kernel
from hdcompass.features import bin_counts
from hdcompass.tuning import fit_tuning


def test_posterior_rows_sum_to_one_and_resets_split_epochs(hd40):
    spikes, _, tc = hd40
    ((_, counts),) = bin_counts(spikes, spikes.time_support, 0.02)
    counts = counts[:2000]
    post, ll = forward_backward(counts, tc, 0.02, sigma=1.0)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)
    assert np.isfinite(ll)
    # a reset is exactly equivalent to two independent runs
    resets = np.zeros(2000, bool)
    resets[1200] = True
    post_r, ll_r = forward_backward(counts, tc, 0.02, 1.0, resets=resets)
    pa, la = forward_backward(counts[:1200], tc, 0.02, 1.0)
    pb, lb = forward_backward(counts[1200:], tc, 0.02, 1.0)
    np.testing.assert_allclose(post_r, np.vstack([pa, pb]), atol=1e-9)
    assert abs(ll_r - (la + lb)) < 1e-6


def test_kernel():
    for s in (0.01, 1.0, 50.0, np.inf):
        k = transition_kernel(60, 0.02, s, eps=0.01)
        assert abs(k.sum() - 1) < 1e-12 and np.all(k > 0)
    np.testing.assert_allclose(transition_kernel(60, 0.02, np.inf), 1 / 60)


def test_hmm_accuracy_beats_independent_decoder(hd40):
    spikes, angle, tc = hd40
    hmm = decode(spikes, tc, spikes.time_support, sigma=1.0)
    indep = decode(spikes, tc, spikes.time_support, sigma=np.inf)
    assert angle_error(hmm.angle, angle) < 0.2
    assert angle_error(hmm.angle, angle) <= angle_error(indep.angle, angle)
    assert np.all((hmm.variance.values >= -1e-12) & (hmm.variance.values <= 1 + 1e-12))


def test_fitted_tuning_decodes(hd40):
    spikes, angle, _ = hd40
    tc = fit_tuning(spikes, angle, spikes.time_support)
    assert angle_error(decode(spikes, tc, spikes.time_support, sigma=1.0).angle, angle) < 0.2


def test_marginal_likelihood_peaks_at_true_sigma(hd40):
    spikes, _, tc = hd40
    grid = np.array([0.1, 1.0, 10.0])
    best, ll = fit_sigma(spikes, tc, spikes.time_support, grid=grid)
    assert best == 1.0 and ll[1] > ll[0] and ll[1] > ll[2]
