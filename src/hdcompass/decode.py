"""Grid hidden-Markov decoder of head direction from spike counts.

The latent state is a single angle, so we discretise the circle into ``n_grid`` bins and run
exact forward–backward on that grid. A particle filter would only approximate the same
posterior with Monte-Carlo noise; in 1-D the grid is exact up to its resolution, deterministic
and cheaper. The transition is a wrapped Gaussian random walk (std ``sigma·√bin_size``) mixed
with a uniform jump of probability ``eps``; being circulant, it is applied by FFT convolution.
Emissions are independent Poisson counts with rate ``tuning · bin_size``.

JAX runs in float64 (``jax_enable_x64`` is switched on at import): posterior tails far from the
mode are below float32 range and FFT round-off would otherwise dominate them.
"""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pynapple as nap  # noqa: E402
from scipy.special import gammaln  # noqa: E402

from .circular import TWO_PI, wrap  # noqa: E402
from .features import bin_counts  # noqa: E402

SIGMA_GRID = np.geomspace(0.1, 50.0, 12)
_TINY = 1e-300


def transition_kernel(n_grid, bin_size, sigma, eps=0.0):
    """Circulant transition kernel: ``k[j]`` = P(angle moves by ``j`` grid steps in one bin).

    Parameters
    ----------
    n_grid : int
    bin_size : float
        Seconds.
    sigma : float
        Diffusion (rad/√s); ``np.inf`` gives a uniform kernel (independent bins).
    eps : float
        Probability of a uniform jump per bin.

    Returns
    -------
    ndarray, shape (n_grid,)
        Sums to 1.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if np.isinf(sigma):
        k = np.full(n_grid, 1.0 / n_grid)
    else:
        s = sigma * np.sqrt(bin_size)
        d = np.arange(n_grid) * TWO_PI / n_grid
        d = np.where(d > np.pi, d - TWO_PI, d)
        w = int(np.ceil(4 * s / TWO_PI)) + 1  # wraps needed to cover ±4 std
        m = np.arange(-w, w + 1)
        k = np.exp(-((d[:, None] + TWO_PI * m[None, :]) ** 2) / (2 * s**2)).sum(axis=1)
        if k.sum() == 0:  # s far below grid spacing
            k[0] = 1.0
        k /= k.sum()
    return (1 - eps) * k + eps / n_grid


@jax.jit
def _fb(log_e, kf, reset):
    n = log_e.shape[1]
    m = log_e.max(axis=1, keepdims=True)
    e = jnp.exp(log_e - m)
    uniform = jnp.full(n, 1.0 / n)

    def conv(x):
        return jnp.clip(jnp.fft.irfft(jnp.fft.rfft(x) * kf, n=n), 0.0)

    def fwd(alpha, x):
        e_t, r = x
        prior = jnp.where(r, uniform, jnp.maximum(conv(alpha), _TINY))
        a = prior * e_t
        c = a.sum()
        return a / c, (a / c, c)

    _, (alpha, c) = jax.lax.scan(fwd, uniform, (e, reset))

    def bwd(beta_next, x):
        e_next, c_next, r_next = x
        b = jnp.where(r_next, jnp.ones(n), conv(e_next * beta_next) / c_next)
        return b, b

    _, beta = jax.lax.scan(bwd, jnp.ones(n), (e[1:], c[1:], reset[1:]), reverse=True)
    beta = jnp.concatenate([beta, jnp.ones((1, n))])
    post = alpha * beta
    post = post / post.sum(axis=1, keepdims=True)
    return post, jnp.log(c).sum() + m.sum()


def forward_backward(counts, tuning, bin_size, sigma, eps=0.0, resets=None):
    """Exact forward–backward on the circular grid, in log-space-normalised (scaled) form.

    Parameters
    ----------
    counts : array_like, shape (T, units)
        Spike counts per bin.
    tuning : array_like, shape (units, n_grid)
        Rates (Hz) at grid angles ``(i + 0.5)·2π/n_grid``. Must be > 0.
    bin_size : float
        Seconds.
    sigma : float
        Diffusion (rad/√s); ``np.inf`` = independent bins.
    eps : float
        Uniform-jump probability per bin.
    resets : array_like of bool, shape (T,), optional
        True where a new epoch starts: the prior restarts uniform and no message crosses the
        boundary, which is identical to running each epoch separately. Bin 0 always resets.

    Returns
    -------
    posterior : ndarray, shape (T, n_grid)
        Rows sum to 1.
    loglik : float
        Log marginal likelihood of all counts (summed over epochs).
    """
    counts = np.asarray(counts, dtype=float)
    lam = np.asarray(tuning, dtype=float) * bin_size
    n = lam.shape[1]
    if len(counts) == 0:
        return np.empty((0, n)), 0.0
    resets = np.zeros(len(counts), bool) if resets is None else np.array(resets, bool)
    resets[0] = True
    log_e = counts @ np.log(lam) - lam.sum(axis=0)
    kf = np.fft.rfft(transition_kernel(n, bin_size, sigma, eps))
    post, ll = _fb(jnp.asarray(log_e), jnp.asarray(kf), jnp.asarray(resets))
    return np.asarray(post), float(ll) - float(gammaln(counts + 1).sum())


@dataclass(frozen=True)
class DecodeResult:
    """Output of :func:`decode`.

    Attributes
    ----------
    angle : nap.Tsd
        Posterior circular mean (rad) per bin.
    variance : nap.Tsd
        Posterior circular variance ``1 − |E[e^{iθ}]|`` per bin, in [0, 1].
    loglik : float
        Log marginal likelihood.
    sigma, eps, bin_size : float
        Model settings used.
    """

    angle: nap.Tsd
    variance: nap.Tsd
    loglik: float
    sigma: float
    eps: float
    bin_size: float


def _counts(spikes, tuning, ep, bin_size):
    if np.shape(tuning)[0] != len(spikes):
        raise ValueError(f"tuning has {np.shape(tuning)[0]} rows, spikes has {len(spikes)} units")
    parts = [(t, c) for t, c in bin_counts(spikes, ep, bin_size) if len(t)]
    if not parts:
        raise ValueError("no complete bins in ep")
    t = np.concatenate([p[0] for p in parts])
    counts = np.concatenate([p[1] for p in parts])
    resets = np.concatenate([np.arange(len(p[0])) == 0 for p in parts])
    support = nap.IntervalSet(
        [p[0][0] - bin_size / 2 for p in parts], [p[0][-1] + bin_size / 2 for p in parts]
    )
    return t, counts, resets, support


def fit_sigma(spikes, tuning, ep, bin_size=0.02, eps=0.0, grid=SIGMA_GRID):
    """Choose ``sigma`` by maximum marginal likelihood over ``grid``.

    Returns
    -------
    sigma : float
    logliks : ndarray, shape (len(grid),)
    """
    _, counts, resets, _ = _counts(spikes, tuning, ep, bin_size)
    ll = np.array([forward_backward(counts, tuning, bin_size, s, eps, resets)[1] for s in grid])
    return float(np.asarray(grid)[np.argmax(ll)]), ll


def decode(spikes, tuning, ep, bin_size=0.02, sigma=None, eps=0.0, sigma_grid=SIGMA_GRID):
    """Decode head direction with the grid HMM, per epoch (never across gaps).

    Parameters
    ----------
    spikes : nap.TsGroup
        Units in the same order as ``tuning`` rows.
    tuning : ndarray, shape (units, n_grid)
    ep : nap.IntervalSet
    bin_size : float
    sigma : float, optional
        Diffusion (rad/√s). None → :func:`fit_sigma` over ``sigma_grid``. ``np.inf`` gives the
        independent-bin Bayesian decoder (as ``nap.decode_bayes`` with a uniform prior), the
        baseline for comparison.
    eps : float
    sigma_grid : array_like

    Returns
    -------
    DecodeResult
    """
    if sigma is None:
        sigma, _ = fit_sigma(spikes, tuning, ep, bin_size, eps, sigma_grid)
    t, counts, resets, support = _counts(spikes, tuning, ep, bin_size)
    post, ll = forward_backward(counts, tuning, bin_size, sigma, eps, resets)
    n = post.shape[1]
    z = post @ np.exp(1j * (np.arange(n) + 0.5) * TWO_PI / n)
    return DecodeResult(
        angle=nap.Tsd(t=t, d=wrap(np.angle(z)), time_support=support),
        variance=nap.Tsd(t=t, d=1 - np.abs(z), time_support=support),
        loglik=ll,
        sigma=float(sigma),
        eps=float(eps),
        bin_size=float(bin_size),
    )
