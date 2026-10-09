"""Scoring rules and backtests used by the validation reports and benchmarks."""
from __future__ import annotations

import numpy as np
from scipy import stats


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss; lower is better. Standard loss for VaR forecasts."""
    d = y - q
    return np.maximum(alpha * d, (alpha - 1) * d)


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per row: E|X-y| - 0.5 E|X-X'|, from samples of shape (n_rows, n_samples)."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    term1 = np.abs(s - y[:, None]).mean(axis=1)
    i = np.arange(1, m + 1)
    term2 = (2.0 / m**2) * ((2 * i - m - 1) * s).sum(axis=1)
    return term1 - 0.5 * term2


def kupiec(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - alpha)
    elif x == n:
        lr = -2 * n * np.log(alpha)
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test (are VaR breaches clustered?). Returns (LR, p-value)."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / max(1, n00 + n01)
    pi1 = n11 / max(1, n10 + n11)
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    def ll(p, k0, k1):
        return (k0 * np.log(1 - p) if k0 and p < 1 else 0.0) + (k1 * np.log(p) if k1 and p > 0 else 0.0)

    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    lr = max(lr, 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def fz0(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """Fissler-Ziegel FZ0 joint loss for (VaR, ES) on the return scale (both negative). Lower is better."""
    e = np.minimum(es, -1e-8)
    hit = (y <= var).astype(float)
    return -hit * (var - y) / (alpha * e) + var / e + np.log(-e) - 1


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 0) -> tuple[float, float]:
    """DM test of equal mean loss with Newey-West variance. Negative stat means A has lower loss."""
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def var_es_from_samples(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.quantile(samples, alpha, axis=1)
    tail = np.where(samples <= q[:, None], samples, np.nan)
    es = np.nanmean(tail, axis=1)
    return q, es


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss for variance forecasts (robust to noisy proxies). Lower is better."""
    r = realized_var / forecast_var
    return r - np.log(r) - 1
