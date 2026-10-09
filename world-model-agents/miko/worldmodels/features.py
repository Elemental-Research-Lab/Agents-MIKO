"""State features. Every feature at date t uses data dated t or earlier only."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def _log_rv(r: pd.Series, w: int) -> pd.Series:
    return 0.5 * np.log(ANN * (r**2).rolling(w, min_periods=w).mean() + 1e-12)


def avg_pairwise_corr(rets: pd.DataFrame, window: int) -> pd.Series:
    x = rets.to_numpy()
    n, k = x.shape
    out = np.full(n, np.nan)
    off = ~np.eye(k, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(window - 1, n):
            c = np.corrcoef(x[t - window + 1 : t + 1].T)
            out[t] = np.nanmean(c[off])
    return pd.Series(out, index=rets.index)


def equity_state_features(returns: pd.DataFrame, levels: pd.DataFrame, stock_cols: list[str]) -> pd.DataFrame:
    """Market-state features for the equity panel (SP500, VIX, WTI, stocks)."""
    r = returns["SP500"]
    logp = np.log(levels["SP500"]).reindex(returns.index)
    f = pd.DataFrame(index=returns.index)
    f["rv5"] = _log_rv(r, 5)
    f["rv21"] = _log_rv(r, 21)
    f["rv63"] = _log_rv(r, 63)
    f["ret21"] = r.rolling(21).sum()
    f["ret63"] = r.rolling(63).sum()
    f["ret252"] = r.rolling(252).sum()
    f["dd252"] = logp - logp.rolling(252, min_periods=252).max()
    vix = levels["VIX"].reindex(returns.index)
    f["vix"] = np.log(vix)
    f["vrp"] = np.log(vix / 100.0) - f["rv21"]
    f["corr63"] = avg_pairwise_corr(returns[stock_cols], 63)
    f["disp21"] = returns[stock_cols].rolling(21).sum().std(axis=1)
    f["oil63"] = returns["WTI"].rolling(63).sum()
    f["oilrv21"] = _log_rv(returns["WTI"], 21)
    return f.dropna()


def generic_state_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Asset-agnostic state features for any return panel (used for the FX benchmark)."""
    f = pd.DataFrame(index=returns.index)
    sq = returns**2
    for w in (5, 21, 63):
        f[f"rv{w}"] = 0.5 * np.log(ANN * sq.rolling(w, min_periods=w).mean().mean(axis=1) + 1e-12)
    f["absret63"] = returns.rolling(63).sum().abs().mean(axis=1)
    if returns.shape[1] > 1:
        f["corr63"] = avg_pairwise_corr(returns, 63)
    return f.dropna()


def forward_targets(returns: pd.Series, levels: pd.Series, h: int = 21) -> pd.DataFrame:
    """What followed each date (uses future data: only ever used for dates whose window has closed)."""
    r = returns
    out = pd.DataFrame(index=r.index)
    out[f"fwd_ret{h}"] = r[::-1].rolling(h).sum()[::-1].shift(-1)
    out[f"fwd_logrv{h}"] = 0.5 * np.log(ANN * (r**2)[::-1].rolling(h).mean()[::-1].shift(-1) + 1e-12)
    lp = np.log(levels.reindex(r.index)).to_numpy()
    n = len(lp)
    mdd = np.full(n, np.nan)
    for t in range(n - h):
        seg = lp[t : t + h + 1]
        peak = np.maximum.accumulate(seg)
        mdd[t] = (seg - peak).min()
    out[f"fwd_mdd{h}"] = mdd
    return out
