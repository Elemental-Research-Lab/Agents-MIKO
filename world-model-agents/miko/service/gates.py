"""Model gates: checks a day's freshly fitted world models must pass before they go live.

Trust rule 11: gates decide. If any gate fails, the service keeps serving the last models that
passed (and says so on /ready), instead of serving models that look broken.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..agents.context import AgentContext
from ..evaluation import kupiec


@dataclass
class Check:
    name: str
    passed: bool
    value: float | str
    rule: str


def run_gates(ctx: AgentContext, backtest_days: int = 250) -> list[Check]:
    as_of = ctx.latest
    out: list[Check] = []

    # 1. State Encoder sanity
    enc = ctx.encoder(as_of)
    F = ctx.features.loc[:as_of]
    probs = enc.filtered_regime_probs(F.iloc[-300:]).iloc[-1]
    out.append(Check("regime probabilities sum to 1", bool(abs(probs.sum() - 1) < 1e-6 and np.isfinite(probs).all()),
                     float(probs.sum()), "|sum - 1| < 1e-6"))
    _, nov = enc.novelty(F)
    out.append(Check("novelty percentile in [0, 1]", bool(0 <= nov <= 1), float(nov), "0 <= x <= 1"))
    fv = enc.forward_risk(F).get("fwd_vol21", float("nan"))
    out.append(Check("forward volatility plausible", bool(0.02 < fv < 3.0), float(fv), "2% < annualized vol < 300%"))

    # 2. Path generator on the S&P 500: simulated vs realized volatility, and a causal VaR backtest
    m = ctx.path_model(as_of, ["SP500"], "statepath")
    r = ctx.returns["SP500"].loc[:as_of]
    sim = m.sample(4000, 1, 0, r.to_frame())[:, 0, 0]
    ratio = float(sim.std() / (r.iloc[-21:].std() + 1e-12))
    out.append(Check("1-day simulated vs 21-day realized vol", bool(0.33 < ratio < 3.0), ratio, "0.33 < ratio < 3"))
    # rolling 1-day 99% VaR over the last N sessions, volatility filter updated causally
    from ..worldmodels.paths import garch_filter

    p = m.prm[0]
    x = 100 * r.to_numpy()
    sig = np.sqrt(garch_filter(x, p))  # sig[t] = forecast for day t made at t-1
    z = np.sort(m.z[:, 0])
    q01 = np.quantile(z, 0.01)
    idx = np.arange(len(x) - backtest_days, len(x))
    var = p["mu"] + q01 * sig[idx]  # filter + residual quantile (the state head is a same-day overlay)
    hits = x[idx] < var
    pval = kupiec(hits, 0.01)[1]
    out.append(Check("1-day 99% VaR backtest, last 250 sessions (Kupiec)", bool(pval >= 0.001 and hits.mean() <= 0.04),
                     f"{hits.mean():.3f} breach rate, p={pval:.3g}", "Kupiec p >= 0.001 and breach rate <= 4%"))

    # 3. Scenario engine: a canned shock is hit exactly and fills in a coherent sell-off
    eng = ctx.scenario_engine(as_of, 21)
    c = eng.condition({"SP500": np.log(0.9)}, "kernel")
    paths = eng.paths(c, 200, 0)
    hit = float(np.max(np.abs(paths.sum(axis=1)[:, eng.factors.index("SP500")] - np.log(0.9))))
    out.append(Check("scenario paths hit the shock", bool(hit < 1e-9), hit, "max error < 1e-9"))
    stocks = [eng.factors.index(s) for s in ctx.stock_names]
    share_down = float((c.mean[stocks] < 0).mean())
    out.append(Check("S&P -10% fills in mostly falling stocks", bool(share_down >= 0.7), share_down, ">= 70% of stocks down"))
    return out


def gates_to_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
