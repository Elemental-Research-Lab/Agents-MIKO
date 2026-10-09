"""Planner benchmark: monthly portfolio planning on 20 US large caps, 1995-2022, out of sample.

Setup in the spirit of DeMiguel, Garlappi and Uppal (2009), "Optimal versus naive
diversification" (Review of Financial Studies): rebalance monthly, estimate only on data
before each rebalance date, hold for the month, compare against 1/N.

Two families are compared.
A. Fully invested classics (no risk goal): 1/N, inverse volatility, minimum variance
   (Ledoit-Wolf), mean-variance max Sharpe (sample), equal risk contribution, HRP, minimum
   CVaR on historical scenarios. Implementations from the open-source skfolio library.
B. Risk-goal plans: "21-day CVaR 95% <= 6%, max 20% per name" (and "annualized vol <= 10%").
   The Planner (world-model futures + LP + check on unseen futures) against the same goal met
   by scaling 1/N or minimum CVaR with historical risk estimates, plus ablations that keep the
   Planner's optimizer but swap the world model for GARCH-t or block-bootstrap futures.

Scores: annualized return, volatility, Sharpe (with the Jobson-Korkie-Memmel test vs 1/N),
certainty-equivalent return (gamma=1), max drawdown, turnover, Sharpe net of 10 bps costs,
and for risk-goal plans the realized 21-day CVaR 95% against the 6% goal, overall and in crisis
months.

Caveat: the 20 stocks survived to 2022 (skfolio dataset), which flatters every strategy's
returns equally; comparisons between strategies are what matter.

Run:  python bench/planner_portfolio.py [--fast]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, md_table, save_json

from miko import data as D
from miko.agents.planner import Goal, risk_of, solve
from miko.worldmodels.features import equity_state_features, forward_targets
from miko.worldmodels.paths import BlockBootstrap, GarchT, StatePath
from miko.worldmodels.state_encoder import StateEncoder

warnings.filterwarnings("ignore")
START = "1995-01-01"
H = 21
N_PATHS = 2000  # check paths
N_OPT = 4000  # optimisation paths
WINDOW = 504
COST_BPS = 10
GOALS = {"cvar6": Goal("cvar95", 0.06, max_weight=0.2), "vol10": Goal("vol", 0.10, max_weight=0.2)}


def rebalance_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).max()
    return [d for d in me if d >= pd.Timestamp(START) and idx.get_loc(d) + 1 < len(idx)]


# ------------------------------------------------------------------------- classic strategies
def classic_weights(name: str, sr: pd.DataFrame) -> np.ndarray:
    from skfolio import RiskMeasure
    from skfolio.moments import LedoitWolf
    from skfolio.optimization import EqualWeighted, HierarchicalRiskParity, InverseVolatility, MeanRisk, ObjectiveFunction, RiskBudgeting
    from skfolio.prior import EmpiricalPrior

    if name == "1/N":
        m = EqualWeighted()
    elif name == "inverse volatility":
        m = InverseVolatility()
    elif name == "min variance (Ledoit-Wolf)":
        m = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE,
                     prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf()))
    elif name == "mean-variance max Sharpe (sample)":
        m = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE)
    elif name == "equal risk contribution":
        m = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)
    elif name == "HRP":
        m = HierarchicalRiskParity()
    elif name == "min CVaR (historical)":
        m = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.CVAR)
    else:
        raise ValueError(name)
    m.fit(sr)
    return np.asarray(m.weights_, dtype=float)


CLASSICS = ["1/N", "inverse volatility", "min variance (Ledoit-Wolf)", "mean-variance max Sharpe (sample)",
            "equal risk contribution", "HRP", "min CVaR (historical)"]


def hist_risk(w: np.ndarray, sr_window: pd.DataFrame, goal: Goal) -> float:
    """Risk of w from trailing history: overlapping 21-day returns (CVaR) or daily vol."""
    lr = np.log1p(sr_window.to_numpy())
    cs = np.vstack([np.zeros(lr.shape[1]), np.cumsum(lr, axis=0)])
    paths = np.stack([lr[t : t + H] for t in range(len(lr) - H + 1)])  # (n, H, k) overlapping windows
    if goal.measure == "vol":
        return float((sr_window.to_numpy() @ w).std() * np.sqrt(252))
    return risk_of(w, paths, goal.measure)


def scale(w, r, goal):
    return w * min(goal.limit / r, 1.0 / w.sum()) if r > 0 else w


# ------------------------------------------------------------------------------- run one
def run_strategy(args):
    name, step = args
    R = D.return_panel()
    stocks = D.stock_names()
    lr = R[stocks]
    sr = np.expm1(lr)
    dates = rebalance_dates(R.index)[::step]
    F = equity_state_features(R, D.price_panel(), stocks)
    T = forward_targets(R["SP500"], D.price_panel()["SP500"], 21)
    rows = []
    model, enc, year = None, None, None
    for d in dates:
        t = R.index.get_loc(d)
        nxt = dates[dates.index(d) + 1] if dates.index(d) + 1 < len(dates) else R.index[min(len(R) - 1, t + H)]
        t_end = R.index.get_loc(nxt) if step == 1 else min(len(R) - 1, t + H)
        hold = lr.iloc[t + 1 : t_end + 1]
        win = sr.iloc[t - WINDOW + 1 : t + 1]
        if name in CLASSICS:
            w = classic_weights(name, win)
        else:
            kind, goal_key = name.split("|")
            goal = GOALS[goal_key]
            if kind.startswith("scaled"):
                base = np.full(len(stocks), 1 / len(stocks)) if "1/N" in kind else classic_weights("min CVaR (historical)", win)
                base = np.minimum(base, goal.max_weight)
                w = scale(base, hist_risk(base, win, goal), goal)
            else:
                if year != d.year or model is None:
                    year = d.year
                    cut = R.index[R.index.searchsorted(pd.Timestamp(f"{d.year}-01-01")) - 1]
                    train = lr.loc[:cut]
                    if kind in ("planner (world model)", "1/N scaled by world-model risk"):
                        enc = StateEncoder().fit(F, cut, T)
                        model = StatePath(encoder=enc, features=F, kernel=False, vol_head=True, widen=0.0).fit(train)
                    elif kind == "1/N scaled by FHS risk (world model without state)":
                        model = StatePath(kernel=False, vol_head=False, widen=0.0).fit(train)
                    elif kind in ("planner + GARCH-t futures", "1/N scaled by GARCH-t risk"):
                        model = GarchT().fit(train)
                    elif kind == "1/N scaled by GJR-GARCH-t risk":
                        model = GarchT(asymmetric=True).fit(train)
                    else:
                        model = BlockBootstrap(10).fit(train)
                hist = lr.loc[:d]
                opt = model.sample(N_OPT, H, int(t), hist)
                chk = model.sample(N_PATHS, H, int(t) + 99991, hist)
                if kind.startswith("1/N scaled by"):
                    base = np.full(len(stocks), 1 / len(stocks))
                    w = scale(base, risk_of(base, chk, goal.measure), goal)
                else:
                    w0 = rows[-1]["drift_end"] if rows else None
                    try:
                        w = solve(opt, goal, w0, COST_BPS)
                    except Exception:
                        w = np.zeros(len(stocks))
                    rc = risk_of(w, chk, goal.measure) if w.sum() > 0 else 0.0
                    if rc > goal.limit:
                        w = w * goal.limit / rc
        # buy-and-hold over the month
        g = np.exp(np.cumsum(hold.to_numpy(), axis=0))
        v = 1 + (g - 1) @ w
        daily = np.diff(np.concatenate([[1.0], v])) / np.concatenate([[1.0], v[:-1]])
        drift = (w * g[-1]) / v[-1] if v[-1] > 0 else w
        rows.append({"date": d, "ret": float(v[-1] - 1), "invested": float(w.sum()), "weights": w, "drift_end": drift,
                     "daily": daily, "n_days": len(hold), "ret21": float((1 + (np.exp(lr.iloc[t + 1 : t + 1 + H].sum().to_numpy()) - 1) @ w)) - 1})
    # turnover from drifted to new weights
    for i in range(1, len(rows)):
        rows[i]["turnover"] = float(np.abs(rows[i]["weights"] - rows[i - 1]["drift_end"]).sum())
    rows[0]["turnover"] = float(np.abs(rows[0]["weights"]).sum())
    return name, rows


def jkm_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Jobson-Korkie test with Memmel (2003) correction: H0 equal Sharpe ratios. Returns (z, p)."""
    T = len(a)
    ma, mb, sa, sb = a.mean(), b.mean(), a.std(ddof=1), b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1 / T) * (2 * sa**2 * sb**2 - 2 * sa * sb * sa * sb * rho + 0.5 * ma**2 * sb**2 + 0.5 * mb**2 * sa**2
                       - (ma * mb / (sa * sb)) * sa * sb * sa * sb * rho**2)
    z = (sb * ma - sa * mb) / np.sqrt(theta)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def summarize(rows: list[dict], ref: np.ndarray | None, ref2: np.ndarray | None = None) -> dict:
    r = np.array([x["ret"] for x in rows])
    to = np.array([x["turnover"] for x in rows])
    net = r - to * COST_BPS * 1e-4
    wealth = np.cumprod(1 + r)
    mdd = (wealth / np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:] - 1).min()
    mu, sd = r.mean() * 12, r.std(ddof=1) * np.sqrt(12)
    q = np.quantile(r, 0.05)
    out = {"ann. return": mu, "ann. vol": sd, "Sharpe": mu / sd, "Sharpe net 10bps": net.mean() * 12 / (net.std(ddof=1) * np.sqrt(12)),
           "CEQ (gamma=1)": mu - 0.5 * sd**2, "max drawdown": mdd, "monthly turnover": to.mean(),
           "avg invested": np.mean([x["invested"] for x in rows]), "realized monthly CVaR95": -r[r <= q].mean()}
    if ref is not None:
        z, p = jkm_test(r, ref)
        out["Sharpe vs 1/N: z"] = z
        out["Sharpe vs 1/N: p"] = p
    if ref2 is not None and not np.array_equal(r, ref2):
        z, p = jkm_test(r, ref2)
        out["Sharpe vs 1/N sized by historical risk: p"] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="every 6th month only (smoke test)")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    step = 6 if args.fast else 1
    risk_names = [f"{k}|{g}" for g in GOALS for k in ("planner (world model)", "1/N scaled by world-model risk", "1/N scaled by FHS risk (world model without state)",
                                                       "1/N scaled by GARCH-t risk", "1/N scaled by GJR-GARCH-t risk", "scaled 1/N (historical risk)",
                                                       "scaled min CVaR (historical risk)", "planner + GARCH-t futures", "planner + block-bootstrap futures")]
    names = CLASSICS + risk_names
    with mp.Pool(args.procs) as pool:
        res = dict(pool.map(run_strategy, [(n, step) for n in names]))
    ref = np.array([x["ret"] for x in res["1/N"]])
    A = pd.DataFrame({n: summarize(res[n], ref if n != "1/N" else None) for n in CLASSICS}).T
    A.index.name = "fully invested (no risk goal)"
    text = [f"# Planner benchmark: 20 US large caps, monthly, {res['1/N'][0]['date'].date()} to {res['1/N'][-1]['date'].date()} ({len(ref)} months{', FAST' if args.fast else ''})", "",
            "## A. Fully invested classics (DeMiguel-Garlappi-Uppal style)", md_table(A, "{:.3f}"), ""]
    out = {"classics": A.to_dict(orient="index")}
    # crisis periods by *holding* month (the month after each rebalance date)
    crisis = [("2000-09", "2002-09"), ("2008-09", "2009-03"), ("2020-02", "2020-04"), ("2022-01", "2022-10")]
    for g, goal in GOALS.items():
        names_g = [n for n in risk_names if n.endswith("|" + g)]
        ref2 = np.array([x["ret"] for x in res[f"scaled 1/N (historical risk)|{g}"]])
        B = pd.DataFrame({n.split("|")[0]: summarize(res[n], ref, ref2) for n in names_g}).T
        # goal compliance: realized 21-day risk vs the limit
        comp = {}
        for n in names_g:
            rr = pd.Series([x["ret21"] for x in res[n]], index=[x["date"] for x in res[n]])
            held = (rr.index + pd.offsets.MonthBegin(1)).to_period("M")
            mask = np.zeros(len(rr), bool)
            for a, b in crisis:
                mask |= (held >= pd.Period(a, "M")) & (held <= pd.Period(b, "M"))
            def realized(x):
                if goal.measure == "vol":
                    return float(x.std(ddof=1) * np.sqrt(252 / H))
                q = np.quantile(x, 0.05)
                return float(-x[x <= q].mean())
            # loss level the goal implies should be exceeded in at most ~5% of months
            thr = goal.limit if goal.measure == "cvar95" else 1.645 * goal.limit * np.sqrt(H / 252)
            comp[n.split("|")[0]] = {f"realized {goal.measure} (all months)": realized(rr.to_numpy()),
                                     f"realized {goal.measure} (crisis months)": realized(rr[mask].to_numpy()),
                                     "goal": goal.limit,
                                     f"share of months losing more than {thr:.1%} (goal implies <= 5%)": float((rr < -thr).mean())}
        C = pd.DataFrame(comp).T
        B.index.name = f"risk goal: {goal.measure} <= {goal.limit:.0%}, max 20% per name"
        C.index.name = f"goal compliance (21-day returns after each rebalance; {int(mask.sum())} crisis months)"
        text += [f"## B. Risk goal {goal.measure} <= {goal.limit:.0%}", md_table(B, "{:.3f}"), "", md_table(C, "{:.3f}"), ""]
        out[g] = {"performance": B.to_dict(orient="index"), "compliance": C.to_dict(orient="index")}
    text += ["Crisis months (holding months): 2000-09 to 2002-09, 2008-09 to 2009-03, 2020-02 to 2020-04, 2022-01 to 2022-10.",
             "Realized CVaR95 over crisis months averages only the worst ~5% of them (2-3 months), so read it as indicative.",
             "Sharpe ratios use raw returns with cash at 0% (no risk-free rate); the Jobson-Korkie-Memmel test compares Sharpe ratios.",
             "Survivorship caveat: all 20 stocks survived to 2022; absolute returns are flattered, comparisons are fair."]
    suffix = "_fast" if args.fast else ""
    (RESULTS / f"planner_portfolio{suffix}.md").write_text("\n".join(text) + "\n")
    save_json(out, f"planner_portfolio{suffix}.json")
    monthly = pd.DataFrame({n: pd.Series([x["ret"] for x in res[n]], index=[x["date"] for x in res[n]]) for n in names})
    monthly.to_csv(RESULTS / f"planner_monthly_returns{suffix}.csv")
    print("\n".join(text))


if __name__ == "__main__":
    main()
