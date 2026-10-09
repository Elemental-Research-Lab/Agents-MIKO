"""Planner agent: finds weights that meet a risk goal; trains policies (brief, section 04).

Weights. The Planner simulates futures for the universe from today's state (world model),
then solves a linear program over those futures: invest as much as the risk goal allows, in
the mix that carries the least risk per unit invested, with position caps and turnover costs.
Risk goals: CVaR 95% of the horizon loss (Rockafellar-Uryasev), annualized volatility, or
CDaR 95% (conditional drawdown at risk; Chekhlov, Uryasev and Zabarankin). The plan is then
checked on a second, independent set of futures the optimizer never saw; if it breaches the
goal there, it is scaled back until it does not. The expected-return prior is flat: world
models are strongest for risk, not return forecasts (brief, section 11).

Policies. `train_policy` learns an exposure rule (volatility targeting with a drawdown brake)
inside the simulated market, checks that it holds on other models' futures, and confirms it on
real history it never saw, next to buy-and-hold and a plain volatility target.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

from ..core.guards import InfeasibleGoalError, MissingInputError, is_blank, require_positive, to_timestamp
from ..core.llm import LLMClient, propose_goal
from ..core.result import LIMITS_SIM, Number, Provenance, Result, params_hash
from .context import AgentContext

MEASURES = ("cvar95", "vol", "cdar95")


@dataclass
class Goal:
    measure: str
    limit: float  # fraction, e.g. 0.06 for 6%
    max_weight: float = 1.0
    long_only: bool = True
    min_invested: float = 0.0

    def __post_init__(self):
        if self.measure not in MEASURES:
            raise MissingInputError(f"risk measure must be one of {MEASURES}")
        if is_blank(self.limit):
            raise MissingInputError("risk limit is blank")
        self.limit = float(self.limit)
        if not self.limit > 0:
            raise InfeasibleGoalError(f"risk limit must be positive, got {self.limit}")
        if not (0 < self.max_weight <= 1):
            raise MissingInputError("max_weight must be in (0, 1]")
        if not self.long_only:
            raise MissingInputError("v1 Planner is long-only")
        if not (0 <= self.min_invested <= 1):
            raise MissingInputError("min_invested must be in [0, 1]")

    @classmethod
    def from_spec(cls, spec: dict) -> "Goal":
        return cls(measure=spec["risk"]["measure"], limit=spec["risk"]["limit_pct"] / 100.0,
                   max_weight=(spec.get("max_weight_pct") or 100.0) / 100.0, long_only=spec.get("long_only", True))


# ----------------------------------------------------------------------------- risk on paths
def horizon_returns(logret: np.ndarray) -> np.ndarray:
    return np.exp(logret.sum(axis=1)) - 1.0


def risk_of(w: np.ndarray, logret: np.ndarray, measure: str, alpha: float = 0.95) -> float:
    H = logret.shape[1]
    if measure == "cdar95":
        g = np.exp(np.cumsum(logret, axis=1)) - 1.0  # (n, H, k) gains per unit invested
        v = g @ w
        v = np.concatenate([np.zeros((len(v), 1)), v], axis=1)
        mdd = (np.maximum.accumulate(v, axis=1) - v).max(axis=1)
        q = np.quantile(mdd, alpha)
        return float(mdd[mdd >= q].mean())
    r = horizon_returns(logret) @ w
    if measure == "vol":
        return float(r.std() * np.sqrt(252.0 / H))
    q = np.quantile(r, 1 - alpha)
    return float(-r[r <= q].mean())


def solve(logret: np.ndarray, goal: Goal, w0: np.ndarray | None = None, cost_bps: float = 10.0, alpha: float = 0.95,
          l2: float = 0.25) -> np.ndarray:
    """Maximize invested share - l2*||w||^2 - turnover cost, subject to the risk goal on the paths.
    The small L2 term makes the flat-prior problem well posed (equal weights when risk is slack)
    and damps the scenario noise that otherwise flips the plan between corner portfolios."""
    import cvxpy as cp

    n, H, k = logret.shape
    w = cp.Variable(k)
    cons = [w >= 0, w <= goal.max_weight, cp.sum(w) <= 1, cp.sum(w) >= goal.min_invested]
    R = horizon_returns(logret)
    if goal.measure == "cvar95":
        z = cp.Variable()
        u = cp.Variable(n, nonneg=True)
        cons += [u >= -R @ w - z]
        risk = z + cp.sum(u) / ((1 - alpha) * n)
    elif goal.measure == "vol":
        C = np.cov(R.T) * (252.0 / H)
        C = 0.5 * (C + C.T) + 1e-10 * np.eye(k)
        risk = cp.norm(np.linalg.cholesky(C).T @ w, 2)
    else:  # cdar95
        G = np.exp(np.cumsum(logret, axis=1)) - 1.0
        V = cp.hstack([G[:, t, :] @ w for t in range(H)])  # stacked by time: block t has n entries
        m = cp.Variable(n * H)
        mdd = cp.Variable(n)
        z = cp.Variable()
        u = cp.Variable(n, nonneg=True)
        for t in range(H):
            blk = slice(t * n, (t + 1) * n)
            cons += [m[blk] >= V[blk], m[blk] >= (m[(t - 1) * n : t * n] if t else 0), mdd >= m[blk] - V[blk]]
        cons += [u >= mdd - z]
        risk = z + cp.sum(u) / ((1 - alpha) * n)
    cons += [risk <= goal.limit]
    obj = cp.sum(w) - l2 * cp.sum_squares(w) - 1e-3 * risk
    if w0 is not None:
        obj -= (cost_bps * 1e-4) * cp.norm(w - w0, 1)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver="CLARABEL")
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise InfeasibleGoalError(f"no long-only plan meets {goal.measure} <= {goal.limit:.2%} with the given constraints (solver: {prob.status})")
    out = np.clip(w.value, 0, None)
    out[out < 1e-6] = 0.0
    return out


def scale_to_limit(w: np.ndarray, logret: np.ndarray, goal: Goal) -> np.ndarray:
    """Scale a portfolio (cash is the rest) so its risk on these paths equals the limit, capped at fully invested."""
    r = risk_of(w, logret, goal.measure)
    if r <= 0:
        return w
    s = min(goal.limit / r, 1.0 / max(w.sum(), 1e-12))
    return w * s


class Planner:
    name = "Planner"

    def __init__(self, ctx: AgentContext | None = None, llm: LLMClient | None = None):
        self.ctx = ctx or AgentContext()
        self.llm = llm or LLMClient()

    def _paths(self, as_of, universe, horizon, n, seed):
        m = self.ctx.path_model(as_of, universe, "statepath")
        return m, m.sample(n, horizon, seed, self.ctx.returns_upto(as_of, universe))

    def plan(self, as_of, universe: list[str], goal: Goal, horizon: int = 21, n_paths: int = 2000, seed: int = 0,
             current: dict[str, float] | None = None, cost_bps: float = 10.0, mode: str = "optimize",
             base: dict[str, float] | None = None) -> dict:
        """Core planning without the result envelope (used by the agent and the benchmark).

        mode="optimize": LP over the futures (composition and size).
        mode="scale": keep a base composition (default 1/N) and size it to the goal with the
        world model's futures. In the 1995-2022 benchmark this did as well as optimizing, with a
        quarter of the turnover (bench/planner_portfolio.py), in line with DeMiguel et al. (2009)."""
        if mode not in ("optimize", "scale"):
            raise MissingInputError("mode must be 'optimize' or 'scale'")
        n_opt = min(n_paths, 600) if goal.measure == "cdar95" else max(n_paths, 4000)
        m, opt = self._paths(as_of, universe, horizon, n_opt, seed)
        _, chk = self._paths(as_of, universe, horizon, n_paths, seed + 10_000)
        w0 = np.array([float((current or {}).get(a, 0.0)) for a in universe])
        if mode == "scale":
            b = np.array([float((base or {}).get(a, 0.0)) for a in universe]) if base else np.full(len(universe), 1.0 / len(universe))
            if (b < 0).any() or b.sum() <= 0:
                raise MissingInputError("base composition must be long-only with positive total weight")
            b = np.minimum(b / b.sum(), goal.max_weight)
            w = scale_to_limit(b, opt, goal)
        else:
            w = solve(opt, goal, w0 if current else None, cost_bps)
        risk_in = risk_of(w, opt, goal.measure)
        risk_chk = risk_of(w, chk, goal.measure)
        scaled = False
        if risk_chk > goal.limit * 1.0001:
            w = w * (goal.limit / risk_chk)
            scaled = True
            risk_chk = risk_of(w, chk, goal.measure)
        if w.sum() < goal.min_invested - 1e-9:
            raise InfeasibleGoalError(f"meeting {goal.measure} <= {goal.limit:.2%} on unseen futures needs less than the minimum {goal.min_invested:.0%} invested")
        return {"weights": w, "risk_opt": risk_in, "risk_check": risk_chk, "scaled_back": scaled, "model": m, "check_paths": chk,
                "turnover": float(np.abs(w - w0).sum())}

    def run(self, as_of, universe: list[str], goal: Goal | dict | None = None, goal_text: str | None = None,
            horizon: int = 21, n_paths: int = 2000, seed: int = 0, current: dict[str, float] | None = None,
            cost_bps: float = 10.0, mode: str = "optimize", base: dict[str, float] | None = None,
            lineage: str | None = None) -> Result:
        ctx = self.ctx
        as_of = to_timestamp(as_of)
        if not universe or len(set(universe)) != len(universe):
            raise MissingInputError("universe is empty or has duplicates")
        ctx.returns_upto(as_of, universe)
        if goal_text is not None:
            goal = Goal.from_spec(propose_goal(goal_text, self.llm))
        elif isinstance(goal, dict):
            goal = Goal(**goal)
        if not isinstance(goal, Goal):
            raise MissingInputError("no goal given")
        if current:
            for a, v in current.items():
                if a not in universe or is_blank(v):
                    raise MissingInputError(f"current weight for '{a}' is blank or outside the universe")
        out = self.plan(as_of, universe, goal, horizon, n_paths, seed, current, cost_bps, mode, base)
        w = out["weights"]
        chk = out["check_paths"]
        R = horizon_returns(chk)
        port = R @ w

        # baselines on the same unseen futures, scaled to the same goal
        k = len(universe)
        vol = np.array([ctx.returns_upto(as_of, [a])[a].iloc[-63:].std() for a in universe])
        bases = {"1/N": np.full(k, 1.0 / k), "inverse volatility": (1 / vol) / (1 / vol).sum()}
        rows = [{"plan": f"Planner (world model, {mode})", "invested": round(float(w.sum()), 4), "risk": round(out["risk_check"], 5),
                 "median_return": round(float(np.median(port)), 5)}]
        for name, b in bases.items():
            b = np.minimum(b, goal.max_weight)
            bs = scale_to_limit(b, chk, goal)
            rows.append({"plan": f"{name} scaled to the goal", "invested": round(float(bs.sum()), 4),
                         "risk": round(risk_of(bs, chk, goal.measure), 5), "median_return": round(float(np.median(R @ bs)), 5)})
        label = {"cvar95": f"CVaR 95% over {horizon} days", "vol": "annualized volatility", "cdar95": f"CDaR 95% (drawdown) over {horizon} days"}[goal.measure]
        numbers = {
            "invested": Number(float(w.sum()), "SIM", "Share of capital invested (rest in cash)", "pct_abs"),
            "risk_check": Number(out["risk_check"], "SIM", f"{label} on unseen futures", "pct_abs"),
            "risk_limit": Number(goal.limit, "CALC", f"Goal: {label} at most", "pct_abs"),
            "median_return": Number(float(np.median(port)), "SIM", f"Median {horizon}-day return on unseen futures", "pct"),
            "turnover": Number(out["turnover"], "CALC", "Turnover vs current weights", "pct_abs"),
            "cost_bps": Number(out["turnover"] * cost_bps, "CALC", "Estimated trading cost", "bps"),
        }
        warnings = []
        if out["scaled_back"]:
            warnings.append("The optimized plan breached the goal on unseen futures and was scaled back until it met it.")
        res = Result(
            agent=self.name, as_of=str(as_of.date()), numbers=numbers,
            provenance=Provenance([ctx.ref(out["model"]), ctx.ref(ctx.encoder(as_of))], seed, ctx.snapshot(),
                                  params_hash({"as_of": str(as_of.date()), "u": universe, "goal": asdict(goal), "h": horizon, "n": n_paths, "seed": seed, "current": current, "mode": mode, "base": base})),
            readout=(f"Invest {w.sum():.0%} of capital across {int((w > 0).sum())} names; on {len(chk)} futures the optimizer never saw, "
                     f"{label} is {out['risk_check']:.2%} against a limit of {goal.limit:.2%}."),
            verdict="meets goal on unseen futures",
            tables={"weights": [{"asset": a, "weight": round(float(x), 4), "badge": "SIM"} for a, x in zip(universe, w) if x > 0] +
                               [{"asset": "cash", "weight": round(float(1 - w.sum()), 4), "badge": "CALC"}],
                    "compared_with_baselines": rows, "goal": [asdict(goal)]},
            warnings=warnings,
            limits=list(LIMITS_SIM) + ["The expected-return prior is flat: the plan spends the risk budget, it does not pick winners."],
        )
        res.trial_id, res.trial_count = ctx.ledger.record(self.name, lineage or "plan:" + ",".join(sorted(universe)), res.provenance.params_hash, res.digest())
        return res

    # ------------------------------------------------------------------------- policies
    @staticmethod
    def run_policy(logret: np.ndarray, theta: dict, sigma0: float, cost_bps: float = 5.0) -> dict:
        """Exposure policy on paths of daily log returns (n, T).
        exposure_t = clip(k * target_vol / ewma_vol_t, 0, 1) * (1 - brake * 1[drawdown_t <= -cut])."""
        n, T = logret.shape
        lam = 0.94
        var = np.full(n, sigma0**2)
        logv = np.zeros(n)
        peak = np.zeros(n)
        expo_prev = np.zeros(n)
        port = np.empty((n, T))
        turnover = np.zeros(n)
        for t in range(T):
            dd = logv - peak
            e = np.clip(theta["k"] * theta["target_vol"] / np.sqrt(252 * var), 0, 1.0)
            e = e * (1 - theta["brake"] * (dd <= np.log(1 - theta["cut"])))
            turnover += np.abs(e - expo_prev)
            r = logret[:, t]
            simple = np.expm1(r)
            port[:, t] = e * simple - cost_bps * 1e-4 * np.abs(e - expo_prev)
            expo_prev = e
            var = lam * var + (1 - lam) * r**2
            logv = logv + r
            peak = np.maximum(peak, logv)
        wealth = np.cumprod(1 + port, axis=1)
        w0 = np.concatenate([np.ones((n, 1)), wealth], axis=1)
        mdd = (w0 / np.maximum.accumulate(w0, axis=1) - 1).min(axis=1)
        flat = port.ravel()
        return {"sharpe": float(flat.mean() / flat.std() * np.sqrt(252)) if flat.std() > 0 else 0.0,
                "ann_return": float(np.mean(wealth[:, -1] ** (252.0 / T) - 1)),
                "mdd_cvar95": float(-np.mean(np.sort(mdd)[: max(1, int(0.05 * n))])),
                "mdd_median": float(-np.median(mdd)), "turnover_per_year": float(np.mean(turnover) * 252.0 / T),
                "daily": port}

    def train_policy(self, as_of, asset: str = "SP500", target_vol: float = 0.12, dd_limit: float = 0.25,
                     years: int = 1, n_paths: int = 1000, seed: int = 0, cost_bps: float = 5.0,
                     confirm_until=None, lineage: str | None = None) -> Result:
        ctx = self.ctx
        as_of = to_timestamp(as_of)
        require_positive(target_vol, "target_vol")
        require_positive(dd_limit, "dd_limit")
        hist = ctx.returns_upto(as_of, [asset])
        T = 252 * years
        sigma0 = float(hist[asset].iloc[-63:].std())
        from ..worldmodels.paths import BlockBootstrap, GarchT, StatePath

        enc = ctx.encoder(as_of)
        wm = StatePath(encoder=enc, features=ctx.features, kernel=False, vol_head=False, widen=0.0).fit(hist)
        sims = {"statepath (world model)": wm.sample(n_paths, T, seed, hist)[..., 0]}
        grid = [{"k": k, "brake": b, "cut": c, "target_vol": target_vol}
                for k, b, c in product((0.6, 0.8, 1.0, 1.2, 1.4), (0.0, 0.25, 0.5, 0.75), (0.08, 0.12, 0.2))
                if not (b == 0.0 and c != 0.08)]
        scores = []
        for th in grid:
            s = self.run_policy(sims["statepath (world model)"], th, sigma0, cost_bps)
            scores.append((th, s))
        feasible = [x for x in scores if x[1]["mdd_cvar95"] <= dd_limit]
        best_th, best = max(feasible, key=lambda x: x[1]["sharpe"]) if feasible else min(scores, key=lambda x: x[1]["mdd_cvar95"])
        # does it hold on other models?
        other = {}
        for name, m in (("garch-t", GarchT()), ("block-bootstrap", BlockBootstrap(10))):
            p = m.fit(hist).sample(n_paths, T, seed + 1, hist)[..., 0]
            s = self.run_policy(p, best_th, sigma0, cost_bps)
            other[name] = {k: round(v, 4) for k, v in s.items() if k != "daily"}
        # confirm on real history after the run date (only possible for past run dates)
        hist_rows = []
        if confirm_until is not None:
            fut = ctx.returns[asset].loc[as_of:to_timestamp(confirm_until)].iloc[1:]
            if len(fut) < 21:
                raise MissingInputError("need at least 21 days after the run date to confirm on history")
            x = fut.to_numpy()[None, :]
            for lab, th in (("trained policy", best_th),
                            ("vol target, no brake", {"k": 1.0, "brake": 0.0, "cut": 0.08, "target_vol": target_vol}),
                            ("buy and hold", {"k": 1e9, "brake": 0.0, "cut": 0.08, "target_vol": target_vol})):
                s = self.run_policy(x, th, sigma0, cost_bps)
                hist_rows.append({"policy": lab, "sharpe": round(s["sharpe"], 3), "ann_return": round(s["ann_return"], 4),
                                  "max_drawdown": round(s["mdd_cvar95"], 4), "turnover_per_year": round(s["turnover_per_year"], 2), "badge": "DATA"})
        numbers = {
            "sim_sharpe": Number(best["sharpe"], "SIM", "Sharpe in simulated futures", "level"),
            "sim_mdd_cvar95": Number(best["mdd_cvar95"], "SIM", "Max drawdown, worst 5% of simulated years", "pct_abs"),
            "k": Number(best_th["k"], "CALC", "Policy: vol-target multiplier", "level"),
            "brake": Number(best_th["brake"], "CALC", "Policy: exposure cut in drawdown", "pct_abs"),
            "cut": Number(best_th["cut"], "CALC", "Policy: drawdown that triggers the brake", "pct_abs"),
            "n_candidates": Number(len(grid), "CALC", "Policies tried (all count as trials)", "count"),
        }
        res = Result(
            agent=self.name, as_of=str(as_of.date()), numbers=numbers,
            provenance=Provenance([ctx.ref(wm), ctx.ref(enc)], seed, ctx.snapshot(),
                                  params_hash({"as_of": str(as_of.date()), "asset": asset, "tv": target_vol, "dd": dd_limit, "n": n_paths, "seed": seed})),
            readout=f"Trained an exposure policy for {asset} in {n_paths} simulated years; " + (
                "it meets the drawdown limit in simulation." if feasible else "no candidate met the drawdown limit; the least risky one is shown."),
            verdict="trained in simulation" + ("; confirmed on history" if hist_rows else "; not yet confirmed on history"),
            tables={"holds_on_other_models": [{"model": k, **v} for k, v in other.items()], "confirm_on_history": hist_rows},
            warnings=[] if feasible else ["No candidate policy met the drawdown limit in simulation."],
            limits=["A policy trained in a world model also learns that model's flaws; it must hold on other models and on real history.",
                    "Simulated results cannot promote a strategy on their own."],
        )
        res.trial_id, res.trial_count = ctx.ledger.record(self.name, lineage or f"policy:{asset}", res.provenance.params_hash, res.digest(), {"n_candidates": len(grid)})
        return res
