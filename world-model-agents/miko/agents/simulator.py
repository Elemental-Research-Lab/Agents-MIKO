"""Simulator agent: runs futures with your book attached (brief, sections 03-04).

Generates many futures from today's market state with the world model, applies your own
trading costs and estimated impact, and reports the distribution of outcomes: medians, tail
losses, drawdowns and a count-based read-out. Baselines (GARCH-t, block bootstrap, random walk)
always run alongside on the same book so you can see whether the world model adds anything,
and model disagreement is reported separately from the normal spread of outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.guards import MissingInputError, is_blank, require_positive, to_timestamp
from ..core.result import LIMITS_SIM, Number, Provenance, Result, count_readout, params_hash
from .context import AgentContext

MAX_GROSS = 3.0


def validate_book(book: dict[str, float] | None, known: list[str]) -> dict[str, float]:
    if not book:
        raise MissingInputError("book is empty")
    out = {}
    for a, w in book.items():
        if a not in known:
            raise MissingInputError(f"unknown asset '{a}' in book")
        if is_blank(w):
            raise MissingInputError(f"weight for '{a}' is blank; refusing to treat it as zero")
        out[a] = float(w)
    gross = sum(abs(w) for w in out.values())
    if gross > MAX_GROSS:
        raise MissingInputError(f"gross exposure {gross:.2f}x exceeds the {MAX_GROSS}x cap")
    return out


def book_paths(logret: np.ndarray, weights: np.ndarray, cost_frac: float = 0.0) -> np.ndarray:
    """Buy-and-hold portfolio value paths (n, H+1) starting at 1, cash earning zero."""
    g = np.exp(np.cumsum(logret, axis=1))  # (n, H, k) price relatives
    v = 1.0 - cost_frac + ((g - 1.0) * weights).sum(axis=2)
    return np.concatenate([np.full((len(v), 1), 1.0 - cost_frac), v], axis=1)


def path_stats(v: np.ndarray, loss_threshold: float) -> dict[str, float]:
    r = v[:, -1] - 1.0
    q01, q05 = np.quantile(r, [0.01, 0.05])
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(v), 1)), v], axis=1), axis=1)[:, 1:]
    mdd = (v / peak - 1.0).min(axis=1)
    return {
        "median": float(np.median(r)), "mean": float(r.mean()),
        "q05": float(q05), "q95": float(np.quantile(r, 0.95)),
        "var95": float(-q05), "cvar95": float(-r[r <= q05].mean()),
        "var99": float(-q01), "cvar99": float(-r[r <= q01].mean()),
        "p_loss": float((r <= -loss_threshold).mean()), "n_loss": int((r <= -loss_threshold).sum()),
        "mdd_median": float(np.median(mdd)), "mdd_q95": float(np.quantile(mdd, 0.05)),
    }


class Simulator:
    name = "Simulator"

    def __init__(self, ctx: AgentContext | None = None):
        self.ctx = ctx or AgentContext()

    def simulate(self, as_of, assets: list[str], horizon: int, n_paths: int, seed: int, kind: str = "statepath") -> np.ndarray:
        """Raw world-model paths (n, H, k) of daily log returns for the assets."""
        m = self.ctx.path_model(as_of, assets, kind)
        return m.sample(n_paths, horizon, seed, self.ctx.returns_upto(as_of, assets))

    def run(self, as_of, book: dict[str, float], horizon: int = 21, n_paths: int = 2000, seed: int = 0,
            capital: float = 1_000_000.0, orders: dict[str, float] | None = None, adv: dict[str, float] | None = None,
            loss_threshold: float = 0.05, baselines: tuple[str, ...] = ("garch-t", "block-bootstrap", "random-walk"),
            lineage: str | None = None) -> Result:
        ctx = self.ctx
        as_of = to_timestamp(as_of)
        if not (1 <= int(horizon) <= 252):
            raise MissingInputError("horizon must be 1-252 trading days")
        if not (100 <= int(n_paths) <= 200_000):
            raise MissingInputError("n_paths must be 100-200,000")
        require_positive(capital, "capital")
        require_positive(loss_threshold, "loss_threshold")
        hist_all = ctx.returns_upto(as_of)
        w = validate_book(book, list(hist_all.columns))
        orders = orders or {}
        for a, q in orders.items():
            if a not in hist_all.columns:
                raise MissingInputError(f"unknown asset '{a}' in orders")
            if is_blank(q):
                raise MissingInputError(f"order size for '{a}' is blank")
            w[a] = w.get(a, 0.0) + float(q) / capital
        validate_book(w, list(hist_all.columns))
        assets = list(w)
        weights = np.array([w[a] for a in assets])
        hist = hist_all[assets]

        # costs of the orders (estimated, not reactive)
        daily_vol = {a: float(hist[a].iloc[-63:].std()) for a in assets}
        costs = ctx.impact.trade_costs({a: float(q) for a, q in orders.items()}, adv, daily_vol) if orders else {}
        cost_frac = sum(costs.values()) / capital

        wm = ctx.path_model(as_of, assets, "statepath")
        paths = wm.sample(n_paths, horizon, seed, hist)
        v = book_paths(paths, weights, cost_frac)
        st = path_stats(v, loss_threshold)

        base_rows = [{"model": f"{wm.name} (world model)", **{k: round(st[k], 5) for k in ("median", "var95", "cvar95", "p_loss", "mdd_q95")}}]
        refs = [ctx.ref(wm), ctx.ref(ctx.encoder(as_of))]
        var95s = [st["var95"]]
        for b in baselines:
            bm = ctx.path_model(as_of, assets, b)
            bs = path_stats(book_paths(bm.sample(n_paths, horizon, seed + 1, hist), weights, cost_frac), loss_threshold)
            base_rows.append({"model": b, **{k: round(bs[k], 5) for k in ("median", "var95", "cvar95", "p_loss", "mdd_q95")}})
            refs.append(ctx.ref(bm))
            var95s.append(bs["var95"])

        fan = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        dates = pd.bdate_range(as_of, periods=horizon + 1)
        fan_rows = [{"day": i, "date_approx": str(d.date()), **{f"q{q}": round(float(fan[j, i]), 5) for j, q in enumerate((5, 25, 50, 75, 95))}}
                    for i, d in enumerate(dates)]

        enc = ctx.encoder(as_of)
        _, nov_pct = enc.novelty(ctx.features.loc[:as_of])
        warnings = []
        if nov_pct >= 0.95:
            warnings.append(f"Today's market state is more unusual than {nov_pct:.0%} of history; read the bands as wider than shown.")

        numbers = {
            "n_paths": Number(n_paths, "SIM", "Simulated futures", "count"),
            "horizon": Number(horizon, "SIM", "Horizon", "days"),
            "median_return": Number(st["median"], "SIM", "Median return", "pct"),
            "q05": Number(st["q05"], "SIM", "5th percentile return", "pct"),
            "q95": Number(st["q95"], "SIM", "95th percentile return", "pct"),
            "var95": Number(st["var95"], "SIM", "VaR 95% (loss)", "pct_abs"),
            "cvar95": Number(st["cvar95"], "SIM", "CVaR 95% (expected loss in worst 5%)", "pct_abs"),
            "var99": Number(st["var99"], "SIM", "VaR 99% (loss)", "pct_abs"),
            "mdd_median": Number(st["mdd_median"], "SIM", "Median max drawdown", "pct"),
            "mdd_q95": Number(st["mdd_q95"], "SIM", "Max drawdown, worst 5% of futures", "pct"),
            "n_loss": Number(st["n_loss"], "SIM", f"Futures losing more than {loss_threshold:.0%}", "count"),
            "costs_bps": Number(1e4 * cost_frac, "CALC", "Trading cost incl. estimated impact", "bps"),
            "model_disagreement_var95": Number(max(var95s) - min(var95s), "SIM", "Model disagreement: range of VaR 95% across models", "pct_abs"),
            "novelty_pct": Number(nov_pct, "SIM", "Market-state novelty percentile", "ratio"),
        }
        res = Result(
            agent=self.name,
            as_of=str(as_of.date()),
            numbers=numbers,
            provenance=Provenance(refs, seed, ctx.snapshot(),
                                  params_hash({"as_of": str(as_of.date()), "book": w, "h": horizon, "n": n_paths, "seed": seed, "orders": orders})),
            readout=count_readout(st["n_loss"], n_paths, f"the book loses more than {loss_threshold:.0%} within {horizon} trading days"),
            tables={"models_side_by_side": base_rows, "fan": fan_rows,
                    "costs": [{"asset": a, "cost": round(float(c), 2), "badge": "CALC"} for a, c in costs.items()]},
            warnings=warnings,
            limits=list(LIMITS_SIM),
        )
        res.trial_id, res.trial_count = ctx.ledger.record(self.name, lineage or "book:" + ",".join(sorted(w)), res.provenance.params_hash, res.digest())
        return res
