"""State Reader agent: reads regime, closest past periods and novelty (brief, section 04).

World-model agent: its outputs are numbers from the State Encoder (SIM), retrieved history
(DATA) and deterministic summaries (CALC). It never writes prose beyond a fixed template.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.guards import to_timestamp
from ..core.result import Number, Provenance, Result, params_hash
from .context import AgentContext

NOVEL = 0.95


class StateReader:
    name = "State Reader"

    def __init__(self, ctx: AgentContext | None = None):
        self.ctx = ctx or AgentContext()

    def run(self, as_of, k_analogs: int = 10, lineage: str = "market-state") -> Result:
        ctx = self.ctx
        as_of = to_timestamp(as_of)
        ctx.returns_upto(as_of)  # refuses non-trading days and dates outside the data
        enc = ctx.encoder(as_of)
        F = ctx.features.loc[:as_of]
        reg_hist = enc.filtered_regime_probs(F)
        probs = reg_hist.iloc[-1]
        regime = str(probs.idxmax())
        nov, nov_pct = enc.novelty(F)
        risk = enc.forward_risk(F)
        an = enc.analogs(F, k_analogs)
        T = ctx.targets
        P = ctx.prices["SP500"]
        rows = []
        for _, a in an.iterrows():
            d = a["date"]
            rows.append({
                "date": str(d.date()),
                "distance": round(float(a["distance"]), 3),
                "regime_then": str(reg_hist.loc[d].idxmax()),
                "sp500_next_21d_%": round(float(100 * (np.exp(T.loc[d, "fwd_ret21"]) - 1)), 2),
                "max_drawdown_next_21d_%": round(float(100 * (np.exp(T.loc[d, "fwd_mdd21"]) - 1)), 2),
                "vol_next_21d_%": round(float(100 * np.exp(T.loc[d, "fwd_logrv21"])), 1),
                "sp500_next_63d_%": round(float(100 * (P.shift(-63).loc[d] / P.loc[d] - 1)), 2),
                "badge": "DATA",
            })
        dd5 = sum(1 for r in rows if r["max_drawdown_next_21d_%"] <= -5)
        f_now = F.iloc[-1]
        numbers = {
            "p_regime": Number(float(probs.max()), "SIM", f"P({regime} regime)", "ratio"),
            "novelty_pct": Number(nov_pct, "SIM", "Novelty percentile vs past days", "ratio"),
            "fwd_vol21": Number(risk["fwd_vol21"], "SIM", "Next-21-day volatility (annualized)", "pct_abs"),
            "p_drawdown": Number(risk["p_drawdown"], "SIM", f"P(S&P 500 drawdown worse than {enc.drawdown_threshold:.0%} in 21 days)", "ratio"),
            "vix": Number(float(np.exp(f_now["vix"])), "DATA", "VIX close", "level"),
            "rv21": Number(float(np.exp(f_now["rv21"])), "CALC", "Realized volatility, last 21 days (annualized)", "pct_abs"),
            "dd252": Number(float(np.exp(f_now["dd252"]) - 1), "CALC", "S&P 500 drawdown from 1-year high", "pct"),
            "corr63": Number(float(f_now["corr63"]), "CALC", "Average stock correlation, last 63 days", "ratio"),
            "analog_dd5_count": Number(dd5, "CALC", f"Closest past periods followed by a 5%+ drawdown (of {len(rows)})", "count"),
            "analog_median_next21": Number(float(np.median([r["sp500_next_21d_%"] for r in rows])) / 100, "CALC", "Median S&P 500 move over the next 21 days after the closest periods", "pct"),
        }
        readout = (f"The market reads as {regime} (probability {probs.max():.0%}). Today is more unusual than "
                   f"{nov_pct:.0%} of past days. In {dd5} of {len(rows)} closest past periods the S&P 500 had a drawdown "
                   "of 5% or more within the next 21 trading days.")
        warnings = []
        if nov_pct >= NOVEL:
            warnings.append(f"Today's state is more unusual than {nov_pct:.0%} of history: closest periods are far away and every band should be read wide.")
        res = Result(
            agent=self.name,
            as_of=str(as_of.date()),
            numbers=numbers,
            provenance=Provenance([ctx.ref(enc)], None, ctx.snapshot(), params_hash({"as_of": str(as_of.date()), "k": k_analogs})),
            readout=readout,
            verdict=f"{regime}",
            tables={"regime_probabilities": [{"regime": k, "probability": round(float(v), 4), "badge": "SIM"} for k, v in probs.items()],
                    "closest_past_periods": rows},
            warnings=warnings,
            limits=["Regimes are fitted labels, not ground truth.", "What followed a past period is history, not a forecast."],
        )
        res.trial_id, res.trial_count = ctx.ledger.record(self.name, lineage, res.provenance.params_hash, res.digest())
        return res
