"""Position & Impact layer: what do my trades cost? (brief, section 02)

Cost of an order, as a fraction of its notional:

    commission + half spread + Y * daily_vol * sqrt(order_notional / ADV_notional)

The last term is the square-root impact law (Almgren et al. 2005; Toth et al. 2011), with
Y ~ 0.5-1 in the empirical literature. In v1 impact is *estimated*, not reactive: our orders
do not move the simulated prices (brief, section 11), and every cost is labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.guards import MissingInputError, require_positive


@dataclass
class ImpactModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    impact_y: float = 0.7
    name: str = "position-impact"
    version: str = "v1"

    def order_cost_frac(self, notional: float, adv_notional: float, daily_vol: float) -> float:
        notional = abs(float(notional))
        if notional == 0:
            return 0.0
        adv = require_positive(adv_notional, "ADV (average daily traded value)")
        vol = require_positive(daily_vol, "daily volatility")
        participation = notional / adv
        impact = self.impact_y * vol * np.sqrt(participation)
        return (self.commission_bps + self.half_spread_bps) * 1e-4 + impact

    def trade_costs(self, trades: dict[str, float], adv: dict[str, float] | None, daily_vol: dict[str, float]) -> dict[str, float]:
        """Cost in currency for each traded asset. Refuses if a traded asset has no ADV."""
        out = {}
        for a, notional in trades.items():
            if notional == 0:
                out[a] = 0.0
                continue
            if adv is None or a not in adv:
                raise MissingInputError(f"no ADV for traded asset '{a}'; impact cannot be estimated (a blank is never zero)")
            out[a] = abs(notional) * self.order_cost_frac(notional, adv[a], daily_vol[a])
        return out

    def card(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "formula": "commission + half spread + Y * sigma_daily * sqrt(Q / ADV)",
            "params": {"commission_bps": self.commission_bps, "half_spread_bps": self.half_spread_bps, "Y": self.impact_y},
            "known_failure_modes": ["not reactive: orders do not move simulated prices (v1)",
                                    "square-root law is an average; single orders vary widely",
                                    "spread and ADV are user inputs, not data"],
        }
