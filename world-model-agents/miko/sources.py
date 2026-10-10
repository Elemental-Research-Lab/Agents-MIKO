"""Data sources the agents run on.

HistoricalSource  the fixed public research panel (1990-2022) used by the benchmarks.
PanelSource       any aligned daily price panel; LiveSource (miko/live.py) builds one from a live
                  feed after its quality checks pass.

Both expose the same interface, so AgentContext and the agents don't care which one they get.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from . import data as D
from .core.guards import MissingInputError

FACTORS = ("SP500", "VIX", "WTI")


@dataclass
class PanelSource:
    """An aligned panel of daily closing levels: SP500, VIX, WTI and stocks, one row per session."""

    levels: pd.DataFrame
    name: str = "panel"
    stock_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = [f for f in FACTORS if f not in self.levels.columns]
        if missing:
            raise MissingInputError(f"price panel lacks required factors {missing}")
        if not self.stock_cols:
            self.stock_cols = [c for c in self.levels.columns if c not in FACTORS]
        if len(self.stock_cols) < 5:
            raise MissingInputError("need at least 5 stocks for correlation and dispersion features")
        if self.levels.isna().any().any():
            raise MissingInputError("price panel has blanks; refusing to fill them with zeros")
        if (self.levels <= 0).any().any():
            raise MissingInputError("price panel has non-positive prices")

    @property
    def prices(self) -> pd.DataFrame:
        return self.levels

    @cached_property
    def returns(self) -> pd.DataFrame:
        return np.log(self.levels).diff().dropna()

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.levels.index[-1])

    @cached_property
    def snapshot_id(self) -> str:
        h = hashlib.sha256(pd.util.hash_pandas_object(self.levels, index=True).values.tobytes())
        return "d-" + h.hexdigest()[:8]

    @property
    def assets(self) -> list[str]:
        return list(self.levels.columns)


class HistoricalSource(PanelSource):
    def __init__(self) -> None:
        super().__init__(D.price_panel(), name="historical", stock_cols=D.stock_names())

    @cached_property
    def snapshot_id(self) -> str:
        return D.snapshot_id()
