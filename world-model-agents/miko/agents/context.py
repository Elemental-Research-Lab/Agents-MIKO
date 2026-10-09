"""Shared context for the world-model agents: point-in-time data, fitted models, the trial ledger.

Models are fitted with a data cutoff equal to the run date (or loaded from a checkpoint with an
earlier cutoff) and cached, so several agents in one session share the same world models.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import data as D
from ..core.guards import MissingInputError, check_cutoff, point_in_time, to_timestamp
from ..core.ledger import TrialLedger
from ..core.result import ModelRef
from ..worldmodels.features import equity_state_features, forward_targets
from ..worldmodels.impact import ImpactModel
from ..worldmodels.paths import BlockBootstrap, GarchT, RandomWalk, StatePath
from ..worldmodels.scenario_engine import ScenarioEngine
from ..worldmodels.state_encoder import StateEncoder

CHECKPOINTS = D.ROOT / "checkpoints"


@dataclass
class AgentContext:
    ledger: TrialLedger = field(default_factory=TrialLedger)
    impact: ImpactModel = field(default_factory=ImpactModel)
    save_checkpoints: bool = False
    _cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ data
    @property
    def returns(self) -> pd.DataFrame:
        return D.return_panel()

    @property
    def prices(self) -> pd.DataFrame:
        return D.price_panel()

    @property
    def features(self) -> pd.DataFrame:
        if "features" not in self._cache:
            self._cache["features"] = equity_state_features(self.returns, self.prices, D.stock_names())
        return self._cache["features"]

    @property
    def targets(self) -> pd.DataFrame:
        if "targets" not in self._cache:
            self._cache["targets"] = forward_targets(self.returns["SP500"], self.prices["SP500"], 21)
        return self._cache["targets"]

    def returns_upto(self, as_of, assets: list[str] | None = None) -> pd.DataFrame:
        r = point_in_time(self.returns, as_of)
        if assets is not None:
            unknown = [a for a in assets if a not in r.columns]
            if unknown:
                raise MissingInputError(f"unknown assets {unknown}; known: {list(r.columns)}")
            r = r[assets]
        if r.index[-1] != to_timestamp(as_of):
            raise MissingInputError(f"{to_timestamp(as_of).date()} is not a trading day in the data (last day on or before it: {r.index[-1].date()})")
        return r

    def snapshot(self) -> str:
        return D.snapshot_id()

    # ---------------------------------------------------------------- models
    def encoder(self, as_of) -> StateEncoder:
        key = ("encoder", to_timestamp(as_of))
        if key not in self._cache:
            enc = self._load(key) or StateEncoder().fit(self.features, as_of, self.targets)
            self._store(key, enc)
        enc = self._cache[key]
        enc.assert_usable(as_of)
        return enc

    def path_model(self, as_of, assets: list[str], kind: str = "statepath"):
        key = (kind, to_timestamp(as_of), tuple(assets))
        if key not in self._cache:
            train = self.returns_upto(as_of, assets)
            if kind == "statepath":
                m = StatePath(encoder=self.encoder(as_of), features=self.features, kernel=False, vol_head=True, widen=0.0)
            elif kind == "garch-t":
                m = GarchT()
            elif kind == "block-bootstrap":
                m = BlockBootstrap(10)
            elif kind == "random-walk":
                m = RandomWalk()
            else:
                raise ValueError(f"unknown path model {kind}")
            self._store(key, m.fit(train))
        m = self._cache[key]
        check_cutoff(m.name, m.cutoff, as_of)
        return m

    def scenario_engine(self, as_of, horizon: int) -> ScenarioEngine:
        key = ("scenario", to_timestamp(as_of), horizon)
        if key not in self._cache:
            self._store(key, ScenarioEngine(horizon=horizon).fit(self.returns, as_of))
        e = self._cache[key]
        check_cutoff(e.name, e.cutoff, as_of)
        return e

    @staticmethod
    def ref(model) -> ModelRef:
        return ModelRef(model.name, model.version, str(pd.Timestamp(model.cutoff).date()))

    # ------------------------------------------------------------ checkpoints
    def _path(self, key) -> Path:
        name = "_".join(str(k.date()) if isinstance(k, pd.Timestamp) else ("-".join(k) if isinstance(k, tuple) else str(k)) for k in key)
        return CHECKPOINTS / f"{name}.pkl"

    def _load(self, key):
        p = self._path(key)
        if self.save_checkpoints and p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def _store(self, key, obj) -> None:
        self._cache[key] = obj
        if self.save_checkpoints:
            CHECKPOINTS.mkdir(exist_ok=True)
            p = self._path(key)
            if not p.exists():
                with p.open("wb") as f:
                    pickle.dump(obj, f)
