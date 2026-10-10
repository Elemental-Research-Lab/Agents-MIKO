"""Shared context for the world-model agents: point-in-time data, fitted models, the trial ledger.

Models are fitted with a data cutoff equal to the run date (or loaded from a checkpoint with an
earlier cutoff) and cached, so several agents in one session share the same world models.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import data as D
from ..core.guards import MissingInputError, check_cutoff, point_in_time, to_timestamp
from ..core.ledger import TrialLedger
from ..core.result import ModelRef
from ..sources import HistoricalSource, PanelSource
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
    source: PanelSource | None = None  # default: the historical research panel
    checkpoint_dir: Path | None = None
    _cache: dict = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.source is None:
            self.source = HistoricalSource()

    # ------------------------------------------------------------------ data
    @property
    def returns(self) -> pd.DataFrame:
        return self.source.returns

    @property
    def prices(self) -> pd.DataFrame:
        return self.source.prices

    @property
    def stock_names(self) -> list[str]:
        return list(self.source.stock_cols)

    @property
    def latest(self) -> pd.Timestamp:
        return self.source.as_of

    @property
    def features(self) -> pd.DataFrame:
        with self._lock:
            if "features" not in self._cache:
                self._cache["features"] = equity_state_features(self.returns, self.prices, self.stock_names)
            return self._cache["features"]

    @property
    def targets(self) -> pd.DataFrame:
        with self._lock:
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
        return self.source.snapshot_id

    # ---------------------------------------------------------------- models
    def encoder(self, as_of) -> StateEncoder:
        key = ("encoder", to_timestamp(as_of))
        with self._lock:
            if key not in self._cache:
                enc = self._load(key) or StateEncoder().fit(self.features, as_of, self.targets)
                self._store(key, enc)
            enc = self._cache[key]
        enc.assert_usable(as_of)
        return enc

    def path_model(self, as_of, assets: list[str], kind: str = "statepath"):
        key = (kind, to_timestamp(as_of), tuple(assets))
        with self._lock:
            return self._path_model(key, as_of, assets, kind)

    def _path_model(self, key, as_of, assets, kind):
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
        with self._lock:
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
        base = self.checkpoint_dir or CHECKPOINTS
        if self.source.name != "historical":  # live models are tied to the exact data snapshot
            base = base / self.source.name / self.source.snapshot_id
        return base / f"{name}.pkl"

    def _load(self, key):
        p = self._path(key)
        if self.save_checkpoints and p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def _store(self, key, obj) -> None:
        self._cache[key] = obj
        if self.save_checkpoints:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                with p.open("wb") as f:
                    pickle.dump(obj, f)
