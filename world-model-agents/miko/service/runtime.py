"""Live runtime: keeps the agents on the latest completed session, with hot swaps.

Every `refresh_seconds` a background thread asks the feed for new closes. When a new session
appears it builds a candidate context (new data, freshly fitted world models), runs the data
quality report and the model gates, and only then swaps it in. Requests keep using the previous
context until the swap, and if anything fails the previous context stays live and /ready reports
why. Nothing is ever served from unchecked data.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..agents.context import AgentContext
from ..core.ledger import TrialLedger
from ..live import LiveFeed, QualityReport, last_complete_session, sessions_between
from .gates import gates_to_dicts, run_gates

log = logging.getLogger("miko.runtime")


@dataclass
class Snapshot:
    ctx: AgentContext
    quality: QualityReport
    gates: list[dict]
    activated_at: str


@dataclass
class LiveRuntime:
    feed: LiveFeed
    ledger_path: Path
    checkpoint_dir: Path | None = None
    refresh_seconds: float = 1800.0
    max_lag: int = 1
    current: Snapshot | None = None
    last_attempt: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = None

    # ------------------------------------------------------------------ refresh
    def refresh(self, force: bool = False) -> dict:
        """Pull data; if there is a new session (or force), build, gate and swap. Never raises."""
        t0 = time.time()
        attempt: dict = {"started": pd.Timestamp.now(tz="America/New_York").isoformat()}
        try:
            src, rep = self.feed.refresh()
            attempt["quality"] = rep.to_dict()
            if src is None:
                attempt["result"] = "rejected: data quality"
                return self._finish(attempt, t0)
            if self.current is not None and not force and src.snapshot_id == self.current.ctx.snapshot():
                attempt["result"] = "no new data"
                return self._finish(attempt, t0)
            ctx = AgentContext(ledger=TrialLedger(self.ledger_path), source=src,
                               save_checkpoints=self.checkpoint_dir is not None, checkpoint_dir=self.checkpoint_dir)
            checks = run_gates(ctx)
            attempt["gates"] = gates_to_dicts(checks)
            if not all(c.passed for c in checks):
                attempt["result"] = "rejected: model gates"
                return self._finish(attempt, t0)
            self._warm(ctx)
            with self._lock:
                self.current = Snapshot(ctx, rep, attempt["gates"], pd.Timestamp.now(tz="America/New_York").isoformat())
            attempt["result"] = f"activated {src.as_of.date()} ({src.snapshot_id})"
        except Exception as e:  # keep serving the last good snapshot
            attempt["result"] = f"error: {type(e).__name__}: {e}"
            attempt["traceback"] = traceback.format_exc(limit=5)
        return self._finish(attempt, t0)

    def _warm(self, ctx: AgentContext) -> None:
        """Fit the models the default requests need, so the first user does not wait."""
        as_of = ctx.latest
        enc = ctx.encoder(as_of)
        enc.filtered_regime_probs(ctx.features.loc[:as_of])
        ctx.path_model(as_of, ctx.stock_names, "statepath")
        ctx.path_model(as_of, list(ctx.prices.columns), "statepath")
        ctx.scenario_engine(as_of, 21)

    def _finish(self, attempt: dict, t0: float) -> dict:
        attempt["seconds"] = round(time.time() - t0, 2)
        self.last_attempt = attempt
        log.info("refresh: %s", json.dumps({k: v for k, v in attempt.items() if k != "traceback"}, default=str))
        return attempt

    # ---------------------------------------------------------------- readiness
    def ctx(self) -> AgentContext | None:
        with self._lock:
            return self.current.ctx if self.current else None

    def readiness(self) -> tuple[bool, dict]:
        snap = self.current
        info: dict = {"last_refresh": {k: v for k, v in self.last_attempt.items() if k != "traceback"}}
        if snap is None:
            info["reason"] = "no live snapshot has passed the gates yet"
            return False, info
        expected = last_complete_session(self.feed.clock())
        lag = sessions_between(snap.ctx.latest, expected)
        info.update({"as_of": str(snap.ctx.latest.date()), "expected_session": str(expected.date()), "lag_sessions": lag,
                     "data_snapshot": snap.ctx.snapshot(), "activated_at": snap.activated_at,
                     "assets": list(snap.ctx.prices.columns), "warnings": snap.quality.warnings})
        if lag > self.max_lag:
            info["reason"] = f"serving data {lag} sessions old (limit {self.max_lag}); latest refresh: {self.last_attempt.get('result')}"
            return False, info
        return True, info

    # --------------------------------------------------------------- background
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.refresh()
                self._stop.wait(self.refresh_seconds)

        self._thread = threading.Thread(target=loop, name="miko-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
