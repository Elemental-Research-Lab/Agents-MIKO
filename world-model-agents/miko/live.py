"""Live daily data: provider, local store, session clock and quality gates.

The world models are daily models, so "live" means: after every NYSE close, pull the new
close, check it, and run every agent on the latest *completed* session. An unfinished
intraday bar is never used (it would be look-ahead from the model's point of view).

Providers
    YahooProvider   Yahoo Finance via yfinance: free, no key, ~15 min delayed, adjusted closes.
                    SP500 = ^GSPC, VIX = ^VIX, WTI = CL=F (front-month futures; the research panel
                    uses EIA spot, so levels differ slightly).
    ReplayProvider  serves the public research panel as if it were arriving day by day; used by
                    the tests and for demos without network access.

Any other feed (Polygon, Alpaca, a broker) only needs `fetch(names, start, end)`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .core.guards import MikoError, MissingInputError
from .sources import FACTORS, PanelSource

log = logging.getLogger("miko.live")
NY = ZoneInfo("America/New_York")
DEFAULT_STOCKS = ["AAPL", "AMD", "BAC", "BBY", "CVX", "GE", "HD", "JNJ", "JPM", "KO",
                  "LLY", "MRK", "MSFT", "PEP", "PFE", "PG", "RRC", "UNH", "WMT", "XOM"]
YAHOO_SYMBOLS = {"SP500": "^GSPC", "VIX": "^VIX", "WTI": "CL=F"}
HISTORY_START = "1990-01-01"
CLOSE_BUFFER = timedelta(minutes=30)  # wait this long after the close before trusting the bar


class DataQualityError(MikoError):
    """The live data failed a hard quality gate; agents must not run on it."""


class PriceProvider(Protocol):
    name: str

    def fetch(self, names: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily closes, index = session dates, columns = internal names (SP500, VIX, WTI, tickers)."""


# --------------------------------------------------------------------------- session clock
def _calendar():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS")


def now_ny() -> datetime:
    return datetime.now(tz=NY)


def last_complete_session(now: datetime) -> pd.Timestamp:
    """Most recent NYSE session whose close (plus a buffer for the feed) has passed."""
    cal = _calendar()
    now = now.astimezone(NY)
    day = pd.Timestamp(now.date())
    sessions = cal.sessions_in_range(day - pd.Timedelta(days=14), day)
    for s in reversed(sessions):
        close = cal.session_close(s).tz_convert(NY).to_pydatetime()
        if now >= close + CLOSE_BUFFER:
            return pd.Timestamp(s).tz_localize(None)
    raise MissingInputError("no completed NYSE session in the last two weeks")


def sessions_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Number of NYSE sessions strictly after a and up to and including b."""
    if b <= a:
        return 0
    return len(_calendar().sessions_in_range(a + pd.Timedelta(days=1), b))


# --------------------------------------------------------------------------------- providers
class YahooProvider:
    name = "yahoo"

    def __init__(self, retries: int = 3, pause: float = 2.0):
        self.retries, self.pause = retries, pause

    def fetch(self, names, start, end):
        import yfinance as yf

        symbols = [YAHOO_SYMBOLS.get(n, n) for n in names]
        back = {YAHOO_SYMBOLS.get(n, n): n for n in names}
        last_err = None
        for attempt in range(self.retries):
            try:
                raw = yf.download(symbols, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                                  end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                  auto_adjust=True, progress=False, threads=True)
                close = parse_yahoo_close(raw, symbols)
                close = close.rename(columns=back)
                close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
                return close
            except Exception as e:  # network errors, rate limits
                last_err = e
                log.warning("yahoo fetch failed (attempt %d): %s", attempt + 1, e)
                time.sleep(self.pause * (2**attempt))
        raise DataQualityError(f"Yahoo Finance fetch failed after {self.retries} attempts: {last_err}")


def parse_yahoo_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """yfinance returns (Price, Ticker) MultiIndex columns; older versions return flat columns
    for a single ticker. Returns closes with one column per requested symbol (NaN if absent)."""
    if raw is None or len(raw) == 0:
        raise DataQualityError("provider returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    return close.reindex(columns=symbols).astype(float)


class ReplayProvider:
    """The research panel served as if live: only rows dated <= `end` exist."""

    name = "replay"

    def __init__(self, levels: pd.DataFrame | None = None):
        from . import data as D

        self.levels = levels if levels is not None else D.price_panel()

    def fetch(self, names, start, end):
        unknown = [n for n in names if n not in self.levels.columns]
        out = self.levels.reindex(columns=names).loc[pd.Timestamp(start):pd.Timestamp(end)]
        if unknown:
            log.warning("replay has no data for %s", unknown)
        return out


# ------------------------------------------------------------------------------- local store
@dataclass
class PriceStore:
    """Local CSV cache of daily closes. Incremental updates; a full refetch for any symbol whose
    recent history changed (Yahoo back-adjusts for dividends and splits)."""

    path: Path
    overlap_days: int = 10
    revision_tol: float = 1e-3

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path, index_col=0, parse_dates=True)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        df.sort_index().to_csv(tmp)
        tmp.replace(self.path)  # atomic swap: readers never see a half-written file

    def update(self, provider: PriceProvider, names: list[str], end: pd.Timestamp) -> pd.DataFrame:
        old = self.load()
        start_full = pd.Timestamp(HISTORY_START)
        new_names = [n for n in names if n not in old.columns]
        if old.empty:
            df = provider.fetch(names, start_full, end)
        else:
            since = old.index[-1] - pd.Timedelta(days=self.overlap_days)
            recent = provider.fetch([n for n in names if n in old.columns], since, end)
            revised = []
            common = old.index.intersection(recent.index)
            for n in recent.columns:
                a, b = old.loc[common, n], recent.loc[common, n]
                ok = a.notna() & b.notna()
                if ok.any() and (np.abs(b[ok] / a[ok] - 1) > self.revision_tol).any():
                    revised.append(n)
            df = old.combine_first(recent)
            df.update(recent)
            refetch = revised + new_names
            if refetch:
                log.info("full refetch for %s", refetch)
                full = provider.fetch(refetch, start_full, end)
                for n in refetch:
                    df[n] = full[n] if n in full.columns else np.nan
        df = df.loc[:end].sort_index()
        df = df[[c for c in names if c in df.columns]]
        self.save(df)
        return df


# ------------------------------------------------------------------------------ quality gates
@dataclass
class QualityReport:
    as_of_expected: str
    as_of_data: str
    lag_sessions: int
    issues: list[str] = field(default_factory=list)  # hard failures
    warnings: list[str] = field(default_factory=list)
    dropped_assets: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}


def build_live_panel(raw: pd.DataFrame, expected: pd.Timestamp, stocks: list[str], max_lag: int = 1,
                     min_history: int = 750) -> tuple[PanelSource | None, QualityReport]:
    """Align a raw close table into an agent-ready panel and run the quality gates."""
    raw = raw.loc[:expected]  # never use an unfinished session
    rep = QualityReport(str(expected.date()), "", 0)
    for f in FACTORS:
        if f not in raw.columns or raw[f].dropna().empty:
            rep.issues.append(f"required factor {f} missing from the feed")
    if rep.issues:
        return None, rep
    sessions = raw["SP500"].dropna().index
    lv = raw.reindex(sessions)
    lv[["VIX", "WTI"]] = lv[["VIX", "WTI"]].ffill(limit=3)
    lv["WTI"] = lv["WTI"].where(lv["WTI"] > 0).ffill(limit=3)  # front-month futures went negative in April 2020
    last = pd.Timestamp(sessions[-1])
    rep.as_of_data = str(last.date())
    rep.lag_sessions = sessions_between(last, expected)
    if rep.lag_sessions > max_lag:
        rep.issues.append(f"stale data: last close {last.date()} is {rep.lag_sessions} sessions behind {expected.date()}")
    elif rep.lag_sessions > 0:
        rep.warnings.append(f"feed has not published {expected.date()} yet; serving {last.date()}")
    # stocks: drop any whose latest close is missing or whose history is too short
    usable = []
    for s in stocks:
        col = lv[s] if s in lv.columns else pd.Series(np.nan, index=lv.index)
        filled = col.ffill(limit=3)
        if pd.isna(filled.iloc[-1]):
            rep.dropped_assets.append(s)
            rep.warnings.append(f"{s}: no recent close; excluded today")
        elif filled.dropna().size < min_history:
            rep.dropped_assets.append(s)
            rep.warnings.append(f"{s}: less than {min_history} sessions of history; excluded")
        else:
            lv[s] = filled
            usable.append(s)
    if len(usable) < 5:
        rep.issues.append(f"only {len(usable)} usable stocks; need at least 5")
        return None, rep
    lv = lv[list(FACTORS) + usable]
    first = lv.apply(lambda c: c.first_valid_index()).max()
    lv = lv.loc[first:]
    gaps = lv.isna().sum()
    if gaps.any():
        rep.issues.append(f"gaps longer than 3 sessions in {list(gaps[gaps > 0].index)}")
    if (lv <= 0).any().any():
        rep.issues.append("non-positive prices in the panel")
    if len(lv) < min_history:
        rep.issues.append(f"only {len(lv)} sessions of aligned history; need {min_history}")
    if rep.issues:
        return None, rep
    r_last = np.log(lv.iloc[-1] / lv.iloc[-2])
    if abs(r_last["SP500"]) > 0.25:
        rep.issues.append(f"S&P 500 moved {r_last['SP500']:+.1%} in one session; held for manual check")
    for s in usable:
        if abs(r_last[s]) > 0.7:
            rep.warnings.append(f"{s} moved {r_last[s]:+.0%} today; check for a split or bad print")
    if rep.issues:
        return None, rep
    return PanelSource(lv, name="live", stock_cols=usable), rep


@dataclass
class LiveFeed:
    """Provider + store + clock. `refresh()` returns (source or None, quality report)."""

    provider: PriceProvider
    store: PriceStore
    stocks: list[str] = field(default_factory=lambda: list(DEFAULT_STOCKS))
    clock: Callable[[], datetime] = now_ny
    max_lag: int = 1

    def refresh(self) -> tuple[PanelSource | None, QualityReport]:
        expected = last_complete_session(self.clock())
        raw = self.store.update(self.provider, list(FACTORS) + self.stocks, expected)
        src, rep = build_live_panel(raw, expected, self.stocks, self.max_lag)
        log.info("live refresh: %s", json.dumps(rep.to_dict()))
        return src, rep
