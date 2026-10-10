"""Live data layer: session clock, provider parsing, local store, quality gates (with planted defects)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from miko import data as D
from miko.live import (NY, DEFAULT_STOCKS, LiveFeed, PriceStore, ReplayProvider, build_live_panel, last_complete_session,
                       parse_yahoo_close)


def at(s: str) -> datetime:
    return pd.Timestamp(s).tz_localize(NY).to_pydatetime()


@pytest.mark.parametrize("now, expected", [
    ("2026-10-09 15:00", "2026-10-08"),  # Friday, market still open: yesterday is the last complete session
    ("2026-10-09 16:45", "2026-10-09"),  # after the close + buffer
    ("2026-10-10 12:00", "2026-10-09"),  # Saturday
    ("2026-10-12 09:00", "2026-10-09"),  # Monday morning (Columbus Day: NYSE open, so Monday is a session)
    ("2026-04-04 12:00", "2026-04-02"),  # Good Friday is an NYSE holiday
    ("2026-11-27 13:45", "2026-11-27"),  # day after Thanksgiving closes at 13:00
])
def test_last_complete_session(now, expected):
    assert last_complete_session(at(now)) == pd.Timestamp(expected)


def test_parse_yahoo_multiindex_and_flat():
    idx = pd.date_range("2024-01-02", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["^GSPC", "AAPL"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4) + 1, index=idx, columns=cols)
    out = parse_yahoo_close(raw, ["^GSPC", "AAPL", "MISSING"])
    assert list(out.columns) == ["^GSPC", "AAPL", "MISSING"] and out["MISSING"].isna().all()
    assert out["AAPL"].iloc[0] == raw[("Close", "AAPL")].iloc[0]
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [1, 2, 3]}, index=idx)
    assert parse_yahoo_close(flat, ["^VIX"])["^VIX"].tolist() == [1.0, 2.0, 3.0]


def test_store_incremental_and_revision(tmp_path):
    lv = D.price_panel()
    prov = ReplayProvider(lv.copy())
    store = PriceStore(tmp_path / "p.csv")
    names = list(lv.columns)
    a = store.update(prov, names, pd.Timestamp("2022-12-20"))
    assert a.index[-1] == pd.Timestamp("2022-12-20")
    b = store.update(prov, names, pd.Timestamp("2022-12-28"))
    assert b.index[-1] == pd.Timestamp("2022-12-28") and len(b) == len(a) + 5
    # planted: the provider back-adjusts AAPL's whole history (a dividend adjustment)
    prov.levels["AAPL"] = prov.levels["AAPL"] * 0.99
    c = store.update(prov, names, pd.Timestamp("2022-12-28"))
    assert np.allclose(c["AAPL"].loc["2000-01-03"], lv["AAPL"].loc["2000-01-03"] * 0.99)


def _raw(end="2022-12-28"):
    return D.price_panel().loc[:end].copy()


def test_panel_ok():
    src, rep = build_live_panel(_raw(), pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert rep.ok and src.as_of == pd.Timestamp("2022-12-28") and len(src.stock_cols) == 20


def test_unfinished_bar_is_never_used():
    raw = _raw()
    src, rep = build_live_panel(raw, pd.Timestamp("2022-12-27"), DEFAULT_STOCKS)  # 12-28 bar exists but is not complete
    assert src.as_of == pd.Timestamp("2022-12-27")


def test_stale_data_is_refused():
    src, rep = build_live_panel(_raw("2022-12-20"), pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)  # planted: feed stopped
    assert src is None and any("stale" in i for i in rep.issues)


def test_one_session_lag_is_served_with_warning():
    src, rep = build_live_panel(_raw("2022-12-27"), pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert src is not None and rep.lag_sessions == 1 and rep.warnings


def test_missing_factor_is_refused():
    src, rep = build_live_panel(_raw().drop(columns=["VIX"]), pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert src is None and any("VIX" in i for i in rep.issues)


def test_stock_without_recent_close_is_dropped_not_zeroed():
    raw = _raw()
    raw.loc["2022-12-21":, "PFE"] = np.nan  # planted: delisted / feed gap longer than 3 sessions
    src, rep = build_live_panel(raw, pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert src is not None and "PFE" not in src.stock_cols and "PFE" in rep.dropped_assets


def test_bad_print_on_index_is_held():
    raw = _raw()
    raw.loc["2022-12-28", "SP500"] *= 0.5  # planted: a bad print
    src, rep = build_live_panel(raw, pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert src is None and any("manual check" in i for i in rep.issues)


def test_too_few_stocks_is_refused():
    src, rep = build_live_panel(_raw()[["SP500", "VIX", "WTI", "AAPL", "MSFT"]], pd.Timestamp("2022-12-28"), DEFAULT_STOCKS)
    assert src is None


def test_feed_end_to_end(tmp_path):
    feed = LiveFeed(ReplayProvider(), PriceStore(tmp_path / "p.csv"), clock=lambda: at("2022-12-28 18:00"))
    src, rep = feed.refresh()
    assert rep.ok and src.as_of == pd.Timestamp("2022-12-28")


def test_yahoo_provider_maps_symbols(monkeypatch):
    """YahooProvider without network: fake yfinance.download returning the real response layout."""
    import yfinance

    from miko.live import YAHOO_SYMBOLS, YahooProvider

    lv = D.price_panel().loc["2022-12-01":"2022-12-28"]
    calls = {}

    def fake_download(symbols, start, end, **kw):
        calls.update(symbols=symbols, start=start, end=end, kw=kw)
        back = {v: k for k, v in YAHOO_SYMBOLS.items()}
        idx = lv.index.tz_localize("America/New_York")  # yfinance returns tz-aware dates
        data = {("Close", s): lv[back.get(s, s)].to_numpy() for s in symbols}
        data.update({("Open", s): lv[back.get(s, s)].to_numpy() for s in symbols})
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(yfinance, "download", fake_download)
    out = YahooProvider().fetch(["SP500", "VIX", "WTI", "AAPL"], pd.Timestamp("2022-12-01"), pd.Timestamp("2022-12-28"))
    assert calls["symbols"] == ["^GSPC", "^VIX", "CL=F", "AAPL"] and calls["kw"]["auto_adjust"] is True
    assert calls["end"] == "2022-12-29"  # yfinance's end date is exclusive
    assert list(out.columns) == ["SP500", "VIX", "WTI", "AAPL"] and out.index.tz is None
    assert np.allclose(out["SP500"], lv["SP500"])


def test_yahoo_provider_retries_then_refuses(monkeypatch):
    import yfinance

    from miko.live import DataQualityError, YahooProvider

    n = {"calls": 0}

    def broken(*a, **k):
        n["calls"] += 1
        raise ConnectionError("network down")  # planted outage

    monkeypatch.setattr(yfinance, "download", broken)
    with pytest.raises(DataQualityError):
        YahooProvider(retries=3, pause=0.0).fetch(["SP500"], pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-28"))
    assert n["calls"] == 3
