"""Public data loaders.

Sources (all public):
- S&P 500 index and 20 large US stocks, daily 1990-2022: bundled with the open-source
  `skfolio` package (skfolio.datasets).
- VIX daily close: github.com/datasets/finance-vix (CBOE data).
- WTI crude daily spot: github.com/datasets/oil-prices (EIA data).
- Exchange rates (8 currencies, daily 1990-2016): the LSTNet dataset used by the public
  GluonTS / TimeGrad benchmark (github.com/laiguokun/multivariate-time-series-data).

Files are cached in data_cache/. The data snapshot id is a hash of the cached inputs, and is
stamped on every result.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("MIKO_DATA_CACHE", ROOT / "data_cache"))

RAW = "https://raw.githubusercontent.com/"
REMOTE = {
    "vix-daily.csv": RAW + "datasets/finance-vix/main/data/vix-daily.csv",
    "wti-daily.csv": RAW + "datasets/oil-prices/main/data/wti-daily.csv",
    "exchange_rate.txt.gz": RAW + "laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
}

# The panel used by the agents ends with the bundled stock data.
PANEL_END = "2022-12-28"


def _fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        urllib.request.urlretrieve(REMOTE[name], p)
    return p


@lru_cache(maxsize=None)
def sp500_index() -> pd.Series:
    from skfolio.datasets import load_sp500_index

    s = load_sp500_index()["SP500"].astype(float)
    s.index = pd.DatetimeIndex(s.index).normalize()
    s.name = "SP500"
    return s


@lru_cache(maxsize=None)
def stocks() -> pd.DataFrame:
    from skfolio.datasets import load_sp500_dataset

    df = load_sp500_dataset().astype(float)
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


@lru_cache(maxsize=None)
def vix() -> pd.Series:
    df = pd.read_csv(_fetch("vix-daily.csv"), parse_dates=["DATE"])
    s = df.set_index("DATE")["CLOSE"].astype(float)
    s.name = "VIX"
    return s


@lru_cache(maxsize=None)
def wti() -> pd.Series:
    df = pd.read_csv(_fetch("wti-daily.csv"), parse_dates=["Date"])
    s = df.set_index("Date")["Price"].astype(float)
    # WTI front-month settled negative on 2020-04-20. Log returns are undefined there, so
    # non-positive prices are treated as missing and forward-filled (documented in the data card).
    s = s.where(s > 0).ffill()
    s.name = "WTI"
    return s


@lru_cache(maxsize=None)
def exchange_rates() -> pd.DataFrame:
    with gzip.open(_fetch("exchange_rate.txt.gz"), "rt") as f:
        df = pd.read_csv(f, header=None)
    df.columns = ["AUD", "GBP", "CAD", "CHF", "CNY", "JPY", "NZD", "SGD"]
    # GluonTS treats the series as business-daily from 1990-01-01; dates are nominal.
    df.index = pd.bdate_range("1990-01-01", periods=len(df))
    return df


@lru_cache(maxsize=None)
def price_panel() -> pd.DataFrame:
    """Daily levels on S&P 500 trading days: SP500, VIX, WTI and 20 stocks."""
    idx = sp500_index()
    out = pd.concat(
        [idx, vix().reindex(idx.index, method="ffill", limit=3), wti().reindex(idx.index, method="ffill", limit=3), stocks()],
        axis=1,
    )
    out = out.loc[:PANEL_END].dropna()
    return out


@lru_cache(maxsize=None)
def return_panel() -> pd.DataFrame:
    """Daily log returns (VIX as log change of the level)."""
    p = price_panel()
    r = np.log(p).diff().dropna()
    return r


def factor_names() -> list[str]:
    return list(return_panel().columns)


def stock_names() -> list[str]:
    return list(stocks().columns)


@lru_cache(maxsize=None)
def snapshot_id() -> str:
    """Hash of every cached input file plus the bundled frames: the data snapshot id."""
    h = hashlib.sha256()
    for name in sorted(REMOTE):
        h.update(_fetch(name).read_bytes())
    h.update(pd.util.hash_pandas_object(price_panel(), index=True).values.tobytes())
    return "d-" + h.hexdigest()[:8]


DATA_CARD = {
    "panel": "SP500, VIX, WTI and 20 US large caps (AAPL, AMD, BAC, BBY, CVX, GE, HD, JNJ, JPM, KO, LLY, MRK, MSFT, PEP, PFE, PG, RRC, UNH, WMT, XOM)",
    "frequency": "daily, S&P 500 trading days, 1990-01-03 to 2022-12-28",
    "returns": "log returns; VIX enters as the log change of its level",
    "known_issues": [
        "Survivorship bias: the 20 stocks are ones that existed through 2022 (skfolio dataset).",
        "WTI went negative on 2020-04-20; non-positive prices are forward-filled.",
        "VIX and WTI are forward-filled by up to 3 days to align holidays.",
    ],
}
