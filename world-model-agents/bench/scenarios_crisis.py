"""Scenario Generator benchmark: fill in real crises from their headline moves.

For every 21-day window from 2000 to 2022, the engine is fitted on data up to the day before
the window, told only what the S&P 500, VIX and WTI oil did over the window (the "shocks"),
and asked to fill in the cumulative moves of 20 large US stocks. We score the filled-in
distribution against what the stocks actually did.

Two sets:
- crises: non-overlapping windows where the S&P 500 fell 10% or more (the stress-test case)
- all: every non-overlapping 21-day window (calm and stressed)

Metrics (per window, averaged; lower is better except IC and coverage):
- RMSE and MAE of the mean fill-in (cumulative log return, in %)
- IC: Spearman rank correlation between filled-in and actual moves across the 20 stocks
- CRPS of the Gaussian (mean, sd) fill-in
- coverage of the central 90% interval (target 0.90) and its absolute calibration error
Paired Wilcoxon signed-rank tests compare the kernel engine with each baseline.

Run:  python bench/scenarios_crisis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, md_table, save_json

from miko import data as D
from miko.worldmodels.scenario_engine import ScenarioEngine

H = 21
SHOCKED = ["SP500", "VIX", "WTI"]
START = "2000-01-03"
METHODS = {"local": "wm v1: local-linear", "slope": "wm v2: local slopes", "stresscov": "wm v3: stress covariance",
           "gauss": "gauss (full-history cov)", "gauss1y": "gauss1y (trailing 1y cov)",
           "beta": "beta (CAPM on S&P)", "zero": "zero (no fill-in)"}
WM = ("local", "slope", "stresscov")
VALID_END = "2007-12-31"  # crisis windows up to here choose the variant; 2008-2022 is the hold-out


def gauss_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def windows(r: pd.DataFrame) -> tuple[list[int], list[int]]:
    x = r["SP500"].to_numpy()
    cs = np.concatenate([[0], np.cumsum(x)])
    n = len(x)
    start = r.index.searchsorted(pd.Timestamp(START))
    allw = list(range(start, n - H + 1, H))
    cum = np.array([cs[t + H] - cs[t] for t in range(n - H + 1)])
    cand = [t for t in range(start, n - H + 1) if cum[t] <= np.log(0.90)]
    crises: list[int] = []
    for t in sorted(cand, key=lambda t: cum[t]):
        if all(abs(t - u) >= H for u in crises):
            crises.append(t)
    return sorted(crises), allw


def run(r: pd.DataFrame, starts: list[int]) -> pd.DataFrame:
    stocks = D.stock_names()
    rows = []
    for t in starts:
        as_of = r.index[t - 1]
        eng = ScenarioEngine(horizon=H).fit(r, as_of)
        actual = r.iloc[t : t + H].sum()
        shocks = {k: float(actual[k]) for k in SHOCKED}
        y = actual[stocks].to_numpy()
        for m in METHODS:
            c = eng.condition(shocks, m)
            idx = [eng.factors.index(s) for s in stocks]
            mu, sd = c.mean[idx], np.maximum(c.sd()[idx], 1e-6)
            lo, hi = mu - 1.645 * sd, mu + 1.645 * sd
            ic = stats.spearmanr(mu, y).correlation if np.std(mu) > 0 else np.nan
            rows.append({
                "window_start": r.index[t].date(), "method": m,
                "sp500_move_%": 100 * (np.exp(shocks["SP500"]) - 1),
                "RMSE_%": 100 * np.sqrt(np.mean((mu - y) ** 2)),
                "MAE_%": 100 * np.mean(np.abs(mu - y)),
                "IC": ic,
                "CRPS_%": 100 * gauss_crps(mu, sd, y).mean(),
                "coverage90": np.mean((y >= lo) & (y <= hi)),
                "calib_err90": abs(np.mean((y >= lo) & (y <= hi)) - 0.90),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, wm: str) -> tuple[pd.DataFrame, dict]:
    t = df.groupby("method")[["RMSE_%", "MAE_%", "IC", "CRPS_%", "coverage90", "calib_err90"]].mean().loc[list(METHODS)]
    t.index = [METHODS[m] + (" <- selected" if m == wm else "") for m in t.index]
    t.index.name = "method"
    tests = {}
    k = df[df.method == wm].set_index("window_start")
    for m in METHODS:
        if m in WM:
            continue
        b = df[df.method == m].set_index("window_start")
        for metric in ("RMSE_%", "CRPS_%", "coverage90", "calib_err90"):
            d = (k[metric] - b[metric]).dropna()
            p = stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
            better = (d > 0) if metric == "coverage90" else (d < 0)
            tests[f"{wm} vs {m}: {metric}"] = {"world model better in": f"{int(better.sum())}/{len(d)} (ties {int((d == 0).sum())})", "median diff": float(d.median()), "wilcoxon p": float(p)}
    return t, tests


def section(label, df, wm, text, out):
    t, tests = summarize(df, wm)
    n = df.window_start.nunique()
    out[label] = {"n_windows": n, "table": t.to_dict(orient="index"), "tests": tests}
    text += [f"## {label}: {n} windows", md_table(t, "{:.3f}"), "",
             "| comparison | world model better in | median diff (pp) | Wilcoxon p |", "|---|---|---|---|"]
    text += [f"| {k} | {v['world model better in']} | {v['median diff']:.3f} | {v['wilcoxon p']:.3g} |" for k, v in tests.items()]
    text.append("")


def main():
    r = D.return_panel()
    crises, allw = windows(r)
    dfc = run(r, crises)
    dfa = run(r, allw)
    dfc.to_csv(RESULTS / "scenarios_crisis_windows.csv", index=False)
    dfa.to_csv(RESULTS / "scenarios_all_windows.csv", index=False)
    valid = dfc[pd.to_datetime(dfc.window_start) <= pd.Timestamp(VALID_END)]
    hold = dfc[pd.to_datetime(dfc.window_start) > pd.Timestamp(VALID_END)]
    sel_scores = valid[valid.method.isin(WM)].groupby("method")["CRPS_%"].mean().sort_values()
    wm = sel_scores.index[0]
    out = {"selected_variant": wm, "selection_crps": sel_scores.to_dict()}
    text = ["# Scenario Generator benchmark: filling in real crises (2000-2022)", "",
            "Shocks given: S&P 500, VIX and WTI cumulative 21-day moves. Filled in: 20 US large caps.",
            "Each window uses an engine fitted only on data before it.",
            "",
            f"World-model variant chosen by lowest mean CRPS on the {valid.window_start.nunique()} crisis windows up to {VALID_END}: **{METHODS[wm]}**",
            "(selection CRPS: " + ", ".join(f"{METHODS[k]} {v:.3f}" for k, v in sel_scores.items()) + ").",
            "The 2008-2022 crisis windows are the hold-out.", ""]
    section("hold-out crises 2008-2022 (S&P 500 down 10% or more in 21 days)", hold, wm, text, out)
    section("validation crises 2000-2007 (used to choose the variant)", valid, wm, text, out)
    section("all non-overlapping 21-day windows 2000-2022", dfa, wm, text, out)
    per = dfc[dfc.method.isin([wm, "gauss", "beta"])].pivot(index="window_start", columns="method", values=["RMSE_%", "CRPS_%"])
    per.columns = [f"{a} {b}" for a, b in per.columns]
    per.insert(0, "S&P 500 move %", dfc[dfc.method == wm].set_index("window_start")["sp500_move_%"])
    text += ["## Per crisis window", md_table(per, "{:.2f}"), ""]
    (RESULTS / "scenarios_crisis.md").write_text("\n".join(text) + "\n")
    save_json(out, "scenarios_crisis.json")
    print("\n".join(text))


if __name__ == "__main__":
    main()
