"""Simulator benchmark 1: S&P 500 risk bands, 2000-2022, out of sample.

Every trading day from 2000-01-03, each path generator (fitted only on data before the day)
simulates 2,000 futures for the next 10 days. We score the 1-day and 10-day return
distributions against what happened:

- VaR 1% / 5%: breach rate, Kupiec and Christoffersen tests, pinball (quantile) loss
- (VaR, ES) at 2.5% (the Basel FRTB level): Fissler-Ziegel FZ0 loss
- whole distribution: CRPS
- Diebold-Mariano tests of the world model against GARCH(1,1)-t and GJR-GARCH-t

Parameters are refitted every 63 trading days on an expanding window; the volatility filter
updates daily. The State Encoder is refitted every January with a cutoff of the prior day.

Run:  python bench/simulator_sp500.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import RESULTS, md_table, save_json

from miko import data as D
from miko.evaluation import christoffersen, crps_from_samples, diebold_mariano, fz0, kupiec, pinball, var_es_from_samples
from miko.worldmodels.features import equity_state_features
from miko.worldmodels.paths import BlockBootstrap, GarchT, HistoricalSim, RandomWalk, StatePath
from miko.worldmodels.state_encoder import StateEncoder

TEST_START = "2000-01-03"
TEST_END = "2022-12-28"
VALID_START, VALID_END = "1996-01-02", "1999-12-31"
N_PATHS = 2000
H = 10
REFIT = 63
ALPHAS = (0.01, 0.025, 0.05)

BASELINES = {
    "random-walk": lambda enc, F: RandomWalk(),
    "historical-sim-500d": lambda enc, F: HistoricalSim(500),
    "block-bootstrap": lambda enc, F: BlockBootstrap(10),
    "garch-t": lambda enc, F: GarchT(),
    "gjr-garch-t": lambda enc, F: GarchT(asymmetric=True),
    "fhs (statepath, no state)": lambda enc, F: StatePath(widen=0.0),
}


def statepath_variant(kernel: bool, head: bool, widen: float):
    return lambda enc, F: StatePath(encoder=enc, features=F, kernel=kernel, vol_head=head, widen=widen)


VARIANTS = {
    f"statepath[kernel={int(k)},head={int(h)},widen={w}]": statepath_variant(k, h, w)
    for k in (False, True) for h in (False, True) for w in (0.0, 0.25)
}
MODELS = {**BASELINES, **VARIANTS}
SELECTED_FILE = "simulator_sp500_selected.json"


def run_model(args):
    name, step, start, end = args
    rets = D.return_panel()
    r = rets[["SP500"]]
    F = equity_state_features(rets, D.price_panel(), D.stock_names())
    dates = r.index
    test_pos = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0]
    test_pos = test_pos[test_pos + H < len(dates)][::step]
    x = r["SP500"].to_numpy()
    rows = []
    model, enc, enc_year, last_fit = None, None, None, -10**9
    for i, t in enumerate(test_pos):
        day = dates[t]
        if enc is None or day.year != enc_year:
            enc = StateEncoder().fit(F, dates[t - 1])
            enc_year = day.year
            last_fit = -10**9
        if t - last_fit >= REFIT:
            model = MODELS[name](enc, F).fit(r.iloc[:t])  # data strictly before the day...
            last_fit = t
        hist = r.iloc[: t + 1]  # returns up to and including the run date (close of day t)
        s = model.sample(N_PATHS, H, seed=int(t), history=hist)[..., 0]
        s1, s10 = s[:, 0], s.sum(axis=1)
        y1, y10 = x[t + 1], x[t + 1 : t + 1 + H].sum()
        row = {"date": day, "y1": y1, "y10": y10}
        for lab, smp in (("1", s1), ("10", s10)):
            row[f"crps{lab}"] = float(crps_from_samples(smp[None, :], np.array([row[f"y{lab}"]]))[0])
            for a in ALPHAS:
                v, e = var_es_from_samples(smp[None, :], a)
                row[f"var{lab}_{a}"] = float(v[0])
                row[f"es{lab}_{a}"] = float(e[0])
        if isinstance(model, StatePath):
            row["novelty_pct"] = model.last_novelty
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    safe = "".join(ch if ch.isalnum() or ch in "=-_." else "_" for ch in name)
    df.to_csv(RESULTS / "sim_sp500" / f"{start[:4]}_{safe}.csv")
    return name, df


def summarize(results: dict[str, pd.DataFrame], wm_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, rows10 = {}, {}
    for name, df in results.items():
        y1, y10 = df["y1"].to_numpy(), df["y10"].to_numpy()
        d = {}
        for a in (0.01, 0.05):
            hits = y1 <= df[f"var1_{a}"].to_numpy()
            d[f"breach {a:.0%} (target {a:.0%})"] = hits.mean()
            d[f"Kupiec p {a:.0%}"] = kupiec(hits, a)[1]
            d[f"Christoffersen p {a:.0%}"] = christoffersen(hits)[1]
            d[f"pinball {a:.0%} (x1e4)"] = 1e4 * pinball(y1, df[f"var1_{a}"].to_numpy(), a).mean()
        d["FZ0 2.5%"] = fz0(y1, df["var1_0.025"].to_numpy(), df["es1_0.025"].to_numpy(), 0.025).mean()
        d["CRPS (x1e4)"] = 1e4 * df["crps1"].mean()
        rows[name] = d
        e = {}
        nonover = np.arange(0, len(df), H)
        for a in (0.01, 0.05):
            hits = y10 <= df[f"var10_{a}"].to_numpy()
            e[f"breach {a:.0%}"] = hits.mean()
            e[f"Kupiec p {a:.0%} (non-overlapping)"] = kupiec(hits[nonover], a)[1]
            e[f"pinball {a:.0%} (x1e4)"] = 1e4 * pinball(y10, df[f"var10_{a}"].to_numpy(), a).mean()
        e["FZ0 2.5%"] = fz0(y10, df["var10_0.025"].to_numpy(), df["es10_0.025"].to_numpy(), 0.025).mean()
        e["CRPS (x1e4)"] = 1e4 * df["crps10"].mean()
        rows10[name] = e
    t1 = pd.DataFrame(rows).T
    t10 = pd.DataFrame(rows10).T
    t1.index.name = t10.index.name = "model"

    dm = {}
    if wm_name is None:
        return t1, t10, dm
    wm = results[wm_name]
    for base in ("garch-t", "gjr-garch-t", "fhs (statepath, no state)"):
        b = results[base]
        for lab, lag in (("1", 0), ("10", H)):
            y = wm[f"y{lab}"].to_numpy()
            for a in (0.01, 0.05):
                la = pinball(y, wm[f"var{lab}_{a}"].to_numpy(), a)
                lb = pinball(y, b[f"var{lab}_{a}"].to_numpy(), a)
                dm[f"{lab}d pinball {a:.0%} vs {base}"] = diebold_mariano(la, lb, lag)
            la = fz0(y, wm[f"var{lab}_0.025"].to_numpy(), wm[f"es{lab}_0.025"].to_numpy(), 0.025)
            lb = fz0(y, b[f"var{lab}_0.025"].to_numpy(), b[f"es{lab}_0.025"].to_numpy(), 0.025)
            dm[f"{lab}d FZ0 2.5% vs {base}"] = diebold_mariano(la, lb, lag)
            dm[f"{lab}d CRPS vs {base}"] = diebold_mariano(wm[f"crps{lab}"].to_numpy(), b[f"crps{lab}"].to_numpy(), lag)
    return t1, t10, dm


def select(t1: pd.DataFrame, t10: pd.DataFrame) -> str:
    """Pick the world-model configuration on the validation period: best mean rank over
    1-day and 10-day FZ0 (2.5%), CRPS and 1% pinball loss."""
    cand = [m for m in t1.index if m.startswith("statepath[")]
    m = pd.DataFrame({
        "fz0_1": t1.loc[cand, "FZ0 2.5%"], "fz0_10": t10.loc[cand, "FZ0 2.5%"],
        "crps_1": t1.loc[cand, "CRPS (x1e4)"], "crps_10": t10.loc[cand, "CRPS (x1e4)"],
        "pin1_1": t1.loc[cand, "pinball 1% (x1e4)"], "pin1_10": t10.loc[cand, "pinball 1% (x1e4)"],
    })
    rank = m.rank().mean(axis=1).sort_values()
    return rank.index[0], rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="every 10th day only (smoke test)")
    ap.add_argument("--validate", action="store_true", help="1996-1999 model selection run")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--models", default=None, help="comma-separated subset to (re)run; others are read from saved CSVs")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild the tables from saved per-day CSVs")
    args = ap.parse_args()
    step = 10 if args.fast else 1
    (RESULTS / "sim_sp500").mkdir(exist_ok=True, parents=True)
    if args.validate:
        start, end = VALID_START, VALID_END
        models = {"gjr-garch-t": BASELINES["gjr-garch-t"], "fhs (statepath, no state)": BASELINES["fhs (statepath, no state)"], **VARIANTS}
    else:
        start, end = TEST_START, TEST_END
        sel = json.loads((RESULTS / SELECTED_FILE).read_text())["selected"]
        # baselines, the selected configuration and the runner-up / no-head ablations
        keep = [sel, "statepath[kernel=1,head=1,widen=0.0]", "statepath[kernel=0,head=0,widen=0.0]"]
        models = {**BASELINES, **{k: VARIANTS[k] for k in dict.fromkeys(keep)}}
    def csv_path(m):
        safe = "".join(ch if ch.isalnum() or ch in "=-_." else "_" for ch in m)
        return RESULTS / "sim_sp500" / f"{start[:4]}_{safe}.csv"

    todo = [] if args.summarize_only else ([m for m in models if m in args.models.split(",")] if args.models else list(models))
    out = {}
    if todo:
        with mp.Pool(min(args.procs, len(todo))) as pool:
            out = dict(pool.map(run_model, [(m, step, start, end) for m in todo]))
    for m in models:
        if m not in out:
            out[m] = pd.read_csv(csv_path(m), index_col=0, parse_dates=True)
    results = {m: out[m] for m in models}
    if args.validate:
        t1, t10, _ = summarize(results)
        best, rank = select(t1, t10)
        save_json({"selected": best, "mean_rank": rank.to_dict(), "period": [start, end]}, SELECTED_FILE)
        text = [f"# Simulator model selection on validation period {start} to {end}", "", f"Selected: **{best}**", "",
                "## 1-day", md_table(t1), "", "## 10-day", md_table(t10), "", "## Mean rank (lower is better)",
                md_table(rank.to_frame("mean rank"))]
        (RESULTS / "simulator_sp500_validation.md").write_text("\n".join(text) + "\n")
        print("\n".join(text))
        return
    t1, t10, dm = summarize(results, wm_name=sel)
    t1 = t1.rename(index={sel: f"{sel} <- selected world model"})
    t10 = t10.rename(index={sel: f"{sel} <- selected world model"})
    n = len(next(iter(results.values())))
    text = [
        f"# Simulator benchmark: S&P 500 risk bands ({start} to {end}, {n} test days{', FAST' if args.fast else ''})",
        "",
        f"World model configuration chosen on 1996-1999 validation data, before this test: `{sel}`.",
        "",
        "## 1-day ahead",
        md_table(t1),
        "",
        "## 10-day ahead (overlapping windows; Kupiec on non-overlapping)",
        md_table(t10),
        "",
        "## Diebold-Mariano: world model vs GARCH baselines (negative stat = world model has lower loss)",
        "| test | DM stat | p-value |",
        "|---|---|---|",
    ] + [f"| {k} | {v[0]:.2f} | {v[1]:.3g} |" for k, v in dm.items()]
    suffix = "_fast" if args.fast else ""
    (RESULTS / f"simulator_sp500{suffix}.md").write_text("\n".join(text) + "\n")
    save_json({"selected": sel, "one_day": t1.to_dict(orient="index"), "ten_day": t10.to_dict(orient="index"), "dm": dm}, f"simulator_sp500{suffix}.json")
    print("\n".join(text))


if __name__ == "__main__":
    main()
