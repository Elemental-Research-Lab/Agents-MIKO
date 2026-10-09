"""State Reader benchmark (S&P 500, 2000-2022, out of sample, refitted every January).

1. What follows: forecast next-21-day realized volatility. QLIKE loss vs EWMA (RiskMetrics),
   GARCH(1,1), GJR-GARCH, HAR (realized vols only), VIX only, and the analogs' own aftermath.
2. Early warning: probability that the S&P 500 suffers a 7%+ drawdown within the next 21 days.
   AUC / Brier / log loss vs VIX-only, realized-vol-only, the 200-day moving-average rule and
   the HMM stress-regime probability.
3. Regime reading vs a public label: NBER US recession months (filtered regime probabilities,
   no look-ahead). AUC of the stress-regime probability vs VIX, realized vol and the MA200 rule.
4. Novelty: do days the State Reader flags as unusual really see bigger moves and worse model
   errors? Results by novelty quintile.

Run:  python bench/state_reader.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import RESULTS, md_table, save_json

from miko import data as D
from miko.evaluation import diebold_mariano, qlike
from miko.worldmodels.features import equity_state_features, forward_targets
from miko.worldmodels.paths import _fit_arch, garch_filter, gjr_horizon_var
from miko.worldmodels.state_encoder import StateEncoder

warnings.filterwarnings("ignore")
H = 21
TEST_START = "2000-01-03"
THRESH = 0.07
# NBER US business-cycle recessions (peak month to trough month), public: nber.org/research/business-cycle-dating
NBER = [("1990-07-01", "1991-03-31"), ("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]


def calibrated_log_reg(X, y_logvol):
    reg = LinearRegression().fit(X, y_logvol)
    c = float(np.mean(np.exp(2 * y_logvol) / np.exp(2 * reg.predict(X))))
    return lambda Xn: c * np.exp(2 * reg.predict(Xn))


def compute():
    R = D.return_panel()
    P = D.price_panel()
    F = equity_state_features(R, P, D.stock_names())
    T = forward_targets(R["SP500"], P["SP500"], H)
    r = R["SP500"]
    x = 100 * r.to_numpy()
    dates = r.index
    ma200 = np.log(P["SP500"]).reindex(dates)
    below_ma = (ma200 < ma200.rolling(200).mean()).astype(float)
    days = F.index[(F.index >= pd.Timestamp(TEST_START)) & F.index.isin(T.dropna().index)]
    years = sorted(set(days.year))
    rows = []
    for yr in years:
        test = days[days.year == yr]
        cutoff = dates[dates.searchsorted(pd.Timestamp(f"{yr}-01-01")) - 1]
        enc = StateEncoder().fit(F, cutoff, T)
        Ztr = enc.transform(F.loc[:cutoff])
        Ttr = T.reindex(Ztr.index)
        closed = Ztr.index[: len(Ztr) - H]
        ok = Ttr.loc[closed].dropna().index
        y = Ttr.loc[ok, "fwd_logrv21"].to_numpy()
        yb = (Ttr.loc[ok, "fwd_mdd21"] <= -THRESH).astype(int).to_numpy()
        har = calibrated_log_reg(Ztr.loc[ok, ["rv5", "rv21", "rv63"]].to_numpy(), y)
        vixm = calibrated_log_reg(Ztr.loc[ok, ["vix"]].to_numpy(), y)
        lg_vix = LogisticRegression(max_iter=2000).fit(Ztr.loc[ok, ["vix"]].to_numpy(), yb)
        lg_rv = LogisticRegression(max_iter=2000).fit(Ztr.loc[ok, ["rv21"]].to_numpy(), yb)
        ct = dates.get_loc(cutoff)
        p_g = _fit_arch(x[: ct + 1], 0, "normal")
        p_j = _fit_arch(x[: ct + 1], 1, "normal")
        s_g, s_j = garch_filter(x, p_g), garch_filter(x, p_j)
        # EWMA variance (RiskMetrics, lambda 0.94)
        e = np.empty(len(x) + 1)
        e[0] = np.var(x[:250])
        for i in range(len(x)):
            e[i + 1] = 0.94 * e[i] + 0.06 * x[i] ** 2
        reg = enc.filtered_regime_probs(F.loc[:test[-1]])
        Zt = enc.transform(F.loc[test])
        fh = har(Zt[["rv5", "rv21", "rv63"]].to_numpy())
        fv = vixm(Zt[["vix"]].to_numpy())
        pv = lg_vix.predict_proba(Zt[["vix"]].to_numpy())[:, 1]
        pr = lg_rv.predict_proba(Zt[["rv21"]].to_numpy())[:, 1]
        hist_rv = np.exp(2 * T["fwd_logrv21"])
        for k, d in enumerate(test):
            t = dates.get_loc(d)
            fu = F.loc[:d]
            risk = enc.forward_risk(fu)
            an = enc.analogs(fu, 10)
            nov, nov_pct = enc.novelty(fu)
            ann = 252.0 / 1e4
            row = {
                "date": d,
                "rv_real": float(np.exp(2 * T.loc[d, "fwd_logrv21"])),
                "dd_event": int(T.loc[d, "fwd_mdd21"] <= -THRESH),
                "abs_r1": abs(r.iloc[t + 1]),
                "ewma": e[t + 1] * ann,
                "garch": gjr_horizon_var(s_g[t + 1 : t + 2], p_g, H)[0] / H * ann,
                "gjr": gjr_horizon_var(s_j[t + 1 : t + 2], p_j, H)[0] / H * ann,
                "har": fh[k],
                "vix_only": fv[k],
                "analogs": float(np.median(hist_rv.loc[an["date"]])),
                "state_reader": risk["fwd_vol21"] ** 2,
                "p_state": risk["p_drawdown"],
                "p_vix": pv[k],
                "p_rv21": pr[k],
                "ma200": below_ma.loc[d],
                "p_stress": float(reg.loc[d, ["elevated", "crisis"]].sum()),
                "novelty_pct": nov_pct,
                "gjr_1d_sd": np.sqrt(s_j[t + 1]) / 100,
            }
            rows.append(row)
        print(yr, len(test), flush=True)
    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(RESULTS / "state_reader_daily.csv")
    return df


def block_bootstrap_auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, years: np.ndarray, n: int = 2000, seed: int = 0):
    """90% CI of AUC(a) - AUC(b) resampling whole calendar years (keeps recessions intact)."""
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    groups = [np.where(years == u)[0] for u in uy]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uy), len(uy))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="rebuild the report from results/state_reader_daily.csv")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / "state_reader_daily.csv", index_col=0, parse_dates=True) if args.from_csv else compute()
    R = D.return_panel()
    r = R["SP500"]
    F = equity_state_features(R, D.price_panel(), D.stock_names())
    df["vix_level"] = np.exp(F["vix"].reindex(df.index))

    # 1. volatility
    vol_models = ["ewma", "garch", "gjr", "har", "vix_only", "analogs", "state_reader"]
    vt = pd.DataFrame({m: {"QLIKE": qlike(df.rv_real, df[m]).mean(),
                           "MSE log-vol": np.mean((0.5 * np.log(df.rv_real) - 0.5 * np.log(df[m])) ** 2)} for m in vol_models}).T
    vt.index.name = "next-21-day volatility forecast"
    dm = {b: diebold_mariano(qlike(df.rv_real, df.state_reader).to_numpy(), qlike(df.rv_real, df[b]).to_numpy(), lag=H) for b in vol_models if b != "state_reader"}
    lmse = lambda m: (0.5 * np.log(df.rv_real) - 0.5 * np.log(df[m])).to_numpy() ** 2
    dm_mse = {b: diebold_mariano(lmse("state_reader"), lmse(b), lag=H) for b in vol_models if b != "state_reader"}

    # 2. drawdown early warning
    y = df.dd_event.to_numpy()
    warn = {}
    for m, lab in [("p_state", "State Reader (5 state features)"), ("p_vix", "VIX only"), ("p_rv21", "21-day realized vol only"),
                   ("p_stress", "HMM stress-regime probability"), ("ma200", "below 200-day moving average")]:
        p = np.clip(df[m].to_numpy(), 1e-4, 1 - 1e-4)
        warn[lab] = {"AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "log loss": log_loss(y, p)}
    wt = pd.DataFrame(warn).T
    wt.index.name = f"P(7% drawdown within 21 days); base rate {y.mean():.3f}"

    # 3. NBER recessions
    rec = pd.Series(0, index=df.index)
    for a, b in NBER:
        rec[(rec.index >= a) & (rec.index <= b)] = 1
    nb = {}
    yrs = df.index.year.to_numpy()
    for m, lab in [("p_stress", "HMM stress-regime probability"), ("vix_level", "VIX level"), ("gjr", "GJR-GARCH volatility"), ("ma200", "below 200-day moving average")]:
        nb[lab] = {"AUC (recession days)": roc_auc_score(rec, df[m]), "mean in recessions": df.loc[rec == 1, m].mean(), "mean otherwise": df.loc[rec == 0, m].mean()}
        if m != "p_stress":
            lo, hi = block_bootstrap_auc_diff(rec.to_numpy(), df["p_stress"].to_numpy(), df[m].to_numpy(), yrs)
            nb[lab]["HMM AUC minus this: 90% CI (year-block bootstrap)"] = f"[{lo:+.3f}, {hi:+.3f}]"
    nt = pd.DataFrame(nb).T
    nt.index.name = f"NBER recession days in test: {int(rec.sum())} of {len(rec)} (3 recessions)"

    # 4. novelty
    df["nov_q"] = pd.qcut(df.novelty_pct.rank(method="first"), 5, labels=["Q1 (most familiar)", "Q2", "Q3", "Q4", "Q5 (most novel)"])
    df["gjr_qlike"] = qlike(df.rv_real, df.gjr)
    df["sr_qlike"] = qlike(df.rv_real, df.state_reader)
    nxt = r.shift(-1).reindex(df.index)
    df["gjr_1pct_breach"] = nxt <= -2.326 * df.gjr_1d_sd
    novt = df.groupby("nov_q", observed=True).agg(
        novelty_pct=("novelty_pct", "median"), mean_abs_next_day_return_pct=("abs_r1", lambda s: 100 * s.mean()),
        realized_vol_next_21d=("rv_real", lambda s: np.sqrt(s).mean()), gjr_qlike=("gjr_qlike", "mean"),
        state_reader_qlike=("sr_qlike", "mean"), gjr_normal_1pct_var_breach=("gjr_1pct_breach", "mean"),
        drawdown_event_rate=("dd_event", "mean"))
    novt.index.name = "novelty quintile"

    text = [
        f"# State Reader benchmark: S&P 500, {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days, refitted every January)",
        "",
        "## 1. Next-21-day volatility (QLIKE; lower is better)",
        md_table(vt, "{:.4f}"),
        "",
        "Diebold-Mariano on QLIKE, State Reader vs each model (negative = State Reader better; Newey-West lag 21):",
        "",
        "| vs | DM stat | p-value |", "|---|---|---|",
    ] + [f"| {k} | {v[0]:.2f} | {v[1]:.3g} |" for k, v in dm.items()] + [
        "",
        "Diebold-Mariano on MSE of log volatility:",
        "",
        "| vs | DM stat | p-value |", "|---|---|---|",
    ] + [f"| {k} | {v[0]:.2f} | {v[1]:.3g} |" for k, v in dm_mse.items()] + [
        "",
        "## 2. Drawdown early warning",
        md_table(wt, "{:.4f}"),
        "",
        "## 3. Regime reading vs NBER recessions (public label)",
        md_table(nt, "{:.3f}"),
        "",
        "## 4. Novelty: are unusual days really different?",
        md_table(novt, "{:.4f}"),
    ]
    (RESULTS / "state_reader.md").write_text("\n".join(text) + "\n")
    save_json({"vol": vt.to_dict(orient="index"), "dm": dm, "dm_log_mse": dm_mse, "warning": wt.to_dict(orient="index"),
               "nber": nt.to_dict(orient="index"), "novelty": novt.to_dict(orient="index")}, "state_reader.json")
    print("\n".join(text))


if __name__ == "__main__":
    main()
