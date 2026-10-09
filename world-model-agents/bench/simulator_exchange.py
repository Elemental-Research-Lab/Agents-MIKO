"""Simulator benchmark 2: the public Exchange-rate benchmark (CRPS-sum).

Dataset and split: GluonTS `exchange_rate_nips` (8 daily exchange rates; 6,071 training days;
5 rolling test windows of 30 business days), the same split used by GP-Copula (Salinas et al.,
2019) and TimeGrad (Rasul et al., 2021).

Metric: CRPS-sum, computed with GluonTS's MultivariateEvaluator exactly as in the TimeGrad code:
quantiles 0.05..0.95, target_agg_funcs={"sum": np.sum}, metric m_sum_mean_wQuantileLoss,
100 sample paths per window. We repeat over 10 seeds and report mean and standard deviation.

Every model is fitted once on the training split; at each test window it conditions on the
observed history up to the window start (as the published models do with their context).

Run:  python bench/simulator_exchange.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, md_table, save_json

from miko.data import CACHE
from miko.worldmodels.features import generic_state_features
from miko.worldmodels.paths import BlockBootstrap, GarchT, HistoricalSim, RandomWalk, StatePath
from miko.worldmodels.state_encoder import StateEncoder

warnings.filterwarnings("ignore")

N_SAMPLES = 100
SEEDS = range(10)

# Published CRPS-sum on Exchange (Rasul et al. 2021, "Autoregressive Denoising Diffusion Models for
# Multivariate Probabilistic Time Series Forecasting", Table 2; lower is better).
PUBLISHED = {
    "VES": (0.005, 0.000),
    "VAR": (0.005, 0.000),
    "VAR-Lasso": (0.012, 0.0002),
    "GARCH": (0.023, 0.000),
    "KVAE": (0.014, 0.002),
    "Vec-LSTM ind-scaling": (0.008, 0.001),
    "Vec-LSTM lowrank-Copula": (0.007, 0.000),
    "GP scaling": (0.009, 0.000),
    "GP Copula": (0.007, 0.000),
    "Transformer MAF": (0.005, 0.003),
    "TimeGrad": (0.006, 0.001),
}
# Yan et al. 2021, "ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous
# Energy-based Generative Models", Table III (same split).
PUBLISHED_SCOREGRAD = {
    "ScoreGrad VP SDE": (0.006, 0.001),
    "ScoreGrad VE SDE": (0.007, 0.001),
    "ScoreGrad sub-VP SDE": (0.006, 0.001),
}


def load():
    from gluonts.dataset.repository import get_dataset

    ds = get_dataset("exchange_rate_nips", path=Path(CACHE) / "gluonts", regenerate=False)
    train = np.stack([np.asarray(e["target"], float) for e in ds.train], axis=1)
    test = list(ds.test)
    windows = []
    for w in range(5):
        block = test[w * 8 : (w + 1) * 8]
        windows.append(np.stack([np.asarray(e["target"], float) for e in block], axis=1))
    start = test[0]["start"]
    return train, windows, start, ds.metadata.prediction_length


def crps_sum(windows, samples, pred_len, start):
    from gluonts.evaluation import MultivariateEvaluator
    from gluonts.model.forecast import SampleForecast

    tss, fcs = [], []
    for full, smp in zip(windows, samples):
        idx = pd.period_range(start=start, periods=len(full), freq="B")
        tss.append(pd.DataFrame(full, index=idx))
        fcs.append(SampleForecast(samples=smp, start_date=idx[-pred_len]))
    ev = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}, num_workers=None)
    agg, _ = ev(iter(tss), iter(fcs), num_series=len(tss))
    return float(agg["m_sum_mean_wQuantileLoss"])


def own_crps_sum(windows, samples, pred_len):
    """Independent re-implementation (quantile-loss approximation of CRPS on the summed series)."""
    qs = (np.arange(20) / 20.0)[1:]
    num, den = 0.0, 0.0
    vals = []
    for q in qs:
        num = 0.0
        den = 0.0
        for full, smp in zip(windows, samples):
            y = full[-pred_len:].sum(axis=1)
            s = smp.sum(axis=2)
            qhat = np.quantile(s, q, axis=0)
            num += 2 * np.sum(np.abs((y - qhat) * ((y <= qhat) - q)))
            den += np.sum(np.abs(y))
        vals.append(num / den)
    return float(np.mean(vals))


def main():
    train, windows, start, H = load()
    lr_train = pd.DataFrame(np.diff(np.log(train), axis=0), index=pd.bdate_range("1990-01-02", periods=len(train) - 1))
    cols = [f"fx{i}" for i in range(train.shape[1])]
    lr_train.columns = cols
    sel = json.loads((RESULTS / "simulator_sp500_selected.json").read_text())["selected"] if (RESULTS / "simulator_sp500_selected.json").exists() else None

    def histories():
        out = []
        for full in windows:
            ctx = full[:-H]
            lr = pd.DataFrame(np.diff(np.log(ctx), axis=0), index=pd.bdate_range("1990-01-02", periods=len(ctx) - 1), columns=cols)
            out.append((lr, ctx[-1]))
        return out

    hist = histories()
    full_lr = hist[-1][0]
    F = generic_state_features(full_lr)  # trailing features only; encoder fitted on the training part
    enc = StateEncoder(regime_features=("rv21", "rv63", "absret63", "corr63")).fit(F, lr_train.index[-1])

    kernel, widen = True, 0.0
    if sel and sel.startswith("statepath["):
        kv = dict(part.split("=") for part in sel.strip("statepath[]").split(","))
        kernel, widen = bool(int(kv["kernel"])), float(kv["widen"])
    models = {
        "random-walk (Gaussian)": RandomWalk(),
        "historical-sim-500d": HistoricalSim(500),
        "block-bootstrap": BlockBootstrap(10),
        "garch-t": GarchT(),
        "gjr-garch-t": GarchT(asymmetric=True),
        "fhs (no state)": StatePath(widen=0.0),
        "statepath (world model)": StatePath(encoder=enc, features=F, kernel=kernel, widen=widen),
    }
    rows = {}
    for name, m in models.items():
        m.fit(lr_train)
        scores, own = [], []
        for seed in SEEDS:
            samples = []
            for w, (lr, last) in enumerate(hist):
                s = m.sample(N_SAMPLES, H, seed=1000 * seed + w, history=lr)
                samples.append(last * np.exp(np.cumsum(s, axis=1)))
            scores.append(crps_sum(windows, samples, H, start))
            own.append(own_crps_sum(windows, samples, H))
        rows[name] = {"CRPS-sum mean": np.mean(scores), "CRPS-sum std": np.std(scores), "own implementation": np.mean(own)}
        print(name, rows[name], flush=True)
    ours = pd.DataFrame(rows).T
    ours.index.name = "model (this repo)"
    pub = pd.DataFrame({**PUBLISHED, **PUBLISHED_SCOREGRAD}, index=["CRPS-sum mean", "CRPS-sum std"]).T
    pub.index.name = "published (Rasul et al. 2021 Table 2; ScoreGrad rows: Yan et al. 2021 Table III)"
    best_pub = pub["CRPS-sum mean"].min()
    text = [
        "# Simulator benchmark: Exchange-rate CRPS-sum (GluonTS exchange_rate_nips)",
        "",
        f"5 rolling windows x 30 business days, {N_SAMPLES} sample paths, {len(SEEDS)} seeds. Lower is better.",
        f"World model settings carried over from the S&P 500 validation: kernel={kernel}, widen={widen}; the VIX head is off (no VIX for FX).",
        "",
        "## This repo",
        md_table(ours, "{:.5f}"),
        "",
        "## Published results on the same split",
        md_table(pub, "{:.3f}"),
        "",
        f"Best published mean: {best_pub:.3f}. Published values are rounded to 3 decimals.",
    ]
    (RESULTS / "simulator_exchange.md").write_text("\n".join(text) + "\n")
    save_json({"ours": ours.to_dict(orient="index"), "published": {**PUBLISHED, **PUBLISHED_SCOREGRAD}}, "simulator_exchange.json")
    print("\n".join(text))


if __name__ == "__main__":
    main()
