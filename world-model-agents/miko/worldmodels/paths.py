"""Path generators: what could happen next?

Every generator implements the same interface so agents and benchmarks can swap them:

    model.fit(train_returns)                       # learn parameters (data up to the cutoff)
    model.sample(n_paths, horizon, seed, history)  # -> array (n_paths, horizon, n_assets) of log returns

`history` is the return panel up to the run date; models condition on it (volatility filter,
market state) but never on anything after it.

Baselines (brief, Figure 1: always run alongside):
    RandomWalk        iid Gaussian with full-sample mean and covariance
    BlockBootstrap    stationary block bootstrap of historical return vectors
    GarchT            GARCH(1,1) with Student-t innovations per asset, Gaussian copula across assets
    FactorModel       market factor GARCH-t plus betas and Gaussian idiosyncratic noise

World model:
    StatePath         state-conditioned filtered historical simulation: GJR-GARCH filters per
                      asset, joint standardized residuals resampled with weights from the State
                      Encoder (days whose market state resembles today's count more), and bands
                      widened when today's state is novel.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..core.guards import MissingInputError, check_cutoff, to_timestamp


class PathModel:
    name = "base"
    version = "v1"
    cutoff: pd.Timestamp | None = None

    def fit(self, train: pd.DataFrame) -> "PathModel":
        raise NotImplementedError

    def sample(self, n_paths: int, horizon: int, seed: int, history: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _set_cutoff(self, train: pd.DataFrame) -> None:
        if len(train) < 250:
            raise MissingInputError(f"{self.name}: need >= 250 days to fit, got {len(train)}")
        if train.isna().any().any():
            raise MissingInputError(f"{self.name}: training data has blanks; refusing to fill them with zeros")
        self.cutoff = pd.Timestamp(train.index[-1])
        self.assets = list(train.columns)

    def _check_history(self, history: pd.DataFrame) -> np.ndarray:
        if list(history.columns) != self.assets:
            raise MissingInputError(f"{self.name}: history columns {list(history.columns)} != fitted {self.assets}")
        if self.cutoff is not None and len(history):
            check_cutoff(self.name, self.cutoff, history.index[-1])
        x = history.to_numpy(dtype=float)
        if np.isnan(x[-1]).any():
            raise MissingInputError(f"{self.name}: latest returns contain blanks")
        return x

    def card(self) -> dict:
        return {"name": self.name, "version": self.version, "cutoff": str(self.cutoff.date()) if self.cutoff is not None else None,
                "assets": getattr(self, "assets", None)}


# --------------------------------------------------------------------------- baselines
class RandomWalk(PathModel):
    name = "random-walk"

    def fit(self, train):
        self._set_cutoff(train)
        x = train.to_numpy()
        self.mu = x.mean(axis=0)
        self.cov = np.cov(x.T).reshape(len(self.assets), len(self.assets))
        return self

    def sample(self, n_paths, horizon, seed, history):
        self._check_history(history)
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(self.mu, self.cov, size=(n_paths, horizon), method="cholesky")
        return z.reshape(n_paths, horizon, len(self.assets))


class BlockBootstrap(PathModel):
    """Stationary block bootstrap (Politis-Romano) of all return vectors up to the run date."""

    name = "block-bootstrap"

    def __init__(self, mean_block: int = 10):
        self.mean_block = mean_block

    def fit(self, train):
        self._set_cutoff(train)
        return self

    def sample(self, n_paths, horizon, seed, history):
        x = self._check_history(history)
        rng = np.random.default_rng(seed)
        n = len(x)
        idx = np.empty((n_paths, horizon), dtype=int)
        idx[:, 0] = rng.integers(0, n, n_paths)
        restart = rng.random((n_paths, horizon)) < 1.0 / self.mean_block
        fresh = rng.integers(0, n, (n_paths, horizon))
        for t in range(1, horizon):
            idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
        return x[idx]


class HistoricalSim(PathModel):
    """Iid draws of return vectors from the most recent `window` days (the bank-standard HS VaR)."""

    name = "historical-sim"

    def __init__(self, window: int = 500):
        self.window = window

    def fit(self, train):
        self._set_cutoff(train)
        return self

    def sample(self, n_paths, horizon, seed, history):
        x = self._check_history(history)[-self.window :]
        rng = np.random.default_rng(seed)
        return x[rng.integers(0, len(x), (n_paths, horizon))]


def _fit_arch(r_pct: np.ndarray, o: int, dist: str):
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False)
    p = res.params
    out = {
        "mu": float(p["mu"]),
        "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]),
        "gamma": float(p.get("gamma[1]", 0.0)),
        "beta": float(p["beta[1]"]),
        "nu": float(p.get("nu", np.nan)),
    }
    # A degenerate optimum (seen on a pegged currency with a third of returns exactly zero)
    # is refused and refitted with Gaussian errors, then without asymmetry.
    sd = float(np.std(r_pct))
    bad = (not np.isfinite(out["mu"])) or abs(out["mu"]) > 5 * sd or out["omega"] > 10 * sd**2 or (dist == "t" and out["nu"] < 2.5)
    if bad:
        if dist != "normal":
            return _fit_arch(r_pct, o, "normal")
        if o:
            return _fit_arch(r_pct, 0, "normal")
        # last resort: RiskMetrics-like persistence around the sample variance
        out = {"mu": float(np.mean(r_pct)), "omega": 1e-3 * sd**2, "alpha": 0.06, "gamma": 0.0, "beta": 0.939, "nu": np.nan}
    # keep the process stationary; arch occasionally lands on the boundary
    pers = out["alpha"] + 0.5 * out["gamma"] + out["beta"]
    if pers >= 0.999:
        s = 0.998 / pers
        out["alpha"] *= s
        out["gamma"] *= s
        out["beta"] *= s
    return out


def garch_filter(r_pct: np.ndarray, prm: dict) -> np.ndarray:
    """Conditional variances sigma2[t] for t = 0..T (sigma2[T] is the one-step-ahead forecast)."""
    e = r_pct - prm["mu"]
    x = (prm["alpha"] + prm["gamma"] * (e < 0)) * e**2
    beta = prm["beta"]
    uncond = prm["omega"] / max(1e-8, 1 - prm["alpha"] - 0.5 * prm["gamma"] - beta)
    # sigma2[t] = omega/(1-beta) + beta^t*(s0 - omega/(1-beta)) + sum_j beta^j x[t-1-j]
    conv = lfilter([0.0, 1.0], [1.0, -beta], np.concatenate([x, [0.0]]))
    t = np.arange(len(conv))
    base = prm["omega"] / (1 - beta)
    sig2 = base + beta**t * (uncond - base) + conv
    return np.maximum(sig2, 1e-10)


class GarchT(PathModel):
    """GARCH(1,1)-t per asset; constant-correlation Gaussian copula across assets."""

    name = "garch-t"

    def __init__(self, asymmetric: bool = False):
        self.o = 1 if asymmetric else 0
        if asymmetric:
            self.name = "gjr-garch-t"

    def fit(self, train):
        self._set_cutoff(train)
        x = 100 * train.to_numpy()
        self.prm = [_fit_arch(x[:, j], self.o, "t") for j in range(x.shape[1])]
        z = np.column_stack([(x[:, j] - p["mu"]) / np.sqrt(garch_filter(x[:, j], p)[:-1]) for j, p in enumerate(self.prm)])
        self.corr = np.corrcoef(z.T).reshape(x.shape[1], x.shape[1]) if x.shape[1] > 1 else np.ones((1, 1))
        return self

    def sample(self, n_paths, horizon, seed, history):
        from scipy.stats import norm, t as student

        h = 100 * self._check_history(history)
        rng = np.random.default_rng(seed)
        k = len(self.assets)
        L = np.linalg.cholesky(self.corr + 1e-10 * np.eye(k))
        g = rng.standard_normal((n_paths, horizon, k)) @ L.T
        u = norm.cdf(g)
        eps = np.empty_like(u)
        for j, p in enumerate(self.prm):
            nu = p["nu"] if np.isfinite(p["nu"]) else 30.0
            eps[..., j] = student.ppf(u[..., j], nu) * np.sqrt((nu - 2) / nu)
        return _simulate_gjr(self.prm, h, eps) / 100.0


class FactorModel(PathModel):
    """Market factor (first column) follows GARCH-t; other assets = alpha + beta*market + iid noise."""

    name = "factor-model"

    def fit(self, train):
        self._set_cutoff(train)
        x = train.to_numpy()
        self.market = GarchT().fit(train.iloc[:, [0]])
        m = x[:, 0]
        X = np.column_stack([np.ones_like(m), m])
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        self.a, self.b = coef[0], coef[1]
        resid = x - X @ coef
        self.resid_sd = resid.std(axis=0)
        self.resid_sd[0] = 0.0
        return self

    def sample(self, n_paths, horizon, seed, history):
        self._check_history(history)
        mk = self.market.sample(n_paths, horizon, seed, history.iloc[:, [0]])[..., 0]
        rng = np.random.default_rng(seed + 1)
        noise = rng.standard_normal((n_paths, horizon, len(self.assets))) * self.resid_sd
        out = self.a + mk[..., None] * self.b + noise
        out[..., 0] = mk
        return out


def _simulate_gjr(prm: list[dict], hist_pct: np.ndarray, eps: np.ndarray, widen: float | np.ndarray = 1.0) -> np.ndarray:
    """Propagate GJR-GARCH variance along paths. eps: (n, H, k) standardized shocks. Returns pct returns."""
    n, H, k = eps.shape
    out = np.empty_like(eps)
    sig2 = np.empty((n, k))
    for j, p in enumerate(prm):
        sig2[:, j] = garch_filter(hist_pct[:, j], p)[-1]
    mu = np.array([p["mu"] for p in prm])
    om = np.array([p["omega"] for p in prm])
    al = np.array([p["alpha"] for p in prm])
    ga = np.array([p["gamma"] for p in prm])
    be = np.array([p["beta"] for p in prm])
    for t in range(H):
        e = np.sqrt(sig2) * widen * eps[:, t, :]
        out[:, t, :] = mu + e
        sig2 = om + (al + ga * (e < 0)) * e**2 + be * sig2
    return out


# --------------------------------------------------------------------------- world model
def gjr_horizon_var(sig2_next: np.ndarray, prm: dict, H: int) -> np.ndarray:
    """Closed-form GJR-GARCH forecast of summed variance over the next H days."""
    phi = prm["alpha"] + 0.5 * prm["gamma"] + prm["beta"]
    vl = prm["omega"] / max(1e-8, 1 - phi)
    if abs(1 - phi) < 1e-8:
        return H * sig2_next
    return H * vl + (sig2_next - vl) * (1 - phi**H) / (1 - phi)


@dataclass
class StatePath(PathModel):
    """State-conditioned filtered historical simulation (the MIKO path generator stand-in).

    Three parts, each switchable for ablations:
    1. GJR-GARCH(1,1) volatility filter per asset (QMLE), propagated along each path.
    2. Joint standardized residuals resampled with State Encoder kernel weights, so the
       *shape* of shocks (tails, cross-asset co-movement) follows days whose market state
       resembles today's. Weighted residuals are re-standardized so the kernel never moves
       the volatility level by itself.
    3. A learned state head: a regression from (GARCH horizon forecast, a few state features;
       by default the VIX) to the realized volatility that followed, trained only on windows
       closed before the cutoff (a GARCH-X / HAR-X style correction). It rescales each asset's
       paths so their horizon variance matches the state's forecast. Using all 13 state
       features overfits (see bench notes), so the head stays small.
    Plus band widening when today's state is novel.

    encoder/features: a fitted StateEncoder and the full state-feature table. Without them the
    model is plain filtered historical simulation.
    """

    encoder: object | None = None
    features: pd.DataFrame | None = None
    kernel: bool = True
    vol_head: bool = True
    head_features: tuple[str, ...] = ("vix",)
    ess_target: float = 1500.0
    floor: float = 0.3
    widen: float = 0.25  # extra band width at maximum novelty
    scale_clip: tuple[float, float] = (0.5, 2.0)
    name: str = "statepath"
    version: str = "v2"
    last_novelty: float | None = field(default=None, init=False)
    last_scale: np.ndarray | None = field(default=None, init=False)

    @property
    def conditioned(self) -> bool:
        return self.encoder is not None and self.features is not None

    def fit(self, train):
        self._set_cutoff(train)
        x = 100 * train.to_numpy()
        self._x = x
        self.prm = [_fit_arch(x[:, j], 1, "normal") for j in range(x.shape[1])]
        self._sig2 = [garch_filter(x[:, j], p) for j, p in enumerate(self.prm)]
        self.z = np.column_stack([(x[:, j] - p["mu"]) / np.sqrt(self._sig2[j][:-1]) for j, p in enumerate(self.prm)])
        self.z_dates = train.index
        self._heads: dict[int, list] = {}
        if self.conditioned:
            common = self.z_dates.intersection(self.features.index)
            self._pos = self.z_dates.get_indexer(common)
            self.z_state = self.z[self._pos]
            self.Z_hist = self.encoder.transform(self.features.loc[common]).to_numpy()
        else:
            self.z_state, self.Z_hist = self.z, None
        return self

    # learned state head -------------------------------------------------------------
    def _head_cols(self) -> list[int]:
        cols = list(self.encoder.columns)
        return [cols.index(c) for c in self.head_features if c in cols]

    def _head(self, H: int):
        """Per asset: log realized vol over the next H days ~ log GARCH forecast + state features.
        Fitted only on windows that closed inside the training data; a multiplicative constant
        makes the variance forecast unbiased in-sample (QLIKE-consistent)."""
        if H in self._heads:
            return self._heads[H]
        from sklearn.linear_model import LinearRegression

        heads = []
        n = len(self._x)
        pos = self._pos
        keep = pos + H < n
        ok = pos[keep]
        Zs = self.Z_hist[keep][:, self._head_cols()]
        for j, p in enumerate(self.prm):
            e2 = (self._x[:, j] - p["mu"]) ** 2
            csum = np.concatenate([[0.0], np.cumsum(e2)])
            rv = csum[ok + 1 + H] - csum[ok + 1]
            v_model = gjr_horizon_var(self._sig2[j][ok + 1], p, H)
            y = 0.5 * np.log(rv + 1e-8)
            X = np.column_stack([0.5 * np.log(v_model), Zs])
            reg = LinearRegression().fit(X, y)
            c = float(np.mean(rv / np.exp(2 * reg.predict(X))))
            heads.append((reg, c))
        self._heads[H] = heads
        return heads

    def head_scale(self, hist_pct: np.ndarray, as_of, H: int) -> np.ndarray:
        if not (self.conditioned and self.vol_head and self._head_cols()):
            return np.ones(len(self.prm))
        z_now = self.encoder.transform(self.features.loc[: to_timestamp(as_of)].iloc[[-1]]).to_numpy()[0][self._head_cols()]
        out = np.ones(len(self.prm))
        for j, ((reg, c), p) in enumerate(zip(self._head(H), self.prm)):
            sig2_next = garch_filter(hist_pct[:, j], p)[-1]
            v_model = gjr_horizon_var(np.array([sig2_next]), p, H)[0]
            yhat = reg.predict(np.concatenate([[0.5 * np.log(v_model)], z_now])[None, :])[0]
            v_state = c * np.exp(2 * yhat)
            out[j] = np.clip(np.sqrt(v_state / v_model), *self.scale_clip)
        return out

    # residual weights ----------------------------------------------------------------
    def weights(self, as_of) -> np.ndarray:
        if not (self.conditioned and self.kernel):
            return np.full(len(self.z_state), 1.0 / len(self.z_state))
        f = self.features.loc[: to_timestamp(as_of)]
        z_now = self.encoder.transform(f.iloc[[-1]]).to_numpy()[0]
        return self.encoder.kernel_weights(self.Z_hist, z_now, self.ess_target, self.floor)

    def novelty_multiplier(self, as_of) -> float:
        if not self.conditioned or self.widen <= 0:
            self.last_novelty = None
            return 1.0
        _, pct = self.encoder.novelty(self.features.loc[: to_timestamp(as_of)])
        self.last_novelty = pct
        return 1.0 + self.widen * max(0.0, (pct - 0.9) / 0.1)

    def sample(self, n_paths, horizon, seed, history):
        h = 100 * self._check_history(history)
        as_of = history.index[-1]
        rng = np.random.default_rng(seed)
        w = self.weights(as_of)
        idx = rng.choice(len(self.z_state), size=(n_paths, horizon), p=w)
        eps = self.z_state[idx]
        if self.conditioned and self.kernel:
            m = w @ self.z_state
            sd = np.sqrt(np.maximum(w @ (self.z_state - m) ** 2, 1e-8))
            eps = (eps - m) / sd
        paths = _simulate_gjr(self.prm, h, eps)
        mu = np.array([p["mu"] for p in self.prm])
        scale = self.head_scale(h, as_of, horizon) * self.novelty_multiplier(as_of)
        self.last_scale = scale
        paths = mu + (paths - mu) * scale
        return paths / 100.0

    def card(self) -> dict:
        c = super().card()
        c.update({
            "kind": "state-conditioned filtered historical simulation",
            "components": ["GJR-GARCH(1,1) volatility filter per asset (QMLE)",
                           "joint standardized residuals resampled with State Encoder kernel weights (shape only)" if self.kernel else "uniform residual resampling",
                           f"learned state head rescaling horizon variance (GARCH forecast + {list(self.head_features)})" if self.vol_head else "no state head",
                           f"novelty band widening up to x{1 + self.widen:.2f}"],
            "state_conditioned": self.conditioned,
            "intended_uses": ["risk bands", "stress paths", "portfolio and policy planning"],
            "known_failure_modes": ["daily model; long horizons compound filter error",
                                    "cannot produce residual shapes never seen in its training window",
                                    "not a forecast of returns: the drift is the fitted constant mean"],
        })
        return c


class Ensemble(PathModel):
    """Runs several generators and pools their paths; keeps per-model output for disagreement."""

    name = "ensemble"

    def __init__(self, models: list[PathModel]):
        self.models = models

    def fit(self, train):
        self._set_cutoff(train)
        for m in self.models:
            m.fit(train)
        return self

    def sample_each(self, n_paths, horizon, seed, history) -> dict[str, np.ndarray]:
        return {m.name: m.sample(n_paths, horizon, seed + i, history) for i, m in enumerate(self.models)}

    def sample(self, n_paths, horizon, seed, history):
        per = n_paths // len(self.models)
        parts = [m.sample(per, horizon, seed + i, history) for i, m in enumerate(self.models)]
        return np.concatenate(parts, axis=0)
