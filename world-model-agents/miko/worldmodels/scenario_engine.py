"""Scenario Engine world model: what if X happens? (brief, section 02)

Takes a few shocks (cumulative moves of some factors over a horizon) and fills in consistent
moves for every other factor, then turns them into full daily paths.

Method ("kernel"): look back at every past H-day window, weight each window by how close its
moves in the shocked factors were to the requested shocks, and fit a weighted local-linear
regression of the other factors on the shocked ones, shrunk toward the full-history fit. The
conditional distribution at the shock is the regression's intercept plus resampled residuals
from the closest windows. Because the weights concentrate on windows that looked like the
shock, crisis-time betas and correlations are used for crisis-sized shocks and calm-time ones
for calm shocks, which a single covariance matrix cannot do.

Baselines (for the benchmark and for transparency):
    zero      other factors do not move
    beta      CAPM: each factor moves beta x the S&P 500 shock
    gauss     Gaussian conditioning on the full-history covariance of H-day moves
    gauss1y   Gaussian conditioning on the trailing one-year covariance
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.guards import MissingInputError, ProposalRejected, to_timestamp

METHODS = ("kernel", "local", "slope", "stresscov", "gauss", "gauss1y", "beta", "zero")
WORLD_MODEL_VARIANTS = ("local", "slope", "stresscov")


@dataclass
class Conditional:
    factors: list[str]
    mean: np.ndarray  # cumulative log moves over the horizon, all factors (shocked ones fixed)
    cov: np.ndarray  # conditional covariance (zero rows/cols for shocked factors)
    shocked: list[str]
    method: str
    residuals: np.ndarray | None = None  # rows of other-factor residuals for resampling
    residual_w: np.ndarray | None = None
    ess: float | None = None
    inflation: float = 1.0

    def sd(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0))


@dataclass
class ScenarioEngine:
    horizon: int = 21
    ess_target: float = 150.0
    shrink: float = 0.5  # ridge pull of local betas toward full-history betas
    cov_shrink: float = 0.3
    market: str = "SP500"
    default_variant: str = "stresscov"  # which world-model variant "kernel" means; chosen on 2000-2007 crises
    name: str = "scenario-engine"
    version: str = "v1"
    cutoff: pd.Timestamp | None = field(default=None, init=False)

    def fit(self, returns: pd.DataFrame, cutoff) -> "ScenarioEngine":
        cut = to_timestamp(cutoff)
        r = returns.loc[:cut]
        if r.isna().any().any():
            raise MissingInputError("scenario engine: returns contain blanks")
        if len(r) < 500:
            raise MissingInputError("scenario engine needs >= 500 days of history")
        self.cutoff = pd.Timestamp(r.index[-1])
        self.factors = list(r.columns)
        self.daily = r.to_numpy()
        H = self.horizon
        cs = np.vstack([np.zeros(len(self.factors)), np.cumsum(self.daily, axis=0)])
        self.X = cs[H:] - cs[:-H]  # overlapping H-day cumulative moves, windows end <= cutoff
        self.mu = self.X.mean(axis=0)
        self.cov = np.cov(self.X.T)
        self.sd = np.sqrt(np.diag(self.cov))
        self.cov1y = np.cov(self.daily[-252:].T) * H
        self.mu1y = np.zeros(len(self.factors))
        return self

    # ------------------------------------------------------------------ checks
    def _idx(self, names) -> list[int]:
        out = []
        for n in names:
            if n not in self.factors:
                raise ProposalRejected(f"unknown factor '{n}'; known factors: {', '.join(self.factors)}")
            out.append(self.factors.index(n))
        return out

    def plausibility(self, shocks: dict[str, float]) -> dict[str, float]:
        """How extreme the joint shock is versus every past H-day window (percentile of Mahalanobis distance)."""
        S = self._idx(shocks)
        s = np.array([shocks[k] for k in shocks])
        C = self.cov[np.ix_(S, S)]
        Ci = np.linalg.pinv(C)
        dev = self.X[:, S] - self.mu[S]
        d_hist = np.sqrt(np.einsum("ij,jk,ik->i", dev, Ci, dev))
        d = float(np.sqrt((s - self.mu[S]) @ Ci @ (s - self.mu[S])))
        per = {k: float((self.X[:, i] <= v).mean()) for k, i, v in zip(shocks, S, s)}
        return {"distance": d, "percentile": float((d_hist < d).mean()), "max_hist_distance": float(d_hist.max()), "per_factor_percentile": per}

    # -------------------------------------------------------------- conditioning
    def condition(self, shocks: dict[str, float], method: str = "kernel") -> Conditional:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not shocks:
            raise MissingInputError("no shocks given")
        for k, v in shocks.items():
            if v is None or not np.isfinite(v):
                raise MissingInputError(f"shock for '{k}' is blank")
        S = self._idx(shocks)
        O = [i for i in range(len(self.factors)) if i not in S]
        s = np.array([shocks[self.factors[i]] for i in S])
        k = len(self.factors)
        mean = np.zeros(k)
        cov = np.zeros((k, k))
        mean[S] = s
        res, rw, ess = None, None, None

        if method == "zero":
            cov[np.ix_(O, O)] = np.diag(np.diag(self.cov)[O])
        elif method == "beta":
            m = self.factors.index(self.market)
            if m not in S:
                raise MissingInputError(f"beta method needs a shock to {self.market}")
            d = self.daily
            var_m = d[:, m].var()
            for i in O:
                b = np.cov(d[:, i], d[:, m])[0, 1] / var_m
                mean[i] = b * shocks[self.market]
                cov[i, i] = (d[:, i] - b * d[:, m]).var() * self.horizon
        elif method in ("gauss", "gauss1y"):
            mu, C = (self.mu, self.cov) if method == "gauss" else (self.mu1y, self.cov1y)
            Css = C[np.ix_(S, S)]
            Cos = C[np.ix_(O, S)]
            K = Cos @ np.linalg.pinv(Css)
            mean[O] = mu[O] + K @ (s - mu[S])
            cov[np.ix_(O, O)] = C[np.ix_(O, O)] - K @ Cos.T
        else:
            variant = self.default_variant if method == "kernel" else method
            Xs = self.X[:, S]
            Y = self.X[:, O]
            z = (Xs - s) / self.sd[S]
            w = self._bandwidth_weights((z**2).sum(axis=1))
            ess = float(1.0 / (w**2).sum())
            if variant == "local":
                mean[O], cov[np.ix_(O, O)], res, rw = self._local_linear(Xs, Y, s, w, S, O)
            elif variant == "slope":
                mean[O], cov[np.ix_(O, O)], res, rw = self._local_slopes(Xs, Y, s, w, S, O)
            elif variant == "stresscov":
                mean[O], cov[np.ix_(O, O)] = self._stress_cov(s, w, S, O)
            else:
                raise ValueError(f"unknown variant {variant}")

        infl = 1.0
        if method in ("kernel",) + WORLD_MODEL_VARIANTS:
            pl = self.plausibility(shocks)
            if pl["distance"] > pl["max_hist_distance"]:
                infl = (pl["distance"] / pl["max_hist_distance"]) ** 2
                cov[np.ix_(O, O)] *= infl
        return Conditional(self.factors, mean, cov, [self.factors[i] for i in S], method, res, rw, ess, infl)

    # ------------------------------------------------------------ world-model variants
    def _global_slopes(self, Xs, Y, S, O):
        return np.linalg.lstsq(np.column_stack([np.ones(len(Xs)), Xs - self.mu[S]]), Y - self.mu[O], rcond=None)[0][1:]

    def _local_linear(self, Xs, Y, s, w, S, O):
        """v1: weighted local-linear fit centred at the shock (local intercept and slopes)."""
        Xc = np.column_stack([np.ones(len(Xs)), Xs - s])
        Bg = self._global_slopes(Xs, Y, S, O)
        W = w[:, None]
        XtWX = Xc.T @ (Xc * W)
        lam = self.shrink * np.trace(XtWX[1:, 1:]) / max(1, len(S))
        P = np.zeros_like(XtWX)
        P[1:, 1:] = np.eye(len(S))
        prior = np.vstack([np.zeros((1, len(O))), Bg])
        coef = np.linalg.solve(XtWX + lam * P, Xc.T @ (Y * W) + lam * P @ prior)
        resid = Y - Xc @ coef
        rc = resid - w @ resid
        Cl = (rc * W).T @ rc
        Cg = np.cov((Y - (self.mu[O] + (Xs - self.mu[S]) @ Bg)).T)
        return coef[0], (1 - self.cov_shrink) * Cl + self.cov_shrink * Cg, resid, w

    def _local_slopes(self, Xs, Y, s, w, S, O):
        """v2: slopes from windows like the shock, intercept anchored at the full-history means
        (what a stock did beyond its beta in one past crisis is not carried into the next)."""
        Bg = self._global_slopes(Xs, Y, S, O)
        ms, mo = w @ Xs, w @ Y
        Xc, Yc = Xs - ms, Y - mo
        W = w[:, None]
        XtWX = Xc.T @ (Xc * W)
        lam = self.shrink * np.trace(XtWX) / max(1, len(S))
        B = np.linalg.solve(XtWX + lam * np.eye(len(S)), Xc.T @ (Yc * W) + lam * Bg)
        mean = self.mu[O] + (s - self.mu[S]) @ B
        resid = Yc - Xc @ B
        Cl = (resid * W).T @ resid
        Cg = np.cov((Y - (self.mu[O] + (Xs - self.mu[S]) @ Bg)).T)
        return mean, (1 - self.cov_shrink) * Cl + self.cov_shrink * Cg, resid - w @ resid, w

    def _stress_cov(self, s, w, S, O):
        """v3: covariance of *daily* returns on the days inside windows like the shock (many more
        observations than whole windows), shrunk toward the full-history daily covariance, scaled
        to the horizon; Gaussian conditioning around the full-history means."""
        H = self.horizon
        n_days = len(self.daily)
        wd = np.zeros(n_days)
        # window t covers days t .. t+H-1
        cw = np.concatenate([[0.0], np.cumsum(w)])
        for d in range(n_days):
            lo, hi = max(0, d - H + 1), min(d, len(w) - 1)
            if hi >= lo:
                wd[d] = cw[hi + 1] - cw[lo]
        wd /= wd.sum()
        m = wd @ self.daily
        dc = self.daily - m
        Cs = (dc * wd[:, None]).T @ dc * H
        Cf = np.cov(self.daily.T) * H
        C = (1 - self.cov_shrink) * Cs + self.cov_shrink * Cf
        K = C[np.ix_(O, S)] @ np.linalg.pinv(C[np.ix_(S, S)])
        mean = self.mu[O] + K @ (s - self.mu[S])
        cov = C[np.ix_(O, O)] - K @ C[np.ix_(O, S)].T
        return mean, cov

    def _bandwidth_weights(self, d2: np.ndarray) -> np.ndarray:
        target = min(self.ess_target, 0.25 * len(d2))
        lo, hi = 1e-3, 1e3
        w = np.ones_like(d2) / len(d2)
        for _ in range(60):
            h = np.sqrt(lo * hi)
            a = -0.5 * d2 / h**2
            w = np.exp(a - a.max())
            w /= w.sum()
            if 1.0 / (w**2).sum() < target:
                lo = h
            else:
                hi = h
        return w

    # -------------------------------------------------------------------- paths
    def sample_terminal(self, cond: Conditional, n: int, seed: int) -> np.ndarray:
        """Cumulative H-day log moves for every factor; shocked factors hit their targets exactly."""
        rng = np.random.default_rng(seed)
        k = len(self.factors)
        S = [self.factors.index(f) for f in cond.shocked]
        O = [i for i in range(k) if i not in S]
        out = np.tile(cond.mean, (n, 1))
        if cond.residuals is not None:
            rows = rng.choice(len(cond.residuals), size=n, p=cond.residual_w)
            out[:, O] += cond.residuals[rows] * np.sqrt(cond.inflation)
        else:
            C = cond.cov[np.ix_(O, O)]
            L = np.linalg.cholesky(C + 1e-10 * np.eye(len(O)))
            out[:, O] += rng.standard_normal((n, len(O))) @ L.T
        return out

    def paths(self, cond: Conditional, n: int, seed: int, base: np.ndarray | None = None) -> np.ndarray:
        """Daily paths (n, H, k). `base` gives the day-to-day texture (e.g. world-model paths);
        a bridge adjustment makes every path's cumulative move equal its sampled terminal move."""
        H, k = self.horizon, len(self.factors)
        term = self.sample_terminal(cond, n, seed)
        if base is None:
            rng = np.random.default_rng(seed + 7)
            Ld = np.linalg.cholesky(np.cov(self.daily[-252:].T) + 1e-12 * np.eye(k))
            base = rng.standard_normal((n, H, k)) @ Ld.T
        if base.shape != (n, H, k):
            raise MissingInputError(f"base paths must have shape {(n, H, k)}, got {base.shape}")
        adj = (term - base.sum(axis=1)) / H
        return base + adj[:, None, :]

    def card(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "cutoff": str(self.cutoff.date()) if self.cutoff is not None else None,
            "horizon_days": self.horizon,
            "factors": self.factors,
            "method": "kernel-weighted local-linear conditioning on past H-day windows, residual resampling, bridge to daily paths",
            "intended_uses": ["stress tests", "what-ifs", "narrative scenarios"],
            "known_failure_modes": ["a scenario is an assumption you choose, not a prediction",
                                    "shocks beyond any past window are extrapolated; bands are widened and flagged",
                                    "only factors in the panel can be shocked or filled in (no macro series in v1)"],
        }
