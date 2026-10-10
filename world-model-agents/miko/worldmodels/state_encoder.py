"""State Encoder world model: where are we now?

Encodes the market into a state with three readings (brief, section 02):
- regime: a Gaussian hidden Markov model over state features, read with *filtered*
  probabilities (forward pass only), so today's regime never uses tomorrow's data;
- closest past periods: nearest neighbours in standardised feature space, restricted to
  dates whose aftermath had fully happened before the run date;
- novelty: how far today sits from its nearest past states, as a percentile of how far
  past days sat from theirs.

It also carries a small learned forward-risk head (ridge / logistic regression from the
state to next-month volatility and drawdown odds), trained only on windows closed before
the cutoff.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ..core.guards import MissingInputError, check_cutoff, to_timestamp

REGIME_NAMES = {
    2: ["calm", "stressed"],
    3: ["calm", "normal", "stressed"],
    4: ["calm", "normal", "elevated", "crisis"],
}

# feature groups get equal total weight in the analog distance
GROUPS = {
    "vol": ["rv5", "rv21", "rv63", "vix", "oilrv21"],
    "trend": ["ret21", "ret63", "ret252", "oil63"],
    "drawdown": ["dd252"],
    "structure": ["corr63", "disp21", "vrp"],
}


@dataclass
class StateEncoder:
    n_regimes: int = 4
    regime_features: tuple[str, ...] = ("rv21", "rv63", "ret63", "dd252", "vix", "corr63")
    k_analogs: int = 10
    followed_window: int = 63  # analog's aftermath must be complete before the run date
    min_separation: int = 42  # analogs at least this many trading days apart
    novelty_k: int = 20
    novelty_gap: int = 21
    drawdown_threshold: float = 0.07
    vol_head_features: tuple[str, ...] = ("rv5", "rv21", "rv63", "vix")
    dd_head_features: tuple[str, ...] = ("vix", "rv21", "dd252", "ret63", "corr63")
    seed: int = 0
    name: str = "state-encoder"
    version: str = "v1"
    cutoff: pd.Timestamp | None = field(default=None, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, features: pd.DataFrame, cutoff, targets: pd.DataFrame | None = None) -> "StateEncoder":
        from hmmlearn.hmm import GaussianHMM
        from sklearn.linear_model import LinearRegression, LogisticRegression

        cut = to_timestamp(cutoff)
        F = features.loc[:cut].dropna()
        if len(F) < 500:
            raise MissingInputError(f"state encoder needs >= 500 days of features before {cut.date()}, got {len(F)}")
        self.cutoff = pd.Timestamp(F.index[-1])
        self.columns = list(F.columns)
        self.mu = F.mean()
        self.sd = F.std().replace(0, 1.0)
        Z = self.transform(F)
        self._weights = self._feature_weights(self.columns)

        # regimes
        rf = [c for c in self.regime_features if c in self.columns]
        self._rf = rf
        X = Z[rf].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(n_components=self.n_regimes, covariance_type="full", n_iter=300, random_state=self.seed, min_covar=1e-3)
            hmm.fit(X)
        vol_col = rf.index("rv21") if "rv21" in rf else 0
        order = np.argsort(hmm.means_[:, vol_col])
        self._hmm_start = hmm.startprob_[order]
        self._hmm_trans = hmm.transmat_[np.ix_(order, order)]
        self._hmm_means = hmm.means_[order]
        self._hmm_covs = hmm.covars_[order]
        self.regime_names = REGIME_NAMES.get(self.n_regimes, [f"regime{i}" for i in range(self.n_regimes)])

        # novelty reference distribution (each past day vs its own past)
        self._hist_Z = Z
        self._hist_novelty = self._novelty_series(Z)

        # forward-risk head, trained only on windows that closed before the cutoff.
        # Kept deliberately small: HAR-style realized vols plus the VIX for next-month volatility,
        # five state features for drawdown odds. (A ridge on all 13 features overfits; see bench notes.)
        self.head = None
        if targets is not None:
            T = targets.reindex(Z.index)
            closed = Z.index[: max(0, len(Z) - 21)]  # target window ends by the cutoff
            ok = T.loc[closed].dropna().index
            if len(ok) > 250:
                vc = [c for c in self.vol_head_features if c in self.columns]
                dc = [c for c in self.dd_head_features if c in self.columns]
                Xv = Z.loc[ok, vc].to_numpy()
                y = T.loc[ok, "fwd_logrv21"].to_numpy()
                reg = LinearRegression().fit(Xv, y)
                rv = np.exp(2 * y)
                calib = float(np.mean(rv / np.exp(2 * reg.predict(Xv))))
                yb = (T.loc[ok, "fwd_mdd21"] <= -self.drawdown_threshold).astype(int).to_numpy()
                logit = None
                if 0 < yb.sum() < len(yb):
                    logit = LogisticRegression(C=1.0, max_iter=2000).fit(Z.loc[ok, dc].to_numpy(), yb)
                self.head = {"vol": reg, "vol_cols": vc, "calib": calib, "dd": logit, "dd_cols": dc, "n_train": int(len(ok))}
        return self

    # ------------------------------------------------------------- helpers
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features[self.columns] - self.mu) / self.sd

    @staticmethod
    def _feature_weights(columns: list[str]) -> np.ndarray:
        w = np.ones(len(columns))
        for g, cols in GROUPS.items():
            present = [c for c in cols if c in columns]
            for c in present:
                w[columns.index(c)] = 1.0 / len(present)
        return np.sqrt(w)

    def _dist(self, Z: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.sqrt((((Z - z) * self._weights) ** 2).sum(axis=1))

    def _novelty_series(self, Z: pd.DataFrame) -> pd.Series:
        X = (Z.to_numpy() * self._weights).astype(np.float32)
        n = len(X)
        k, gap = self.novelty_k, self.novelty_gap
        out = np.full(n, np.nan)
        sq = (X**2).sum(axis=1)
        start = k + gap + 250
        for a in range(start, n, 512):
            b = min(n, a + 512)
            D2 = sq[a:b, None] + sq[None, :b] - 2 * X[a:b] @ X[:b].T
            for i in range(a, b):
                past = D2[i - a, : i - gap]
                kk = np.partition(past, k)[:k]
                out[i] = np.sqrt(np.maximum(kk, 0)).mean()
        return pd.Series(out, index=Z.index).dropna()

    def assert_usable(self, as_of) -> None:
        if self.cutoff is None:
            raise MissingInputError("state encoder is not fitted")
        check_cutoff(f"{self.name} {self.version}", self.cutoff, as_of)

    # ---------------------------------------------------------- readings
    def filtered_regime_probs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Forward-filtered P(regime_t | data up to t). No smoothing, so no look-ahead."""
        cache = self.__dict__.setdefault("_probs_cache", {})
        ck = (len(features), features.index[0], features.index[-1], float(features[self._rf].iloc[-1].sum()))
        if ck in cache:
            return cache[ck].copy()
        out_df = self._filtered_regime_probs(features)
        if len(cache) > 8:
            cache.clear()
        cache[ck] = out_df
        return out_df.copy()

    def _filtered_regime_probs(self, features: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(features)[self._rf].to_numpy()
        K = self.n_regimes
        logB = np.column_stack(
            [multivariate_normal(self._hmm_means[j], self._hmm_covs[j], allow_singular=True).logpdf(X) for j in range(K)]
        )
        if logB.ndim == 1:
            logB = logB[None, :]
        logA = np.log(self._hmm_trans + 1e-300)
        alpha = np.log(self._hmm_start + 1e-300) + logB[0]
        alpha -= logsumexp(alpha)
        out = np.empty_like(logB)
        out[0] = alpha
        for t in range(1, len(X)):
            alpha = logsumexp(alpha[:, None] + logA, axis=0) + logB[t]
            alpha -= logsumexp(alpha)
            out[t] = alpha
        return pd.DataFrame(np.exp(out), index=features.index, columns=self.regime_names)

    def novelty(self, features_upto: pd.DataFrame) -> tuple[float, float]:
        """(novelty distance, percentile vs past days) for the last row of features_upto."""
        Z = self.transform(features_upto)
        X = Z.to_numpy()
        if len(X) < self.novelty_k + self.novelty_gap + 1:
            raise MissingInputError("not enough history to score novelty")
        d = self._dist(X[: -self.novelty_gap], X[-1])
        nov = float(np.sort(d)[: self.novelty_k].mean())
        ref = self._hist_novelty.loc[: Z.index[-1]]
        ref = ref.iloc[:-1] if len(ref) and ref.index[-1] == Z.index[-1] else ref
        pct = float((ref.to_numpy() < nov).mean()) if len(ref) else float("nan")
        return nov, pct

    def analogs(self, features_upto: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        """Closest past dates whose aftermath (followed_window days) closed before the last date."""
        k = k or self.k_analogs
        Z = self.transform(features_upto)
        X = Z.to_numpy()
        n = len(X)
        last_ok = n - 1 - self.followed_window
        if last_ok <= 0:
            raise MissingInputError("not enough history for analogs")
        d = self._dist(X[: last_ok + 1], X[-1])
        order = np.argsort(d)
        chosen: list[int] = []
        for i in order:
            if all(abs(i - j) >= self.min_separation for j in chosen):
                chosen.append(int(i))
            if len(chosen) == k:
                break
        return pd.DataFrame({"date": Z.index[chosen], "distance": d[chosen], "pos": chosen})

    def kernel_weights(self, Z_hist: np.ndarray, z: np.ndarray, ess_target: float = 750.0, floor: float = 0.3) -> np.ndarray:
        """Weights over past states for state-conditioned sampling: Gaussian kernel with a
        bandwidth chosen so the effective sample size is about ess_target, mixed with a
        uniform floor so every past day keeps some weight."""
        d = self._dist(Z_hist, z)
        n = len(d)
        target = min(ess_target, 0.5 * n)
        lo, hi = 1e-3, 50.0
        w = np.ones(n) / n
        for _ in range(40):
            h = np.sqrt(lo * hi)
            w = np.exp(-0.5 * (d / h) ** 2 + 0.5 * (d.min() / h) ** 2)
            w /= w.sum()
            ess = 1.0 / (w**2).sum()
            if ess < target:
                lo = h
            else:
                hi = h
        w = (1 - floor) * w + floor / n
        return w / w.sum()

    def forward_risk(self, features_upto: pd.DataFrame) -> dict[str, float]:
        """Next-21-day annualized volatility and P(max drawdown worse than the threshold)."""
        if not self.head:
            return {}
        z = self.transform(features_upto.iloc[[-1]])
        h = self.head
        var21 = h["calib"] * np.exp(2 * h["vol"].predict(z[h["vol_cols"]].to_numpy())[0])
        out = {"fwd_vol21": float(np.sqrt(var21))}
        if h["dd"] is not None:
            out["p_drawdown"] = float(h["dd"].predict_proba(z[h["dd_cols"]].to_numpy())[0, 1])
        return out

    def card(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "cutoff": str(self.cutoff.date()) if self.cutoff is not None else None,
            "features": self.columns,
            "regimes": self.regime_names,
            "intended_uses": ["regime reading", "analog retrieval", "novelty scoring", "state-conditioning for path generation"],
            "known_failure_modes": [
                "HMM regimes are fitted labels, not ground truth; boundaries move when refitted.",
                "Analogs are only as good as the features; a new kind of shock can look familiar.",
                "Forward-risk head is linear in the state and trained on 1990s-onward US data.",
            ],
        }
