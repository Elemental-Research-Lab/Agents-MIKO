"""World models: correctness, point-in-time behaviour and planted defects."""
import warnings

import numpy as np
import pandas as pd
import pytest

from miko import data as D
from miko.core.guards import MissingInputError, ProposalRejected
from miko.worldmodels.features import equity_state_features, forward_targets
from miko.worldmodels.impact import ImpactModel
from miko.worldmodels.paths import BlockBootstrap, GarchT, RandomWalk, StatePath, _fit_arch, garch_filter
from miko.worldmodels.scenario_engine import ScenarioEngine
from miko.worldmodels.state_encoder import StateEncoder

CUT = "2007-12-31"
AS_OF = "2008-10-10"


@pytest.fixture(scope="module")
def panel():
    R = D.return_panel()
    F = equity_state_features(R, D.price_panel(), D.stock_names())
    T = forward_targets(R["SP500"], D.price_panel()["SP500"], 21)
    return R, F, T


@pytest.fixture(scope="module")
def enc(panel):
    R, F, T = panel
    return StateEncoder().fit(F, CUT, T)


def test_features_are_trailing_only(panel):
    R, F, _ = panel
    # planted defect: corrupt everything after a date; features up to that date must not change
    d = pd.Timestamp("2005-06-30")
    R2 = R.copy()
    R2.loc[R2.index > d] = 0.5
    P2 = np.exp(R2.cumsum()) * D.price_panel().iloc[0]
    F2 = equity_state_features(R2, P2.reindex(R2.index), D.stock_names())
    common = F.loc[: d].index.intersection(F2.index)
    cols = [c for c in F.columns if c not in ("vix", "vrp", "dd252")]  # level-based columns use the rebuilt price index
    assert np.allclose(F.loc[common, cols], F2.loc[common, cols], equal_nan=True)


def test_garch_filter_matches_arch():
    from arch import arch_model

    x = 100 * D.return_panel()["SP500"].loc[:CUT].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, rescale=False).fit(disp="off")
    s = garch_filter(x, _fit_arch(x, 1, "normal"))
    assert np.max(np.abs(np.sqrt(s[:-1])[50:] / res.conditional_volatility[50:] - 1)) < 0.01


def test_regime_probabilities_are_filtered_not_smoothed(enc, panel):
    _, F, _ = panel
    p1 = enc.filtered_regime_probs(F.loc[:AS_OF])
    F2 = F.copy()
    F2.loc[F2.index > AS_OF] = F2.loc[F2.index > AS_OF] * 0 + 5  # planted: wild future states
    p2 = enc.filtered_regime_probs(F2.loc[:"2009-12-31"])
    assert np.allclose(p1.values, p2.loc[:AS_OF].values)


def test_analogs_aftermath_closed_before_run_date(enc, panel):
    _, F, _ = panel
    an = enc.analogs(F.loc[:AS_OF], 10)
    last_ok = F.loc[:AS_OF].index[-1 - enc.followed_window]
    assert (an["date"] <= last_ok).all()
    assert an["date"].diff().dropna().abs().min() > pd.Timedelta(days=1)


def test_encoder_refuses_run_before_cutoff(enc):
    from miko.core.guards import CutoffError

    with pytest.raises(CutoffError):
        enc.assert_usable("2005-01-03")


def test_statepath_is_point_in_time(enc, panel):
    R, F, _ = panel
    train = R[["SP500", "JPM"]].loc[:CUT]
    hist = R[["SP500", "JPM"]].loc[:AS_OF]
    m1 = StatePath(encoder=enc, features=F, kernel=True).fit(train)
    F2 = F.copy()
    F2.loc[F2.index > AS_OF, "vix"] = 10.0  # planted: absurd future VIX
    m2 = StatePath(encoder=enc, features=F2, kernel=True).fit(train)
    a = m1.sample(500, 10, 7, hist)
    b = m2.sample(500, 10, 7, hist)
    assert np.allclose(a, b)
    assert a.shape == (500, 10, 2) and np.isfinite(a).all()


def test_statepath_refuses_blank_latest_return(enc, panel):
    R, F, _ = panel
    m = StatePath(encoder=enc, features=F).fit(R[["SP500"]].loc[:CUT])
    hist = R[["SP500"]].loc[:AS_OF].copy()
    hist.iloc[-1, 0] = np.nan  # planted blank
    with pytest.raises(MissingInputError):
        m.sample(100, 5, 0, hist)


def test_bands_widen_in_a_crisis(enc, panel):
    R, F, _ = panel
    m = StatePath(encoder=enc, features=F, kernel=False).fit(R[["SP500"]].loc[:CUT])
    calm = m.sample(4000, 10, 0, R[["SP500"]].loc[:CUT]).sum(axis=1).std()
    crisis = m.sample(4000, 10, 0, R[["SP500"]].loc[:AS_OF]).sum(axis=1).std()
    assert crisis > 2 * calm


def test_path_model_blocked_before_its_cutoff(panel):
    from miko.core.guards import CutoffError

    R, _, _ = panel
    m = GarchT().fit(R[["SP500"]].loc[:CUT])
    with pytest.raises(CutoffError):
        m.sample(100, 5, 0, R[["SP500"]].loc[:"2005-06-30"])  # planted: run dated before the model's cutoff


@pytest.mark.parametrize("M", [RandomWalk, BlockBootstrap, GarchT])
def test_baselines_run(M, panel):
    R, _, _ = panel
    m = M().fit(R[["SP500", "XOM"]].loc[:CUT])
    s = m.sample(200, 5, 0, R[["SP500", "XOM"]].loc[:AS_OF])
    assert s.shape == (200, 5, 2) and np.isfinite(s).all()


def test_baseline_refuses_blank_training_data(panel):
    R, _, _ = panel
    tr = R[["SP500"]].loc[:CUT].copy()
    tr.iloc[100, 0] = np.nan  # planted blank
    with pytest.raises(MissingInputError):
        RandomWalk().fit(tr)


def test_scenario_paths_hit_shocks_exactly(panel):
    R, _, _ = panel
    eng = ScenarioEngine().fit(R, AS_OF)
    shocks = {"SP500": np.log(0.85), "WTI": np.log(1.3)}
    for method in ("kernel", "gauss", "beta"):
        c = eng.condition(shocks, method)
        p = eng.paths(c, 300, 0)
        cum = p.sum(axis=1)
        assert np.allclose(cum[:, eng.factors.index("SP500")], shocks["SP500"])
        assert np.allclose(cum[:, eng.factors.index("WTI")], shocks["WTI"])


def test_scenario_engine_refuses_bad_shocks(panel):
    R, _, _ = panel
    eng = ScenarioEngine().fit(R, AS_OF)
    with pytest.raises(ProposalRejected):
        eng.condition({"GOLD": -0.1})  # planted: factor the model does not know
    with pytest.raises(MissingInputError):
        eng.condition({"SP500": float("nan")})  # planted blank
    with pytest.raises(MissingInputError):
        eng.condition({})


def test_beyond_history_shock_widens_bands(panel):
    R, _, _ = panel
    eng = ScenarioEngine().fit(R, AS_OF)
    mild = eng.condition({"SP500": np.log(0.95)})
    wild = eng.condition({"SP500": np.log(0.30)})  # planted: far beyond any past month
    assert mild.inflation == 1.0 and wild.inflation > 1.0


def test_impact_model():
    im = ImpactModel()
    small = im.order_cost_frac(1e5, 1e9, 0.02)
    big = im.order_cost_frac(1e8, 1e9, 0.02)
    assert big > small > 0
    with pytest.raises(MissingInputError):
        im.trade_costs({"AAPL": 1e6}, None, {"AAPL": 0.02})  # planted: no ADV
    with pytest.raises(MissingInputError):
        im.order_cost_frac(1e6, float("nan"), 0.02)
