"""The four world-model agents: happy paths, trust rules and planted defects."""
import json

import numpy as np
import pytest

from miko import data as D
from miko.agents.planner import Goal, Planner, risk_of
from miko.agents.scenario_generator import ScenarioGenerator
from miko.agents.simulator import Simulator
from miko.agents.state_reader import StateReader
from miko.core.guards import CutoffError, InfeasibleGoalError, MissingInputError, ProposalRejected
from miko.core.llm import LLMClient

from conftest import AS_OF

BOOK = {"AAPL": 0.25, "JPM": 0.25, "XOM": 0.25, "PG": 0.25}


def _badges_ok(res):
    d = res.to_dict()
    assert all(n["badge"] in ("SIM", "DATA", "CALC") for n in d["numbers"].values())
    assert d["provenance"]["data_snapshot"].startswith("d-")
    assert res.trial_id and res.trial_count >= 1
    json.dumps(d)  # serialisable


# ------------------------------------------------------------------------- State Reader
def test_state_reader(ctx):
    res = StateReader(ctx).run(AS_OF)
    _badges_ok(res)
    probs = [r["probability"] for r in res.tables["regime_probabilities"]]
    assert abs(sum(probs) - 1) < 1e-6
    assert len(res.tables["closest_past_periods"]) == 10
    assert 0 <= res.numbers["novelty_pct"].value <= 1


def test_state_reader_flags_novel_crisis(ctx):
    res = StateReader(ctx).run("2008-10-10")
    assert res.numbers["novelty_pct"].value > 0.95
    assert res.warnings  # bands must be read wide
    assert res.numbers["p_drawdown"].value > 0.5


@pytest.mark.parametrize("bad", ["2022-12-25", "1985-01-02", None])
def test_state_reader_refuses_bad_dates(ctx, bad):
    with pytest.raises(MissingInputError):
        StateReader(ctx).run(bad)


# ---------------------------------------------------------------------------- Simulator
def test_simulator_book(ctx):
    res = Simulator(ctx).run(AS_OF, BOOK, n_paths=1000)
    _badges_ok(res)
    n = res.numbers
    assert n["n_loss"].value <= n["n_paths"].value
    assert n["cvar95"].value >= n["var95"].value > 0
    models = [r["model"] for r in res.tables["models_side_by_side"]]
    assert any("world model" in m for m in models) and "garch-t" in models  # baselines run alongside
    assert "Not a forecast" in " ".join(res.limits)
    assert res.readout.startswith("In ")


def test_simulator_costs_need_adv(ctx):
    sim = Simulator(ctx)
    with pytest.raises(MissingInputError):
        sim.run(AS_OF, BOOK, n_paths=200, orders={"AAPL": 1e6})  # planted: no ADV
    res = sim.run(AS_OF, BOOK, n_paths=200, orders={"AAPL": 1e6}, adv={"AAPL": 1e10})
    assert res.numbers["costs_bps"].value > 0


@pytest.mark.parametrize("book", [{"AAPL": float("nan")}, {"NOPE": 0.5}, {}, {"AAPL": 5.0}])
def test_simulator_refuses_bad_books(ctx, book):
    with pytest.raises(MissingInputError):
        Simulator(ctx).run(AS_OF, book, n_paths=200)


def test_simulator_blocks_model_trained_after_run_date(ctx):
    # planted: put a model fitted through 2022 in the cache under a 2015 key
    from miko.worldmodels.paths import RandomWalk

    m = RandomWalk().fit(ctx.returns_upto(AS_OF, ["AAPL"]))
    ctx._cache[("statepath", __import__("pandas").Timestamp("2015-06-30"), ("AAPL",))] = m
    with pytest.raises(CutoffError):
        Simulator(ctx).run("2015-06-30", {"AAPL": 1.0}, n_paths=200, baselines=())
    del ctx._cache[("statepath", __import__("pandas").Timestamp("2015-06-30"), ("AAPL",))]


def test_simulator_is_reproducible(ctx):
    a = Simulator(ctx).run(AS_OF, BOOK, n_paths=500, seed=3, baselines=())
    b = Simulator(ctx).run(AS_OF, BOOK, n_paths=500, seed=3, baselines=())
    assert a.digest() == b.digest()
    assert b.trial_count == a.trial_count + 1  # every run is a counted trial


# ------------------------------------------------------------------- Scenario Generator
def test_scenario_from_narrative(ctx):
    res = ScenarioGenerator(ctx).run(AS_OF, narrative="S&P -20%, oil +40%, VIX to 45, Fed on hold", n_paths=500, book=BOOK, texture="gauss")
    _badges_ok(res)
    rows = {r["factor"]: r for r in res.tables["filled_in"]}
    assert rows["SP500"]["median_%"] == pytest.approx(-20.0, abs=1e-6)
    assert rows["WTI"]["median_%"] == pytest.approx(40.0, abs=1e-6)
    assert rows["JPM"]["median_%"] < 0  # filled in consistently with a crash
    assert any("Fed on hold" in w for w in res.warnings)  # unknown parts are reported, not invented


def test_scenario_rejects_llm_invented_numbers(ctx):
    fake = LLMClient(provider="fake", scripted=['{"shocks": [{"factor": "SP500", "kind": "pct", "value": -35}]}'])
    with pytest.raises(ProposalRejected):
        ScenarioGenerator(ctx, fake).run(AS_OF, narrative="a bad month for stocks", n_paths=200)


def test_scenario_rejects_llm_unknown_factor(ctx):
    fake = LLMClient(provider="fake", scripted=['{"shocks": [{"factor": "GOLD", "kind": "pct", "value": 10}]}'])
    with pytest.raises(ProposalRejected):
        ScenarioGenerator(ctx, fake).run(AS_OF, narrative="gold +10%", n_paths=200)


def test_scenario_accepts_valid_llm_proposal(ctx):
    fake = LLMClient(provider="fake", scripted=['Here: {"shocks": [{"factor": "WTI", "kind": "pct", "value": -30}], "ignored": []}'])
    res = ScenarioGenerator(ctx, fake).run(AS_OF, narrative="oil collapses 30%", n_paths=300, texture="gauss")
    assert {r["factor"]: r for r in res.tables["filled_in"]}["WTI"]["median_%"] == pytest.approx(-30.0)


def test_scenario_replay(ctx):
    res = ScenarioGenerator(ctx).replay("2020-02-24", {"WTI": 20}, n_paths=300, texture="gauss")
    rows = {r["factor"]: r for r in res.tables["filled_in"]}
    actual = {r["factor"]: r["actual_%"] for r in res.tables["actual_window_%"]}
    assert rows["SP500"]["median_%"] == pytest.approx(actual["SP500"], abs=0.01)
    assert rows["WTI"]["median_%"] == pytest.approx(20.0)


# ------------------------------------------------------------------------------ Planner
UNIVERSE = D.stock_names()


@pytest.mark.parametrize("goal", [Goal("cvar95", 0.06, max_weight=0.2), Goal("vol", 0.10, max_weight=0.2)])
def test_planner_meets_goal_on_unseen_futures(ctx, goal):
    res = Planner(ctx).run(AS_OF, UNIVERSE, goal, n_paths=1000)
    _badges_ok(res)
    assert res.numbers["risk_check"].value <= goal.limit * 1.0001
    w = {r["asset"]: r["weight"] for r in res.tables["weights"]}
    assert max(v for k, v in w.items() if k != "cash") <= 0.2 + 1e-6
    assert abs(sum(w.values()) - 1) < 1e-3


def test_planner_scale_mode_keeps_composition(ctx):
    goal = Goal("cvar95", 0.06, max_weight=0.2)
    res = Planner(ctx).run(AS_OF, UNIVERSE, goal, n_paths=1000, mode="scale")
    w = [r["weight"] for r in res.tables["weights"] if r["asset"] != "cash"]
    assert max(w) - min(w) < 1e-9  # 1/N composition, only the size changes
    assert res.numbers["risk_check"].value <= goal.limit * 1.0001


def test_planner_from_text_goal(ctx):
    res = Planner(ctx).run(AS_OF, UNIVERSE, goal_text="keep monthly CVaR under 6%, max 20% per name", n_paths=1000)
    assert res.numbers["risk_limit"].value == pytest.approx(0.06)


def test_planner_refuses_infeasible_goal(ctx):
    with pytest.raises(InfeasibleGoalError):
        Planner(ctx).run(AS_OF, UNIVERSE, Goal("cvar95", 0.001, max_weight=0.2, min_invested=1.0), n_paths=500)
    with pytest.raises(InfeasibleGoalError):
        Goal("cvar95", -0.05)


def test_planner_rejects_llm_invented_goal(ctx):
    fake = LLMClient(provider="fake", scripted=['{"risk": {"measure": "cvar95", "limit_pct": 8}, "max_weight_pct": 20, "long_only": true}'])
    with pytest.raises(ProposalRejected):
        Planner(ctx, fake).run(AS_OF, UNIVERSE, goal_text="keep tail risk modest, max 20% per name")


def test_planner_trains_and_confirms_policy(ctx):
    res = Planner(ctx).train_policy("2007-12-31", n_paths=200, confirm_until="2010-12-31")
    assert res.numbers["n_candidates"].value > 10
    pols = {r["policy"]: r for r in res.tables["confirm_on_history"]}
    assert {"trained policy", "buy and hold"} <= set(pols)
    assert len(res.tables["holds_on_other_models"]) == 2
