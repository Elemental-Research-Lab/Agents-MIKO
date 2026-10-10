"""The HTTP service end to end: live refresh, gates, hot swap, every endpoint, refusals, staleness, auth, load."""
import concurrent.futures as cf
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from miko.live import NY, LiveFeed, PriceStore, ReplayProvider
from miko.service import runtime as runtime_mod
from miko.service.app import create_app
from miko.service.gates import Check
from miko.service.runtime import LiveRuntime


class Clock:
    def __init__(self, s):
        self.set(s)

    def set(self, s):
        self.now = pd.Timestamp(s).tz_localize(NY).to_pydatetime()

    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    clock = Clock("2022-12-20 18:00")
    feed = LiveFeed(ReplayProvider(), PriceStore(d / "prices.csv"), clock=clock)
    rt = LiveRuntime(feed, d / "ledger.jsonl", None, refresh_seconds=3600)
    app = create_app(runtime=rt, api_key="secret", background=False, max_concurrent=2)
    with TestClient(app) as client:
        yield client, rt, clock


H = {"X-API-Key": "secret"}
BOOK = {"AAPL": 0.25, "JPM": 0.25, "XOM": 0.25, "PG": 0.25}


def test_health_ready_status(svc):
    c, rt, _ = svc
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/ready")
    assert r.status_code == 200 and r.json()["as_of"] == "2022-12-20"
    s = c.get("/status").json()
    assert all(g["passed"] for g in s["gates"]) and s["provider"] == "replay"


def test_auth_required(svc):
    c, _, _ = svc
    assert c.post("/v1/state", json={}).status_code == 401
    assert c.post("/v1/state", json={}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_state_endpoint(svc):
    c, _, _ = svc
    r = c.post("/v1/state", json={}, headers=H)
    assert r.status_code == 200
    j = r.json()
    assert j["as_of"] == "2022-12-20" and j["trial_id"] and j["numbers"]["novelty_pct"]["badge"] == "SIM"
    assert len(j["tables"]["closest_past_periods"]) == 10


def test_simulate_endpoint(svc):
    c, _, _ = svc
    r = c.post("/v1/simulate", json={"book": BOOK, "n_paths": 1000, "orders": {"AAPL": 100000}, "adv": {"AAPL": 1e10}}, headers=H)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["numbers"]["costs_bps"]["value"] > 0 and len(j["tables"]["models_side_by_side"]) == 4
    assert "Not a forecast" in " ".join(j["limits"])


def test_scenario_endpoints(svc):
    c, _, _ = svc
    r = c.post("/v1/scenario", json={"narrative": "S&P -15%, oil +30%, Fed on hold", "n_paths": 500, "book": BOOK}, headers=H)
    assert r.status_code == 200, r.text
    rows = {x["factor"]: x for x in r.json()["tables"]["filled_in"]}
    assert rows["SP500"]["median_%"] == pytest.approx(-15.0) and r.json()["warnings"]
    r = c.post("/v1/scenario", json={"shocks": [{"factor": "VIX", "kind": "level", "value": 40}], "n_paths": 500}, headers=H)
    assert r.status_code == 200
    r = c.post("/v1/scenario/replay", json={"window_start": "2020-02-24", "overrides": {"WTI": 20}, "n_paths": 500}, headers=H)
    assert r.status_code == 200 and r.json()["tables"]["actual_window_%"]


def test_plan_endpoints(svc):
    c, _, _ = svc
    u = ["AAPL", "JPM", "XOM", "PG", "MSFT", "KO", "JNJ", "WMT"]
    r = c.post("/v1/plan", json={"universe": u, "goal": {"measure": "cvar95", "limit": 0.06, "max_weight": 0.25}}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["numbers"]["risk_check"]["value"] <= 0.06 * 1.0001
    r = c.post("/v1/plan", json={"universe": u, "goal_text": "keep volatility below 10%, max 25% per name", "mode": "optimize"}, headers=H)
    assert r.status_code == 200, r.text
    r = c.post("/v1/plan/policy", json={"n_paths": 200}, headers=H)
    assert r.status_code == 200 and r.json()["numbers"]["n_candidates"]["value"] > 10


@pytest.mark.parametrize("path, body, code", [
    ("/v1/simulate", {"book": {"NOPE": 1.0}}, 422),                          # unknown asset
    ("/v1/simulate", {"book": {"AAPL": None}}, 422),                         # blank weight
    ("/v1/simulate", {"book": {"AAPL": 1.0}, "as_of": "2030-01-02"}, 422),   # future run date
    ("/v1/simulate", {"book": {"AAPL": 1.0}, "n_paths": 10**7}, 422),        # beyond limits
    ("/v1/simulate", {"book": {"AAPL": 1.0}, "orders": {"AAPL": 1e6}}, 422), # impact without ADV
    ("/v1/scenario", {}, 422),                                              # neither shocks nor narrative
    ("/v1/scenario", {"shocks": [{"factor": "GOLD", "value": 5}]}, 422),     # factor not in the world model
    ("/v1/plan", {"universe": ["AAPL", "MSFT"], "goal": {"measure": "cvar95", "limit": 0.001, "min_invested": 1.0}}, 422),  # infeasible
    ("/v1/plan", {"universe": ["AAPL", "MSFT"], "goal_text": "be safe"}, 422),  # no number to work with
])
def test_refusals(svc, path, body, code):
    c, _, _ = svc
    r = c.post(path, json=body, headers=H)
    assert r.status_code == code, r.text
    assert "detail" in r.json()


def test_concurrent_requests(svc):
    c, _, _ = svc
    body = {"book": BOOK, "n_paths": 500, "baselines": False}
    t0 = time.time()
    with cf.ThreadPoolExecutor(6) as ex:
        codes = list(ex.map(lambda i: c.post("/v1/simulate", json={**body, "seed": i}, headers=H).status_code, range(6)))
    assert codes == [200] * 6
    assert time.time() - t0 < 60


def test_gate_failure_keeps_last_good_models(svc, monkeypatch):
    c, rt, clock = svc
    before = rt.current.ctx.snapshot()
    clock.set("2022-12-21 18:00")
    monkeypatch.setattr(runtime_mod, "run_gates", lambda ctx: [Check("planted failing gate", False, 0.0, "must fail")])
    a = c.post("/admin/refresh", headers=H).json()
    assert a["result"] == "rejected: model gates"
    assert rt.current.ctx.snapshot() == before  # still serving the 12-20 models
    assert c.post("/v1/state", json={}, headers=H).json()["as_of"] == "2022-12-20"  # lag 1 is tolerated


def test_new_session_hot_swaps(svc):
    c, rt, clock = svc
    clock.set("2022-12-21 18:00")
    a = c.post("/admin/refresh", headers=H).json()
    assert a["result"].startswith("activated 2022-12-21"), a
    assert c.post("/v1/state", json={}, headers=H).json()["as_of"] == "2022-12-21"
    clock.set("2022-12-21 23:00")
    assert c.post("/admin/refresh", headers=H).json()["result"] == "no new data"


def test_intraday_uses_previous_close(svc):
    c, rt, clock = svc
    clock.set("2022-12-22 11:00")  # market open: the 12-22 bar is unfinished
    c.post("/admin/refresh", headers=H)
    assert c.get("/ready").json()["as_of"] == "2022-12-21"


def test_stale_feed_turns_service_unready(svc):
    c, rt, clock = svc
    clock.set("2023-01-10 18:00")  # planted: the feed stopped on 2022-12-28
    a = c.post("/admin/refresh", headers=H).json()
    assert a["result"] == "rejected: data quality"
    assert c.get("/ready").status_code == 503
    r = c.post("/v1/state", json={}, headers=H)
    assert r.status_code == 503 and "not ready" in r.text
