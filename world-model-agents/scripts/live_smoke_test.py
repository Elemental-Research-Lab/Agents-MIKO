"""Live smoke test: run on a machine with internet before calling the service production-ready.

    python scripts/live_smoke_test.py            # Yahoo Finance, today's latest close
    python scripts/live_smoke_test.py --replay   # offline rehearsal on the research panel

It pulls live closes, runs the data-quality report and the model gates, starts the HTTP app
in-process, calls every endpoint the way a client would, checks each answer, and prints a
PASS/FAIL table with latencies. Exit code 0 only if everything passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="use the bundled research panel instead of Yahoo")
    ap.add_argument("--data-dir", default=None, help="keep the price cache here (default: a temp dir)")
    args = ap.parse_args()

    import pandas as pd
    from fastapi.testclient import TestClient

    from miko.live import NY, LiveFeed, PriceStore, ReplayProvider, YahooProvider, last_complete_session, now_ny
    from miko.service.app import create_app
    from miko.service.runtime import LiveRuntime

    d = Path(args.data_dir or tempfile.mkdtemp(prefix="miko_live_"))
    if args.replay:
        fixed = pd.Timestamp("2022-12-28 18:00").tz_localize(NY).to_pydatetime()
        feed = LiveFeed(ReplayProvider(), PriceStore(d / "prices_replay.csv"), clock=lambda: fixed)
    else:
        feed = LiveFeed(YahooProvider(), PriceStore(d / "prices_yahoo.csv"))
    rt = LiveRuntime(feed, d / "ledger.jsonl", None)
    print(f"expected session: {last_complete_session(feed.clock()).date()}  (now {feed.clock().astimezone(NY):%Y-%m-%d %H:%M} NY)")
    print(f"data dir: {d}\n")

    results = []

    def check(name, ok, detail="", secs=None):
        results.append((name, bool(ok), detail, secs))

    app = create_app(runtime=rt, api_key="smoke", background=False)
    H = {"X-API-Key": "smoke"}
    with TestClient(app) as c:
        st = c.get("/status").json()
        q = st.get("quality") or (st.get("last_refresh") or {}).get("quality") or {}
        check("data quality", q.get("ok"), f"as of {q.get('as_of_data')}, lag {q.get('lag_sessions')}, warnings {len(q.get('warnings', []))}")
        for g in st.get("gates") or (st.get("last_refresh") or {}).get("gates") or []:
            check(f"gate: {g['name']}", g["passed"], str(g["value"]))
        r = c.get("/ready")
        check("ready", r.status_code == 200, json.dumps({k: r.json().get(k) for k in ("as_of", "lag_sessions", "reason")}))
        if r.status_code != 200:
            print(json.dumps(st.get("last_refresh"), indent=2, default=str))
        else:
            stocks = r.json()["assets"][3:]
            book = {s: 1.0 / min(5, len(stocks)) for s in stocks[:5]}
            calls = [
                ("State Reader", "/v1/state", {}, lambda j: 0 <= j["numbers"]["novelty_pct"]["value"] <= 1),
                ("Simulator", "/v1/simulate", {"book": book, "orders": {stocks[0]: 100_000}, "adv": {stocks[0]: 1e9}},
                 lambda j: j["numbers"]["cvar95"]["value"] >= j["numbers"]["var95"]["value"] > 0),
                ("Scenario (narrative)", "/v1/scenario", {"narrative": "S&P -15%, oil +25%, VIX to 40", "book": book},
                 lambda j: abs({x["factor"]: x for x in j["tables"]["filled_in"]}["SP500"]["median_%"] + 15) < 1e-6),
                ("Scenario (replay)", "/v1/scenario/replay", {"window_start": "2020-02-24", "overrides": {"WTI": 20}},
                 lambda j: bool(j["tables"]["actual_window_%"])),
                ("Planner (risk goal)", "/v1/plan", {"universe": stocks[:10], "goal_text": "keep monthly CVaR under 6%, max 20% per name"},
                 lambda j: j["numbers"]["risk_check"]["value"] <= 0.06 * 1.0001),
                ("Planner (policy)", "/v1/plan/policy", {"n_paths": 200}, lambda j: j["numbers"]["n_candidates"]["value"] > 0),
            ]
            for name, path, body, ok in calls:
                t0 = time.time()
                resp = c.post(path, json=body, headers=H)
                secs = time.time() - t0
                good = resp.status_code == 200 and ok(resp.json())
                detail = resp.json().get("readout", "")[:90] if resp.status_code == 200 else resp.text[:200]
                check(name, good, detail, secs)
            refuse = c.post("/v1/simulate", json={"book": {"NOT_A_TICKER": 1.0}}, headers=H)
            check("refuses unknown asset", refuse.status_code == 422, refuse.json().get("detail", "")[:90])

    w = max(len(n) for n, *_ in results)
    for name, ok, detail, secs in results:
        t = f"{secs:5.1f}s" if secs is not None else "      "
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {t}  {detail}")
    passed = all(ok for _, ok, *_ in results)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'} ({sum(ok for _, ok, *_ in results)}/{len(results)})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
