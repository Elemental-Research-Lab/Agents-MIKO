"""Soak test: run the live service day by day over a past period, as if each close were arriving live.

    python scripts/replay_soak.py --start 2022-10-03 --end 2022-12-28

Each simulated evening the runtime pulls the new close, rebuilds and gates the models, swaps them
in, and every endpoint is called. Writes results/live_soak.md with refresh times, latencies and
any failures.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import logging

    logging.disable(logging.INFO)
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient

    from miko.live import NY, LiveFeed, PriceStore, ReplayProvider, _calendar
    from miko.service.app import create_app
    from miko.service.runtime import LiveRuntime

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-10-03")
    ap.add_argument("--end", default="2022-12-28")
    args = ap.parse_args()
    sessions = _calendar().sessions_in_range(args.start, args.end)
    d = Path(tempfile.mkdtemp(prefix="miko_soak_"))
    now = {"t": None}
    feed = LiveFeed(ReplayProvider(), PriceStore(d / "p.csv"), clock=lambda: now["t"])
    rt = LiveRuntime(feed, d / "ledger.jsonl", None)
    app = create_app(runtime=rt, api_key="k", background=False)
    H = {"X-API-Key": "k"}
    book = {"AAPL": 0.25, "JPM": 0.25, "XOM": 0.25, "PG": 0.25}
    calls = {
        "state": ("/v1/state", {}),
        "simulate": ("/v1/simulate", {"book": book}),
        "scenario": ("/v1/scenario", {"narrative": "S&P -10%, oil +20%", "book": book}),
        "plan": ("/v1/plan", {"universe": ["AAPL", "JPM", "XOM", "PG", "MSFT", "KO", "JNJ", "WMT"],
                              "goal": {"measure": "cvar95", "limit": 0.06, "max_weight": 0.25}}),
    }
    rows, failures = [], []
    now["t"] = pd.Timestamp(sessions[0]).tz_localize(NY).replace(hour=18).to_pydatetime()
    with TestClient(app) as c:
        for s in sessions:
            now["t"] = pd.Timestamp(s).tz_localize(NY).replace(hour=18).to_pydatetime()
            a = c.post("/admin/refresh", headers=H).json()
            row = {"session": str(pd.Timestamp(s).date()), "refresh_s": a["seconds"], "refresh": a["result"].split(" (")[0]}
            if not (a["result"].startswith("activated") or a["result"] == "no new data"):
                failures.append(f"{row['session']} refresh: {a['result']}")
            ready = c.get("/ready").json()
            if ready.get("as_of") != row["session"]:
                failures.append(f"{row['session']}: serving {ready.get('as_of')}")
            for name, (path, body) in calls.items():
                t0 = time.time()
                r = c.post(path, json=body, headers=H)
                row[f"{name}_s"] = round(time.time() - t0, 3)
                if r.status_code != 200:
                    failures.append(f"{row['session']} {name}: {r.status_code} {r.text[:120]}")
                elif r.json()["as_of"] != row["session"]:
                    failures.append(f"{row['session']} {name}: answered for {r.json()['as_of']}")
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    lat = df[[c for c in df.columns if c.endswith("_s")]].describe(percentiles=[0.5, 0.95]).T[["mean", "50%", "95%", "max"]]
    lat.index.name = "seconds"
    lines = [f"# Live soak test (replayed closes {sessions[0].date()} to {sessions[-1].date()}, {len(sessions)} sessions)", "",
             "Each simulated evening: pull the new close, rebuild and gate the models, hot-swap, then call every endpoint.", "",
             f"Failures: {len(failures)}", *[f"- {f}" for f in failures], "",
             "| step | mean | median | p95 | max |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {r['mean']:.2f} | {r['50%']:.2f} | {r['95%']:.2f} | {r['max']:.2f} |" for i, r in lat.iterrows()]
    lines += ["", "Times in seconds on 2 CPU cores. Refresh includes refitting every world model and running all gates.",
              "First call after a swap may include fitting models for a new asset set."]
    out = ROOT / "results" / "live_soak.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
