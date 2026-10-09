"""Run all four world-model agents once and save their results, model cards and checkpoints.

    python scripts/demo.py [--as-of 2022-12-28]

Outputs:
    results/examples/*.json        full results (numbers with badges, tables, provenance, trial ids)
    results/examples/*.txt         plain-text result cards
    checkpoints/*.pkl              fitted world models for this run date (reload with pickle)
    checkpoints/cards/*.json       model cards
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miko import data as D  # noqa: E402
from miko.agents.context import AgentContext  # noqa: E402
from miko.agents.planner import Goal, Planner  # noqa: E402
from miko.agents.scenario_generator import ScenarioGenerator  # noqa: E402
from miko.agents.simulator import Simulator  # noqa: E402
from miko.agents.state_reader import StateReader  # noqa: E402
from miko.core.ledger import TrialLedger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2022-12-28")
    args = ap.parse_args()
    out = ROOT / "results" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(ledger=TrialLedger(ROOT / "results" / "trial_ledger.jsonl"), save_checkpoints=True)
    book = {"AAPL": 0.2, "MSFT": 0.2, "JPM": 0.2, "XOM": 0.2, "PG": 0.2}

    runs = {
        "state_reader": lambda: StateReader(ctx).run(args.as_of),
        "simulator": lambda: Simulator(ctx).run(args.as_of, book, orders={"AAPL": 250_000}, adv={"AAPL": 1.0e10}),
        "scenario_narrative": lambda: ScenarioGenerator(ctx).run(args.as_of, narrative="S&P -20%, oil +40%, VIX to 45, Fed on hold", book=book),
        "scenario_replay_covid_oil_up": lambda: ScenarioGenerator(ctx).replay("2020-02-24", {"WTI": 20}, book=book),
        "planner_cvar": lambda: Planner(ctx).run(args.as_of, D.stock_names(), goal_text="keep monthly CVaR under 6%, max 20% per name"),
        "planner_drawdown": lambda: Planner(ctx).run(args.as_of, D.stock_names(), Goal("cdar95", 0.08, max_weight=0.2)),
        "planner_policy_2008": lambda: Planner(ctx).train_policy("2007-12-31", confirm_until="2012-12-31", n_paths=1000),
    }
    for name, fn in runs.items():
        res = fn()
        res.save(out / f"{name}.json")
        (out / f"{name}.txt").write_text(res.card() + "\n")
        print(res.card(), "\n")

    cards = ROOT / "checkpoints" / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for key, obj in ctx._cache.items():
        if hasattr(obj, "card"):
            name = "_".join(str(getattr(k, "date", lambda: k)()) if hasattr(k, "date") else ("-".join(k) if isinstance(k, tuple) else str(k)) for k in key)
            (cards / f"{name}.json").write_text(json.dumps(obj.card(), indent=2, default=str))
    (cards / "data_card.json").write_text(json.dumps({**D.DATA_CARD, "snapshot": D.snapshot_id()}, indent=2))
    (cards / "impact_model.json").write_text(json.dumps(ctx.impact.card(), indent=2))


if __name__ == "__main__":
    main()
