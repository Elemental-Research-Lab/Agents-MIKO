# Running the agents live

The four world-model agents run as one HTTP service on live daily data. After every NYSE close
the service pulls the new closes, checks them, refits every world model, runs the model gates,
and only then switches to the new models. Requests are always answered from the latest
*completed* session that passed every check; an unfinished intraday bar is never used.

## Status

| Check | Where | Result |
|---|---|---|
| Unit, agent and planted-defect tests | `pytest -q` | 94 passed |
| Service end to end (every endpoint, auth, refusals, staleness, gate failure, hot swap, intraday, concurrency) | `tests/test_service.py` | passed |
| Day-by-day soak: 61 replayed sessions, refresh + all endpoints each evening | `scripts/replay_soak.py` → `results/live_soak.md` | 0 failures |
| Real HTTP server, clean non-editable install (what the Docker image runs) | `uvicorn miko.service.app:app` | passed |
| **Live Yahoo Finance feed** | `python scripts/live_smoke_test.py` | **must be run on a machine with internet** (the build machine had no access to market data). The Yahoo response parsing, retries and outage handling are tested offline against the real response layout |
| Docker image build | `docker compose up --build` | recipe written; build not run (no Docker daemon on the build machine) |

Run the live smoke test before calling it production-ready. It prints a PASS/FAIL line per check
and exits non-zero on any failure.

## Run it

```bash
pip install -e ".[live]"
python scripts/live_smoke_test.py                 # 1. prove the live feed + every agent work today
export MIKO_API_KEY=change-me
uvicorn miko.service.app:app --host 0.0.0.0 --port 8000   # 2. serve
# or
MIKO_API_KEY=change-me docker compose up --build -d
```

First start downloads about 35 years of daily closes (~1 minute) and fits the models (~5 s).
Until that finishes, `/ready` returns 503.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MIKO_PROVIDER` | `yahoo` | `yahoo` (live) or `replay` (research panel, for rehearsals) |
| `MIKO_DATA_DIR` | `./live_data` (`/data` in Docker) | price cache, trial ledger, model checkpoints |
| `MIKO_STOCKS` | the 20 research stocks | comma-separated tickers the service can price |
| `MIKO_REFRESH_SECONDS` | `1800` | how often to look for a new close |
| `MIKO_MAX_LAG` | `1` | sessions of staleness tolerated before `/ready` fails |
| `MIKO_API_KEY` | unset | if set, all `/v1` and `/admin` calls need header `X-API-Key` |
| `MIKO_MAX_CONCURRENT` | `2` | agent runs at once; others wait up to 60 s, then 503 |
| `MIKO_REPLAY_NOW` | `2022-12-28T18:00` | replay only: the simulated clock |

## Endpoints

| Method | Path | Agent / purpose |
|---|---|---|
| GET | `/health` | process is up (use for liveness) |
| GET | `/ready` | 200 only if data is fresh and today's models passed the gates (use for readiness) |
| GET | `/status` | as-of date, data snapshot, quality report, every gate result, last refresh |
| POST | `/v1/state` | State Reader |
| POST | `/v1/simulate` | Simulator |
| POST | `/v1/scenario` | Scenario Generator (shocks or narrative) |
| POST | `/v1/scenario/replay` | Scenario Generator, replay a past window with changes |
| POST | `/v1/plan` | Planner, weights for a risk goal (default `mode: "scale"`) |
| POST | `/v1/plan/policy` | Planner, train an exposure policy |
| POST | `/admin/refresh` | force a refresh now (`?force=true` refits even without new data) |

Interactive docs at `/docs`. Examples:

```bash
H='X-API-Key: change-me'
curl -s -XPOST localhost:8000/v1/state -H "$H" -H 'content-type: application/json' -d '{}'
curl -s -XPOST localhost:8000/v1/simulate -H "$H" -H 'content-type: application/json' \
  -d '{"book": {"AAPL": 0.5, "MSFT": 0.5}, "orders": {"AAPL": 250000}, "adv": {"AAPL": 1.2e10}}'
curl -s -XPOST localhost:8000/v1/scenario -H "$H" -H 'content-type: application/json' \
  -d '{"narrative": "S&P -20%, oil +40%, VIX to 45", "book": {"AAPL": 0.5, "XOM": 0.5}}'
curl -s -XPOST localhost:8000/v1/plan -H "$H" -H 'content-type: application/json' \
  -d '{"universe": ["AAPL","MSFT","JPM","XOM","PG","KO"], "goal_text": "keep monthly CVaR under 6%, max 25% per name"}'
```

Every answer is a MIKO result: numbers with SIM/DATA/CALC badges, a count read-out, provenance
(model versions and cutoffs, seed, data snapshot) and a trial id in the ledger.

## What the service refuses, and how

| Situation | Response |
|---|---|
| Blank or unknown input, unknown asset, impact without ADV, infeasible goal, LLM proposal that invents a number or factor | 422 with the reason |
| A model asked about a period before its cutoff | 409 |
| Data more than `MIKO_MAX_LAG` sessions old, or no snapshot has passed the gates | 503, `/ready` explains |
| Too many concurrent runs | 503 "busy", retry |
| Bad print (S&P 500 moving more than 25% in a session), missing SP500/VIX/WTI, gaps over 3 sessions, fewer than 5 usable stocks | the day is rejected; the last good snapshot keeps serving until it is too stale |
| A stock without a recent close or with under 750 sessions of history | excluded that day, listed in `/status` warnings |
| A failed model gate | the new models are not activated; the previous ones keep serving |

## Model gates (run on every refresh)

1. Regime probabilities sum to 1; novelty percentile in [0, 1]; forward volatility between 2% and 300%.
2. S&P 500 path generator: 1-day simulated volatility within 0.33-3x of the last 21 sessions'; causal 1-day 99% VaR backtest over the last 250 sessions passes Kupiec (p ≥ 0.001) with at most 4% breaches.
3. Scenario engine: canned paths hit an S&P -10% shock exactly, and at least 70% of stocks fill in as falling.

## Operations

- **Data:** Yahoo Finance via `yfinance`, adjusted daily closes, about 15 minutes delayed. The service waits 30 minutes after each close before trusting the bar, follows the NYSE calendar (holidays, early closes), and refetches a symbol's whole history when Yahoo back-adjusts it for a dividend or split. WTI is the front-month future (`CL=F`).
- **Swapping the feed:** implement `fetch(names, start, end)` (see `miko/live.py`) and pass it to `LiveFeed`. Nothing else changes.
- **Scaling:** one worker process holds the models in memory. For more throughput run several containers behind a load balancer; each refreshes independently from the same feed.
- **Monitoring:** alert on `/ready` != 200 for more than one refresh interval, and on `"result"` values other than `activated …` or `no new data` in the `refresh:` log lines.
- **Audit:** the trial ledger (`$MIKO_DATA_DIR/trial_ledger.jsonl`) records every agent run; checkpoints of each day's models are saved under `$MIKO_DATA_DIR/checkpoints/live/<snapshot>/`.

## Limits

- Daily models: "real time" means the latest completed close, not intraday ticks.
- Impact is estimated, not reactive, and needs ADV from the caller.
- The LLM layer defaults to the deterministic rule parser; set `MIKO_LLM=gemini|groq|ollama` with a key to use a model.
- Simulated results are not forecasts of returns, and nothing here places orders. Promotion to trading stays with Validate and a human approval, per the brief.
