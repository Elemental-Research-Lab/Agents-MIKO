"""HTTP API for the four world-model agents, running on live daily data.

    uvicorn miko.service.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health              process is up
    GET  /ready               200 only if live data is fresh and today's models passed their gates
    GET  /status              data as-of, snapshot, model gates, last refresh, assets
    POST /v1/state            State Reader
    POST /v1/simulate         Simulator (book, optional orders with ADV)
    POST /v1/scenario         Scenario Generator (shocks or narrative)
    POST /v1/scenario/replay  Scenario Generator, replay a past window with changes
    POST /v1/plan             Planner (weights for a risk goal)
    POST /v1/plan/policy      Planner (train an exposure policy)
    POST /admin/refresh       force a refresh (API key required)

Refusals (blank inputs, unknown assets, infeasible goals, LLM proposals that fail checks) are
422 with the reason; a model asked about a period it has seen is 409; stale data or a failed
gate is 503. Every successful run is logged in the trial ledger and carries its trial id.

Configuration (environment variables)
    MIKO_PROVIDER        yahoo (default) | replay
    MIKO_DATA_DIR        where the price cache, ledger and checkpoints live (default ./live_data)
    MIKO_STOCKS          comma-separated tickers (default: the 20 research stocks)
    MIKO_REFRESH_SECONDS how often to look for a new close (default 1800)
    MIKO_MAX_LAG         sessions of staleness tolerated before /ready fails (default 1)
    MIKO_API_KEY         if set, every /v1 and /admin call needs header X-API-Key
    MIKO_MAX_CONCURRENT  agent runs at once (default 2); extra requests wait up to 60 s
    MIKO_REPLAY_NOW      replay only: the simulated "now", e.g. 2022-12-28T18:00
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agents.context import AgentContext
from ..agents.planner import Goal, Planner
from ..agents.scenario_generator import ScenarioGenerator
from ..agents.simulator import Simulator
from ..agents.state_reader import StateReader
from ..core.guards import CutoffError, InfeasibleGoalError, LookAheadError, MikoError, MissingInputError, ProposalRejected
from ..live import DEFAULT_STOCKS, NY, LiveFeed, PriceStore, ReplayProvider, YahooProvider, now_ny
from .runtime import LiveRuntime

logging.basicConfig(level=os.environ.get("MIKO_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("miko.api")


# ------------------------------------------------------------------------- configuration
def runtime_from_env() -> LiveRuntime:
    data_dir = Path(os.environ.get("MIKO_DATA_DIR", "live_data"))
    stocks = [s.strip().upper() for s in os.environ.get("MIKO_STOCKS", ",".join(DEFAULT_STOCKS)).split(",") if s.strip()]
    provider_name = os.environ.get("MIKO_PROVIDER", "yahoo")
    clock = now_ny
    if provider_name == "replay":
        provider = ReplayProvider()
        fixed = pd.Timestamp(os.environ.get("MIKO_REPLAY_NOW", "2022-12-28T18:00")).tz_localize(NY).to_pydatetime()
        clock = lambda: fixed  # noqa: E731
    elif provider_name == "yahoo":
        provider = YahooProvider()
    else:
        raise ValueError(f"unknown MIKO_PROVIDER {provider_name!r}")
    max_lag = int(os.environ.get("MIKO_MAX_LAG", "1"))
    feed = LiveFeed(provider, PriceStore(data_dir / f"prices_{provider.name}.csv"), stocks, clock, max_lag)
    return LiveRuntime(feed, data_dir / "trial_ledger.jsonl", data_dir / "checkpoints",
                       float(os.environ.get("MIKO_REFRESH_SECONDS", "1800")), max_lag)


# ----------------------------------------------------------------------------- schemas
class StateRequest(BaseModel):
    as_of: date | None = None
    k_analogs: int = Field(10, ge=1, le=30)


class SimulateRequest(BaseModel):
    book: dict[str, float] = Field(..., description="weights of capital per asset, e.g. {'AAPL': 0.5, 'MSFT': 0.5}")
    as_of: date | None = None
    horizon: int = Field(21, ge=1, le=252)
    n_paths: int = Field(2000, ge=100, le=20000)
    seed: int = 0
    capital: float = Field(1_000_000.0, gt=0)
    orders: dict[str, float] | None = None
    adv: dict[str, float] | None = Field(None, description="average daily traded value per ordered asset, in currency")
    loss_threshold: float = Field(0.05, gt=0, lt=1)
    baselines: bool = True


class Shock(BaseModel):
    factor: str
    kind: Literal["pct", "level"] = "pct"
    value: float


class ScenarioRequest(BaseModel):
    shocks: list[Shock] | None = None
    narrative: str | None = Field(None, max_length=500)
    as_of: date | None = None
    horizon: int = Field(21, ge=5, le=63)
    n_paths: int = Field(2000, ge=100, le=10000)
    seed: int = 0
    book: dict[str, float] | None = None
    loss_threshold: float = Field(0.05, gt=0, lt=1)


class ReplayRequest(BaseModel):
    window_start: date
    overrides: dict[str, float] = Field(..., description="pct moves to impose instead of what happened, e.g. {'WTI': 20}")
    horizon: int = Field(21, ge=5, le=63)
    n_paths: int = Field(2000, ge=100, le=10000)
    seed: int = 0
    book: dict[str, float] | None = None


class GoalSpec(BaseModel):
    measure: Literal["cvar95", "vol", "cdar95"]
    limit: float = Field(..., gt=0, lt=1, description="e.g. 0.06 for 6%")
    max_weight: float = Field(1.0, gt=0, le=1)
    min_invested: float = Field(0.0, ge=0, le=1)


class PlanRequest(BaseModel):
    universe: list[str] = Field(..., min_length=2, max_length=40)
    goal: GoalSpec | None = None
    goal_text: str | None = Field(None, max_length=300)
    as_of: date | None = None
    mode: Literal["optimize", "scale"] = "scale"
    base: dict[str, float] | None = None
    current: dict[str, float] | None = None
    horizon: int = Field(21, ge=5, le=63)
    n_paths: int = Field(2000, ge=500, le=10000)
    seed: int = 0
    cost_bps: float = Field(10.0, ge=0, le=500)


class PolicyRequest(BaseModel):
    asset: str = "SP500"
    as_of: date | None = None
    target_vol: float = Field(0.12, gt=0, lt=1)
    dd_limit: float = Field(0.25, gt=0, lt=1)
    n_paths: int = Field(500, ge=100, le=2000)
    seed: int = 0
    confirm_until: date | None = None


# --------------------------------------------------------------------------------- app
def create_app(runtime: LiveRuntime | None = None, api_key: str | None = None, background: bool = True,
               max_concurrent: int | None = None) -> FastAPI:
    rt_holder: dict = {}
    key = api_key if api_key is not None else os.environ.get("MIKO_API_KEY")
    sem = threading.BoundedSemaphore(max_concurrent or int(os.environ.get("MIKO_MAX_CONCURRENT", "2")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = runtime or runtime_from_env()
        rt_holder["rt"] = rt
        if background:
            rt.start()
        else:
            rt.refresh()
        yield
        rt.stop()

    app = FastAPI(title="MIKO world-model agents", version="0.2.0", lifespan=lifespan)

    def rt() -> LiveRuntime:
        return rt_holder["rt"]

    def auth(x_api_key: str | None = Header(None)) -> None:
        if key and x_api_key != key:
            raise HTTPException(401, "missing or wrong X-API-Key")

    def live_ctx() -> AgentContext:
        ok, info = rt().readiness()
        ctx = rt().ctx()
        if not ok or ctx is None:
            raise HTTPException(503, {"error": "not ready", **info})
        return ctx

    def as_of_or_latest(ctx: AgentContext, d: date | None) -> pd.Timestamp:
        if d is None:
            return ctx.latest
        ts = pd.Timestamp(d)
        if ts > ctx.latest:
            raise MissingInputError(f"{ts.date()} is after the latest completed session {ctx.latest.date()}")
        return ts

    def run(fn):
        if not sem.acquire(timeout=60):
            raise HTTPException(503, "busy: too many agent runs in progress; retry shortly")
        try:
            t0 = time.time()
            res = fn()
            out = res.to_dict()
            out["latency_seconds"] = round(time.time() - t0, 3)
            return out
        finally:
            sem.release()

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        t0 = time.time()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        log.info("%s %s %s %.3fs rid=%s", request.method, request.url.path, resp.status_code, time.time() - t0, rid)
        return resp

    @app.exception_handler(MikoError)
    async def miko_error(request: Request, exc: MikoError):
        code = 409 if isinstance(exc, (CutoffError, LookAheadError)) else 422
        return JSONResponse(status_code=code, content={"error": type(exc).__name__, "detail": str(exc)})

    # ---------------------------------------------------------------- operations
    @app.get("/health")
    def health():
        return {"status": "ok", "time": datetime.now(tz=NY).isoformat()}

    @app.get("/ready")
    def ready():
        ok, info = rt().readiness()
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **info})

    @app.get("/status")
    def status():
        ok, info = rt().readiness()
        snap = rt().current
        return {"ready": ok, **info, "gates": snap.gates if snap else None,
                "quality": snap.quality.to_dict() if snap else None, "provider": rt().feed.provider.name}

    @app.post("/admin/refresh", dependencies=[Depends(auth)])
    def admin_refresh(force: bool = False):
        return rt().refresh(force=force)

    # -------------------------------------------------------------------- agents
    @app.post("/v1/state", dependencies=[Depends(auth)])
    def state(req: StateRequest):
        ctx = live_ctx()
        return run(lambda: StateReader(ctx).run(as_of_or_latest(ctx, req.as_of), req.k_analogs))

    @app.post("/v1/simulate", dependencies=[Depends(auth)])
    def simulate(req: SimulateRequest):
        ctx = live_ctx()
        b = ("garch-t", "block-bootstrap", "random-walk") if req.baselines else ()
        return run(lambda: Simulator(ctx).run(as_of_or_latest(ctx, req.as_of), req.book, req.horizon, req.n_paths, req.seed,
                                              req.capital, req.orders, req.adv, req.loss_threshold, b))

    @app.post("/v1/scenario", dependencies=[Depends(auth)])
    def scenario(req: ScenarioRequest):
        ctx = live_ctx()
        if (req.shocks is None) == (req.narrative is None):
            raise MissingInputError("give exactly one of shocks or narrative")
        shocks = {"shocks": [s.model_dump() for s in req.shocks]} if req.shocks else None
        return run(lambda: ScenarioGenerator(ctx).run(as_of_or_latest(ctx, req.as_of), shocks=shocks, narrative=req.narrative,
                                                      horizon=req.horizon, n_paths=req.n_paths, seed=req.seed, book=req.book,
                                                      loss_threshold=req.loss_threshold))

    @app.post("/v1/scenario/replay", dependencies=[Depends(auth)])
    def replay(req: ReplayRequest):
        ctx = live_ctx()
        return run(lambda: ScenarioGenerator(ctx).replay(pd.Timestamp(req.window_start), req.overrides, horizon=req.horizon,
                                                         n_paths=req.n_paths, seed=req.seed, book=req.book))

    @app.post("/v1/plan", dependencies=[Depends(auth)])
    def plan(req: PlanRequest):
        ctx = live_ctx()
        if (req.goal is None) == (req.goal_text is None):
            raise MissingInputError("give exactly one of goal or goal_text")
        goal = Goal(**req.goal.model_dump()) if req.goal else None
        return run(lambda: Planner(ctx).run(as_of_or_latest(ctx, req.as_of), req.universe, goal, goal_text=req.goal_text,
                                            horizon=req.horizon, n_paths=req.n_paths, seed=req.seed, current=req.current,
                                            cost_bps=req.cost_bps, mode=req.mode, base=req.base))

    @app.post("/v1/plan/policy", dependencies=[Depends(auth)])
    def policy(req: PolicyRequest):
        ctx = live_ctx()
        as_of = as_of_or_latest(ctx, req.as_of)
        until = pd.Timestamp(req.confirm_until) if req.confirm_until else None
        if until is not None and until > ctx.latest:
            raise MissingInputError("confirm_until is in the future")
        return run(lambda: Planner(ctx).train_policy(as_of, req.asset, req.target_vol, req.dd_limit, n_paths=req.n_paths,
                                                     seed=req.seed, confirm_until=until))

    return app


app = create_app()
