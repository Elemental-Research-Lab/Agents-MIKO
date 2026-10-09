"""Scenario Generator agent: turns shocks into full paths (brief, sections 03-04).

Input is either a shock spec or a written narrative. A narrative goes through the LLM layer,
which only *proposes* a shock spec; the agent then checks it (known factors, numbers taken from
the user's text) and scores how extreme it is against history before the Scenario Engine fills
in consistent moves for every other factor and turns them into daily paths. If a book is given,
the book is run through the scenario paths.

Modes: free (shocks or narrative) and replay (a real past window with one or more factors changed).
"""
from __future__ import annotations

import numpy as np

from ..core.guards import MissingInputError, ProposalRejected, is_blank, to_timestamp
from ..core.llm import LLMClient, check_shock_spec, propose_shocks
from ..core.result import Number, Provenance, Result, count_readout, params_hash
from .context import AgentContext
from .simulator import book_paths, path_stats, validate_book

LEVEL_FACTORS = {"VIX"}


class ScenarioGenerator:
    name = "Scenario Generator"

    def __init__(self, ctx: AgentContext | None = None, llm: LLMClient | None = None):
        self.ctx = ctx or AgentContext()
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------ helpers
    def to_log_shocks(self, spec: dict, as_of) -> dict[str, float]:
        """Convert user units (pct move, or level for VIX) to cumulative log moves."""
        levels = self.ctx.prices.loc[:to_timestamp(as_of)].iloc[-1]
        out = {}
        for s in spec["shocks"]:
            f, kind, v = s["factor"], s["kind"], s.get("value")
            if is_blank(v):
                raise MissingInputError(f"shock for {f} has no value")
            v = float(v)
            if kind == "pct":
                if v <= -100:
                    raise ProposalRejected(f"{f} cannot fall {abs(v)}%")
                out[f] = float(np.log1p(v / 100.0))
            else:
                if f not in LEVEL_FACTORS and f not in levels.index:
                    raise ProposalRejected(f"no level for {f}")
                if v <= 0:
                    raise ProposalRejected(f"{f} level must be positive")
                out[f] = float(np.log(v / levels[f]))
        return out

    # ---------------------------------------------------------------------- run
    def run(self, as_of, shocks: dict | None = None, narrative: str | None = None, horizon: int = 21,
            n_paths: int = 2000, seed: int = 0, book: dict[str, float] | None = None, loss_threshold: float = 0.05,
            texture: str = "world", lineage: str | None = None, _replay: dict | None = None) -> Result:
        ctx = self.ctx
        as_of = to_timestamp(as_of)
        hist = ctx.returns_upto(as_of)
        factors = list(hist.columns)
        ignored: list[str] = []
        if narrative is not None and shocks is not None:
            raise MissingInputError("give shocks or a narrative, not both")
        if narrative is not None:
            spec = propose_shocks(narrative, factors, self.llm)
            ignored = spec.get("ignored", [])
            source = "narrative"
        elif shocks is not None:
            spec = shocks if "shocks" in shocks else {"shocks": [{"factor": k, "kind": "pct", "value": v} for k, v in shocks.items()]}
            text = " ".join(str(s.get("value")) for s in spec["shocks"])
            check_shock_spec(spec, text, factors)
            source = "spec"
        else:
            raise MissingInputError("no shocks or narrative given")
        log_shocks = self.to_log_shocks(spec, as_of)

        eng = ctx.scenario_engine(as_of, horizon)
        pl = eng.plausibility(log_shocks)
        cond = eng.condition(log_shocks, "kernel")
        base = None
        refs = [ctx.ref(eng)]
        if texture == "world":
            wm = ctx.path_model(as_of, factors, "statepath")
            base = wm.sample(n_paths, horizon, seed + 11, hist)
            refs += [ctx.ref(wm), ctx.ref(ctx.encoder(as_of))]
        paths = eng.paths(cond, n_paths, seed, base)
        cum = paths.sum(axis=1)
        simple = np.exp(cum) - 1

        rows = []
        for j, f in enumerate(factors):
            shocked = f in log_shocks
            rows.append({
                "factor": f, "shocked": shocked,
                "median_%": round(float(100 * np.median(simple[:, j])), 2),
                "p05_%": round(float(100 * np.quantile(simple[:, j], 0.05)), 2),
                "p95_%": round(float(100 * np.quantile(simple[:, j], 0.95)), 2),
                "badge": "DATA" if shocked else "SIM",
            })
        numbers = {
            "horizon": Number(horizon, "CALC", "Scenario horizon", "days"),
            "n_paths": Number(n_paths, "SIM", "Scenario paths", "count"),
            "plausibility_pct": Number(pl["percentile"], "CALC", "Joint shock is more extreme than this share of past windows", "ratio"),
            "effective_windows": Number(cond.ess or 0.0, "CALC", "Effective number of past windows used to fill in", "level"),
        }
        warnings = []
        if ignored:
            warnings.append("Not in the factor set, so ignored: " + "; ".join(ignored) + " (no macro or rates series in v1).")
        if pl["distance"] > pl["max_hist_distance"]:
            warnings.append(f"The shock is beyond every past {horizon}-day window; fill-ins are extrapolated and bands widened x{np.sqrt(cond.inflation):.2f}.")
        elif pl["percentile"] >= 0.99:
            warnings.append(f"The shock is more extreme than {pl['percentile']:.1%} of past {horizon}-day windows.")
        readout = "Scenario: " + ", ".join(
            f"{s['factor']} {'to ' + format(float(s['value']), 'g') if s['kind'] == 'level' else format(float(s['value']), '+.1f') + '%'}" for s in spec["shocks"]
        ) + f" over {horizon} trading days."
        tables = {"filled_in": rows, "shock_spec": spec["shocks"]}
        if book:
            w = validate_book(book, factors)
            idx = [factors.index(a) for a in w]
            v = book_paths(paths[:, :, idx], np.array(list(w.values())))
            st = path_stats(v, loss_threshold)
            numbers.update({
                "book_median": Number(st["median"], "SIM", "Book median return under the scenario", "pct"),
                "book_var95": Number(st["var95"], "SIM", "Book VaR 95% under the scenario", "pct_abs"),
                "book_cvar95": Number(st["cvar95"], "SIM", "Book CVaR 95% under the scenario", "pct_abs"),
                "book_n_loss": Number(st["n_loss"], "SIM", f"Scenario paths where the book loses more than {loss_threshold:.0%}", "count"),
            })
            readout += " " + count_readout(st["n_loss"], n_paths, f"the book loses more than {loss_threshold:.0%}")
        res = Result(
            agent=self.name, as_of=str(as_of.date()), numbers=numbers,
            provenance=Provenance(refs, seed, ctx.snapshot(), params_hash({"as_of": str(as_of.date()), "spec": spec, "h": horizon, "seed": seed, "book": book, "replay": _replay})),
            readout=readout, tables=tables, warnings=warnings,
            limits=["A scenario is an assumption you choose, not a prediction.", "Simulated. Not a forecast of returns."],
            verdict=f"{'replay' if _replay else 'free'} scenario from {source}",
        )
        res.trial_id, res.trial_count = ctx.ledger.record(self.name, lineage or "scenario", res.provenance.params_hash, res.digest())
        return res

    def replay(self, window_start, overrides: dict[str, float], anchors: tuple[str, ...] = ("SP500", "VIX", "WTI"),
               horizon: int = 21, **kw) -> Result:
        """Replay a real past window and change one thing: the anchors keep their actual moves
        except the overridden ones (pct moves), and everything else is filled in again from data
        before the window."""
        ctx = self.ctx
        r = ctx.returns
        t = r.index.searchsorted(to_timestamp(window_start))
        if t <= 0 or t + horizon > len(r):
            raise MissingInputError("replay window outside the data")
        as_of = r.index[t - 1]
        actual = r.iloc[t : t + horizon].sum()
        shocks = []
        for f in anchors:
            if f in overrides:
                continue
            shocks.append({"factor": f, "kind": "pct", "value": round(float(100 * (np.exp(actual[f]) - 1)), 4)})
        for f, v in overrides.items():
            shocks.append({"factor": f, "kind": "pct", "value": float(v)})
        res = self.run(as_of, shocks={"shocks": shocks}, horizon=horizon, _replay={"window_start": str(r.index[t].date()), "overrides": overrides}, **kw)
        res.tables["actual_window_%"] = [{"factor": f, "actual_%": round(float(100 * (np.exp(actual[f]) - 1)), 2), "badge": "DATA"} for f in r.columns]
        return res
