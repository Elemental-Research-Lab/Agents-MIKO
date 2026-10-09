"""LLM layer: LLMs propose, world models check (brief, section 04).

The world-model agents use an LLM in two places, both *parsing*, never computing:
- Scenario Generator: a written narrative ("S&P -20%, oil +40%, VIX to 45") -> a shock spec
- Planner: a written goal ("keep monthly CVaR under 6%, max 20% per name") -> a goal spec

Whatever proposes the spec, the agent checks it before use: every factor must exist in the
world model, every number must appear in the user's own text (an LLM may restate a number,
never invent one), and magnitudes are scored against history.

Providers (env MIKO_LLM): rules (default, deterministic, offline) | fake (scripted, for tests)
| gemini | groq | ollama. Network providers read GEMINI_API_KEY / GROQ_API_KEY / OLLAMA_HOST.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .guards import MissingInputError, ProposalRejected, assert_numbers_from_user

FACTOR_ALIASES = {
    "SP500": ["s&p 500", "s&p500", "s&p", "sp500", "spx", "spy", "equities", "equity market", "stock market", "the market", "market", "stocks"],
    "VIX": ["vix", "volatility index", "implied vol", "implied volatility"],
    "WTI": ["wti", "crude oil", "crude", "oil"],
    "AAPL": ["aapl", "apple"], "AMD": ["amd"], "BAC": ["bac", "bank of america"], "BBY": ["bby", "best buy"],
    "CVX": ["cvx", "chevron"], "GE": ["general electric"], "HD": ["home depot"], "JNJ": ["jnj", "johnson & johnson", "johnson and johnson"],
    "JPM": ["jpm", "jpmorgan", "jp morgan"], "KO": ["coca-cola", "coca cola", "coke"], "LLY": ["lly", "eli lilly", "lilly"],
    "MRK": ["mrk", "merck"], "MSFT": ["msft", "microsoft"], "PEP": ["pep", "pepsi", "pepsico"], "PFE": ["pfe", "pfizer"],
    "PG": ["procter & gamble", "procter and gamble", "p&g"], "RRC": ["rrc", "range resources"], "UNH": ["unh", "unitedhealth"],
    "WMT": ["wmt", "walmart"], "XOM": ["xom", "exxon", "exxonmobil"],
}
# whole-word tickers that are also common words are matched only in upper case
CASE_SENSITIVE = {"GE": ["GE"], "HD": ["HD"], "KO": ["KO"], "PG": ["PG"]}

SYSTEM_SHOCKS = (
    "Convert the user's market scenario into JSON: {\"shocks\": [{\"factor\": <one of FACTORS>, "
    "\"kind\": \"pct\" | \"level\", \"value\": <number copied from the user's text>}], \"ignored\": [<phrases not about FACTORS>]}. "
    "Use kind=pct for percentage moves (negative for falls) and kind=level for 'X to N'. Never invent numbers. FACTORS: {factors}"
)
SYSTEM_GOAL = (
    "Convert the user's portfolio goal into JSON: {\"risk\": {\"measure\": \"cvar95\" | \"vol\" | \"cdar95\", \"limit_pct\": <number>}, "
    "\"max_weight_pct\": <number or null>, \"long_only\": true|false}. cvar95 = expected loss in the worst 5% of months; "
    "vol = annualized volatility; cdar95 = drawdown in the worst 5% of futures. Copy numbers from the user's text; never invent them."
)


@dataclass
class LLMClient:
    provider: str = field(default_factory=lambda: os.environ.get("MIKO_LLM", "rules"))
    model: str | None = None
    scripted: list[str] = field(default_factory=list)  # for provider="fake"
    calls: list[dict] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        p = self.provider
        if p == "fake":
            if not self.scripted:
                raise MissingInputError("fake LLM has no scripted response left")
            return self.scripted.pop(0)
        if p == "gemini":
            from google import genai  # pip install google-genai

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            r = client.models.generate_content(model=self.model or "gemini-3.5-flash-lite", contents=user,
                                               config={"system_instruction": system, "temperature": 0})
            return r.text
        if p == "groq":
            from openai import OpenAI

            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
            r = client.chat.completions.create(model=self.model or "openai/gpt-oss-120b", temperature=0,
                                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content
        if p == "ollama":
            import urllib.request

            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            body = json.dumps({"model": self.model or "llama3.1:8b", "stream": False, "format": "json",
                               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
            with urllib.request.urlopen(urllib.request.Request(host + "/api/chat", body, {"Content-Type": "application/json"})) as f:
                return json.loads(f.read())["message"]["content"]
        raise ValueError(f"unknown LLM provider {p!r}")


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ProposalRejected("LLM reply contained no JSON object")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ProposalRejected(f"LLM reply was not valid JSON: {e}") from e


# ------------------------------------------------------------------ rule-based proposer
_NUM = r"([-+−]?\d+(?:\.\d+)?)"


def _find_factor_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    low = text.lower()
    for f, aliases in FACTOR_ALIASES.items():
        for a in sorted(aliases, key=len, reverse=True):
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", low):
                spans.append((m.start(), m.end(), f))
    for f, toks in CASE_SENSITIVE.items():
        for t in toks:
            for m in re.finditer(r"(?<![A-Za-z0-9])" + t + r"(?![A-Za-z0-9])", text):
                spans.append((m.start(), m.end(), f))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last_end = [], -1
    for s in spans:  # drop spans nested inside a longer earlier match
        if s[0] >= last_end:
            out.append(s)
            last_end = s[1]
    return out


def rules_propose_shocks(narrative: str) -> dict:
    """Deterministic parser for short scenario sentences, clause by clause."""
    shocks, ignored = [], []
    for clause in re.split(r"[,;\n]| and ", narrative):
        c = clause.strip()
        if not c:
            continue
        spans = _find_factor_spans(c)
        if not spans:
            ignored.append(c)
            continue
        f = spans[0][2]
        tail = c[spans[0][1]:]
        head = c[: spans[0][0]]
        lev = re.search(r"\bto\s+" + _NUM + r"(?!\s*%)", tail)
        pct = re.search(_NUM + r"\s*%", tail) or re.search(_NUM + r"\s*%", head)
        if lev and not pct:
            shocks.append({"factor": f, "kind": "level", "value": float(lev.group(1).replace("−", "-"))})
            continue
        if not pct:
            ignored.append(c)
            continue
        v = float(pct.group(1).replace("−", "-"))
        context = c.lower()
        if not pct.group(1).startswith(("+", "-", "−")):
            if re.search(r"\b(down|fall|falls|fell|drop|drops|dropped|decline|declines|crash|crashes|lose|loses|sell-?off|slump)\b", context):
                v = -abs(v)
        shocks.append({"factor": f, "kind": "pct", "value": v})
    return {"shocks": shocks, "ignored": ignored}


def rules_propose_goal(text: str) -> dict:
    t = text.lower()
    measure, limit = None, None
    pats = [("cdar95", r"(?:drawdown|draw-down|dd)[^0-9%]*?" + _NUM + r"\s*%"),
            ("cvar95", r"(?:cvar|expected shortfall|tail loss|es)\b[^0-9%]*?" + _NUM + r"\s*%"),
            ("vol", r"(?:vol|volatility|risk)\b[^0-9%]*?" + _NUM + r"\s*%")]
    for m_name, pat in pats:
        m = re.search(pat, t)
        if m:
            measure, limit = m_name, abs(float(m.group(1)))
            break
    mw = re.search(r"(?:max(?:imum)?|at most|no more than|cap(?:ped)?(?: at)?)\s*" + _NUM + r"\s*%\s*(?:per|in any|each|a)\s*(?:name|stock|asset|position|holding)", t)
    if not mw:
        mw = re.search(_NUM + r"\s*%\s*(?:max|maximum|cap)\s*(?:per|in any|each)\s*(?:name|stock|asset|position)", t)
    return {"risk": {"measure": measure, "limit_pct": limit} if measure else None,
            "max_weight_pct": abs(float(mw.group(1))) if mw else None,
            "long_only": not bool(re.search(r"\bshort(s|ing)?\b(?! of)", t))}


# --------------------------------------------------------------------------- propose
def propose_shocks(narrative: str, factors: list[str], llm: LLMClient | None = None) -> dict:
    llm = llm or LLMClient()
    if llm.provider == "rules":
        spec = rules_propose_shocks(narrative)
    else:
        spec = _extract_json(llm.complete(SYSTEM_SHOCKS.replace("{factors}", ", ".join(factors)), narrative))
    check_shock_spec(spec, narrative, factors)
    return spec


def check_shock_spec(spec: dict, narrative: str, factors: list[str]) -> None:
    shocks = spec.get("shocks")
    if not isinstance(shocks, list) or not shocks:
        raise ProposalRejected("no usable shocks found in the narrative")
    seen = set()
    for s in shocks:
        if s.get("factor") not in factors:
            raise ProposalRejected(f"proposed factor {s.get('factor')!r} is not in the world model")
        if s.get("kind") not in ("pct", "level"):
            raise ProposalRejected(f"shock kind must be pct or level, got {s.get('kind')!r}")
        if s["factor"] in seen:
            raise ProposalRejected(f"factor {s['factor']} shocked twice")
        seen.add(s["factor"])
    assert_numbers_from_user([s.get("value") for s in shocks], narrative)


def propose_goal(text: str, llm: LLMClient | None = None) -> dict:
    llm = llm or LLMClient()
    spec = rules_propose_goal(text) if llm.provider == "rules" else _extract_json(llm.complete(SYSTEM_GOAL, text))
    if not spec.get("risk") or spec["risk"].get("measure") not in ("cvar95", "vol", "cdar95"):
        raise ProposalRejected("could not find a risk limit (CVaR, volatility or drawdown with a % number) in the goal")
    nums = [spec["risk"]["limit_pct"]] + ([spec["max_weight_pct"]] if spec.get("max_weight_pct") is not None else [])
    assert_numbers_from_user(nums, text)
    return spec
