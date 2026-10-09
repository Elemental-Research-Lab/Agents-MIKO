"""The MIKO result envelope (brief, section 08).

Every agent returns a Result: numbers with source badges, a distribution (not a single
answer), a count-based read-out, provenance (model, cutoff, seed, data snapshot) and the
trial id that the run was logged under.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .guards import MissingInputError

BADGES = ("SIM", "DATA", "CALC")  # world model, retrieved data, deterministic engine

LIMITS_SIM = [
    "Simulated. Not a forecast of returns.",
    "Impact estimated, not reactive: in v1 your orders do not move simulated prices.",
]


@dataclass
class Number:
    value: float
    badge: str
    label: str
    unit: str = ""  # pct (signed move) | pct_abs (level, e.g. volatility) | ratio | count | days | level | bps | usd

    def __post_init__(self) -> None:
        if self.badge not in BADGES:
            raise ValueError(f"badge must be one of {BADGES}, got {self.badge!r}")
        if self.value is None or (isinstance(self.value, (float, np.floating)) and math.isnan(self.value)):
            raise MissingInputError(f"number '{self.label}' is blank; refusing to report it as zero")
        self.value = float(self.value)

    def fmt(self) -> str:
        v = self.value
        if self.unit == "pct":
            s = f"{v * 100:+.1f}%"
        elif self.unit == "pct_abs":
            s = f"{v * 100:.1f}%"
        elif self.unit == "count":
            s = f"{int(round(v))}"
        elif self.unit == "bps":
            s = f"{v:.1f} bps"
        elif self.unit == "days":
            s = f"{int(round(v))} days"
        else:
            s = f"{v:.3g}"
        return f"{s} [{self.badge}]"


@dataclass
class ModelRef:
    name: str
    version: str
    cutoff: str


@dataclass
class Provenance:
    models: list[ModelRef]
    seed: int | None
    data_snapshot: str
    params_hash: str
    code_version: str = "miko-wm-agents 0.1.0"


@dataclass
class Result:
    agent: str
    as_of: str
    numbers: dict[str, Number]
    provenance: Provenance
    readout: str = ""
    verdict: str | None = None
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    trial_id: str | None = None
    trial_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return json.loads(json.dumps(d, default=_jsonable))

    def digest(self) -> str:
        d = self.to_dict()
        d.pop("trial_id", None)
        d.pop("trial_count", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    def card(self) -> str:
        """Plain-text rendering of the result anatomy."""
        lines = [f"{self.agent} · as of {self.as_of}"]
        if self.verdict:
            lines.append(f"Verdict: {self.verdict}")
        if self.readout:
            lines.append(self.readout)
        for k, n in self.numbers.items():
            lines.append(f"  {n.label}: {n.fmt()}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        models = ", ".join(f"{m.name} {m.version} (cutoff {m.cutoff})" for m in self.provenance.models)
        lines.append(
            f"  Model: {models} · seed {self.provenance.seed} · data {self.provenance.data_snapshot}"
            f" · trial {self.trial_id} (#{self.trial_count} in lineage)"
        )
        for lim in self.limits:
            lines.append(f"  {lim}")
        return "\n".join(lines)


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=_jsonable).encode()).hexdigest()[:12]


def count_readout(k: int, n: int, event: str) -> str:
    """Distribution read-out in counts, e.g. 'In 7 of 20 simulated futures ...'."""
    if n <= 0:
        raise MissingInputError("read-out needs at least one path")
    # Express on a 20-path scale like the brief when n is large, keeping the exact count too.
    scaled = int(round(20 * k / n))
    return f"In {scaled} of 20 simulated futures {event} ({k} of {n} paths)."
