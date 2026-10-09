"""Trust-rule guards shared by every agent.

Each guard raises instead of guessing. The rules come from the MIKO brief:
- a blank input is never treated as zero (rule 11)
- point-in-time by default: no model or dataset sees past the run date (rule 5)
- a world model is blocked for any run dated before its data cutoff
- LLMs propose, world models check; LLMs never produce a number (rule 2)
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


class MikoError(Exception):
    """Base class. Agents refuse with one of these rather than return a partial result."""


class MissingInputError(MikoError):
    """A required input is missing, blank or NaN."""


class LookAheadError(MikoError):
    """Data dated after the run date reached a computation."""


class CutoffError(MikoError):
    """A world model was asked to run on a date it has already seen."""


class InfeasibleGoalError(MikoError):
    """No plan meets the stated risk goal."""


class ProposalRejected(MikoError):
    """An LLM proposal failed a world-model check."""


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    return False


def require(value: Any, name: str) -> Any:
    """Return value, or refuse if it is blank. A blank is never zero."""
    if is_blank(value):
        raise MissingInputError(f"'{name}' is blank; refusing (a blank input is never treated as zero)")
    return value


def require_positive(value: Any, name: str) -> float:
    require(value, name)
    v = float(value)
    if not v > 0:
        raise MissingInputError(f"'{name}' must be > 0, got {v}")
    return v


def to_timestamp(as_of: Any) -> pd.Timestamp:
    require(as_of, "as_of")
    return pd.Timestamp(as_of).normalize()


def point_in_time(frame: pd.DataFrame | pd.Series, as_of: Any) -> pd.DataFrame | pd.Series:
    """Slice data to rows dated on or before the run date."""
    ts = to_timestamp(as_of)
    out = frame.loc[:ts]
    if len(out) == 0:
        raise MissingInputError(f"no data on or before {ts.date()}")
    return out


def assert_no_lookahead(frame: pd.DataFrame | pd.Series, as_of: Any, what: str = "data") -> None:
    ts = to_timestamp(as_of)
    if len(frame) and pd.Timestamp(frame.index.max()) > ts:
        raise LookAheadError(
            f"{what} contains rows dated {pd.Timestamp(frame.index.max()).date()} after run date {ts.date()}"
        )


def check_cutoff(model_name: str, cutoff: Any, as_of: Any) -> None:
    """Block a model for any run dated before its data cutoff."""
    c, a = to_timestamp(cutoff), to_timestamp(as_of)
    if a < c:
        raise CutoffError(
            f"model '{model_name}' has data cutoff {c.date()} but the run is dated {a.date()}; "
            "it has already seen this period and is blocked"
        )


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    return {abs(float(x)) for x in _NUM.findall(text or "")}


def assert_numbers_from_user(proposal_numbers: Iterable[float], user_text: str) -> None:
    """LLMs never produce a number: every number in an LLM proposal must appear in the user's text."""
    allowed = numbers_in(user_text)
    for x in proposal_numbers:
        if is_blank(x):
            raise ProposalRejected("proposal contains a blank number")
        if not any(math.isclose(abs(float(x)), a, rel_tol=1e-9, abs_tol=1e-9) for a in allowed):
            raise ProposalRejected(
                f"proposal contains the number {x}, which does not appear in the user's text; "
                "LLMs may restate numbers, never invent them"
            )
