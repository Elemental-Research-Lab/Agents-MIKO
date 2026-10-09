"""Trust-rule guards and scoring rules. Each guard must catch a planted defect (trust rule 12)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from miko.core.guards import (CutoffError, LookAheadError, MissingInputError, ProposalRejected, assert_no_lookahead,
                              assert_numbers_from_user, check_cutoff, point_in_time, require)
from miko.core.ledger import TrialLedger
from miko.core.result import Number
from miko.evaluation import christoffersen, crps_from_samples, diebold_mariano, kupiec, pinball


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan"), np.nan])
def test_blank_is_never_zero(blank):
    with pytest.raises(MissingInputError):
        require(blank, "x")


def test_number_refuses_blank_value():
    with pytest.raises(MissingInputError):
        Number(float("nan"), "SIM", "median")
    with pytest.raises(ValueError):
        Number(1.0, "GUESS", "median")  # unknown source badge


def test_lookahead_is_caught():
    df = pd.DataFrame({"x": range(5)}, index=pd.bdate_range("2020-01-01", periods=5))
    assert_no_lookahead(df.iloc[:3], df.index[2])
    with pytest.raises(LookAheadError):
        assert_no_lookahead(df, df.index[2])  # planted: rows after the run date
    assert point_in_time(df, df.index[2]).index.max() == df.index[2]


def test_cutoff_guard_blocks_model_on_seen_period():
    check_cutoff("m", "2020-01-01", "2020-06-01")
    with pytest.raises(CutoffError):
        check_cutoff("m", "2020-06-01", "2020-01-01")  # planted: model trained past the run date


def test_llm_cannot_invent_numbers():
    assert_numbers_from_user([-20, 40], "S&P -20%, oil +40%")
    with pytest.raises(ProposalRejected):
        assert_numbers_from_user([-25], "S&P -20%")  # planted: number not in the user's text


def test_ledger_counts_every_trial(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    for i in range(3):
        _, n = led.record("a", "lin", "p", "d")
    assert n == 3 and led.count("other") == 0


def test_kupiec_and_christoffersen():
    rng = np.random.default_rng(0)
    hits = rng.random(5000) < 0.01
    assert kupiec(hits, 0.01)[1] > 0.01
    assert kupiec(rng.random(5000) < 0.03, 0.01)[1] < 1e-6  # planted: too many breaches
    clustered = np.zeros(5000, bool)
    clustered[1000:1050] = True  # planted: all breaches in one cluster
    assert christoffersen(clustered)[1] < 1e-6


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(1)
    s = rng.standard_normal((1, 200_000))
    y = np.array([0.7])
    z = 0.7
    exact = z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi)
    assert abs(crps_from_samples(s, y)[0] - exact) < 2e-3


def test_pinball_and_dm():
    y = np.array([-0.02, 0.01, 0.0])
    assert pinball(y, np.full(3, -0.01), 0.05).shape == (3,)
    rng = np.random.default_rng(2)
    a = rng.random(2000)
    stat, p = diebold_mariano(a, a + 0.1)
    assert stat < 0 and p < 1e-6
