import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def ctx(tmp_path_factory):
    from miko.agents.context import AgentContext
    from miko.core.ledger import TrialLedger

    return AgentContext(ledger=TrialLedger(tmp_path_factory.mktemp("ledger") / "trials.jsonl"))


AS_OF = "2022-12-28"
