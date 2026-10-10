"""Real-network test against Yahoo Finance. Skipped unless MIKO_LIVE_TESTS=1 (run it on a machine with internet)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MIKO_LIVE_TESTS") != "1", reason="set MIKO_LIVE_TESTS=1 to hit Yahoo Finance")


def test_live_smoke():
    script = Path(__file__).resolve().parent.parent / "scripts" / "live_smoke_test.py"
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900)
    print(r.stdout[-4000:])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
