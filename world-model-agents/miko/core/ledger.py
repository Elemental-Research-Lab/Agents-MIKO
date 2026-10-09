"""Trial ledger: every run is a counted trial, including agent runs (trust rule 6)."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


class TrialLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("MIKO_LEDGER", "results/trial_ledger.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, agent: str, lineage: str, params_hash: str, result_digest: str, extra: dict[str, Any] | None = None) -> tuple[str, int]:
        trial_id = uuid.uuid4().hex[:10]
        row = {
            "trial_id": trial_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "agent": agent,
            "lineage": lineage,
            "params_hash": params_hash,
            "result_digest": result_digest,
            **(extra or {}),
        }
        with _LOCK:
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return trial_id, self.count(lineage)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def count(self, lineage: str) -> int:
        return sum(1 for r in self.rows() if r.get("lineage") == lineage)
