"""Per-file audit report — counts, hashes, pass/fail. Never document text/PII.

The report records WHAT happened (how many boxes, which rule sources fired,
verification status) but never the actual matched strings.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AuditReport:
    source_name: str
    source_sha256: str
    output_name: Optional[str] = None
    output_sha256: Optional[str] = None
    pages: int = 0
    boxes_total: int = 0
    boxes_by_source: Dict[str, int] = field(default_factory=dict)
    verification_passed: bool = False
    verification_reasons: List[str] = field(default_factory=list)
    mode: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def count_by_source(labels: List[str]) -> Dict[str, int]:
    return dict(Counter(labels))
