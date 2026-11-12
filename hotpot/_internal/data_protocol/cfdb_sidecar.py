from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .manifest import ProgramId


EXPECTED_ROWS = {"cavity": 581, "cylinder": 8231, "tube": 188}
FAMILY_PROGRAM = {
    "cavity": ProgramId.CFDB_CAVITY,
    "cylinder": ProgramId.CFDB_CYLINDER,
    "tube": ProgramId.CFDB_TUBE,
}


class CertifiedCFDBSidecar:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve(strict=True)
        raw_bytes = self.path.read_bytes()
        payload = json.loads(raw_bytes)
        entries = tuple(payload.get("entries", ()))
        if (
            payload.get("mapping_certified") is not True
            or payload.get("scope") != "OFFICIAL_TRAIN_ONLY"
            or payload.get("official_test_accessed") is not False
            or len(entries) != 9000
        ):
            raise ValueError("CFDB sidecar is not the certified 9000-row train map")
        families = Counter()
        for index, entry in enumerate(entries):
            if int(entry.get("index", -1)) != index:
                raise ValueError(f"CFDB row order drift at {index}")
            family = str(entry.get("regime", ""))
            if family not in FAMILY_PROGRAM:
                raise ValueError(f"unsupported CFDB family {family!r}")
            case_key = str(entry.get("case_key", ""))
            parts = case_key.split("/")
            if len(parts) < 3 or parts[0] != family or parts[1] not in {"bc", "geo", "prop"}:
                raise ValueError(f"invalid CFDB case key {case_key!r}")
            families[family] += 1
        if dict(families) != EXPECTED_ROWS:
            raise ValueError(f"CFDB family population drifted: {dict(families)}")
        self.entries = entries
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()

    def program(self, row: int) -> ProgramId:
        if row < 0 or row >= len(self.entries):
            raise IndexError(row)
        return FAMILY_PROGRAM[str(self.entries[row]["regime"])]

    def family(self, row: int) -> str:
        if row < 0 or row >= len(self.entries):
            raise IndexError(row)
        return str(self.entries[row]["regime"])

    def case_key(self, row: int) -> str:
        if row < 0 or row >= len(self.entries):
            raise IndexError(row)
        return str(self.entries[row]["case_key"])


__all__ = ["CertifiedCFDBSidecar", "EXPECTED_ROWS", "FAMILY_PROGRAM"]
