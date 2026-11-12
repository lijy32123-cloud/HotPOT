from __future__ import annotations

from typing import Any

from hotpot._internal.data_protocol.cfdb_sidecar import FAMILY_PROGRAM


class CertifiedTestSidecar:
    def __init__(self, payload: dict[str, Any]) -> None:
        entries = payload.get("entries")
        if (
            payload.get("mapping_certified") is not True
            or payload.get("scope") != "OFFICIAL_TEST_ONLY"
            or not isinstance(entries, list)
            or len(entries) != 1000
        ):
            raise ValueError("official-test sidecar contract drifted")
        self.entries = entries

    def family(self, row: int) -> str:
        value = str(self.entries[int(row)]["regime"])
        if value not in FAMILY_PROGRAM:
            raise ValueError(value)
        return value

    def program(self, row: int):
        return FAMILY_PROGRAM[self.family(row)]


__all__ = ["CertifiedTestSidecar"]
