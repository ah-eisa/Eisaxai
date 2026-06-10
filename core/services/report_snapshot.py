from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Optional


@dataclass(frozen=True)
class FieldRecord:
    value: Any
    source: str
    timestamp: str
    delay_minutes: Optional[int] = None


class ReportSnapshot:
    REQUIRED_FIELDS = [
        "price",
        "entry",
        "stop",
        "target",
        "beta",
        "pe",
        "forward_pe",
        "sma50",
        "sma200",
        "week52_high",
        "week52_low",
        "market_cap",
        "div_yield",
    ]

    CANONICAL_FIELDS: frozenset[str] = frozenset({
        "price",
        "div_yield",
        "week52_high",
        "week52_low",
        "pe",
        "forward_pe",
    })

    def __init__(self, raw_data: dict):
        self._data: dict[str, FieldRecord] = {}
        self._locked = False
        self._audit_log: list[dict[str, Any]] = []
        self._load(raw_data)

    def _assert_mutable(self) -> None:
        if self._locked:
            raise RuntimeError("Snapshot is frozen and cannot be modified")

    def _assert_frozen(self) -> None:
        if not self._locked:
            raise RuntimeError("Snapshot must be frozen before read")

    def _coerce_record(self, entry: Any) -> FieldRecord:
        if isinstance(entry, dict):
            if "value" not in entry:
                raise ValueError("Snapshot field payload must include 'value'")
            return FieldRecord(
                value=entry["value"],
                source=entry.get("source", "unknown"),
                timestamp=entry.get("timestamp", datetime.now().isoformat()),
                delay_minutes=entry.get("delay_minutes"),
            )

        return FieldRecord(
            value=entry,
            source="unknown",
            timestamp=datetime.now().isoformat(),
        )

    def _load(self, raw_data: dict) -> None:
        for field in self.REQUIRED_FIELDS:
            if field not in raw_data:
                raise ValueError(f"Missing required field: {field}")

        for field, entry in raw_data.items():
            self._data[field] = self._coerce_record(entry)

    def set(self, field: str, entry: Any) -> None:
        self._assert_mutable()
        self._data[field] = self._coerce_record(entry)

    def freeze(self) -> None:
        if self._locked:
            return
        self._data = MappingProxyType(dict(self._data))
        self._locked = True
        self._audit_log.append(
            {
                "event": "snapshot_frozen",
                "timestamp": datetime.now().isoformat(),
                "fields_count": len(self._data),
            }
        )

    @property
    def _source_map(self) -> dict[str, str]:
        """Return {field: source} for all currently stored fields."""
        return {k: v.source for k, v in dict(self._data).items()}

    @property
    def _timestamp_map(self) -> dict[str, str]:
        """Return {field: timestamp} for all currently stored fields."""
        return {k: v.timestamp for k, v in dict(self._data).items()}

    def get_canonical(self, field: str) -> dict:
        """
        Return value + provenance for a canonical field.
        Raises KeyError if field is not in CANONICAL_FIELDS.
        Raises RuntimeError if snapshot is not frozen.
        """
        if field not in self.CANONICAL_FIELDS:
            raise KeyError(f"{field!r} is not a canonical field")
        record = self.get_record(field)
        return {
            "value":     record.value,
            "source":    record.source,
            "timestamp": record.timestamp,
        }

    def get(self, field: str) -> Any:
        self._assert_frozen()
        if field not in self._data:
            raise KeyError(f"Field not found: {field}")
        return self._data[field].value

    def get_record(self, field: str) -> FieldRecord:
        self._assert_frozen()
        if field not in self._data:
            raise KeyError(f"Field not found: {field}")
        return self._data[field]

    def is_cached(self, field: str) -> bool:
        self._assert_frozen()
        return self.get_record(field).source == "cache"

    def get_audit_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._audit_log]


def _snapshot_fixture() -> dict:
    return {
        "ticker": {"value": "MSFT", "source": "fallback", "timestamp": "2026-04-14T00:00:00"},
        "price": {"value": 395.0, "source": "realtime", "timestamp": "2026-04-14T00:00:00"},
        "entry": {"value": 382.0, "source": "calculated", "timestamp": "2026-04-14T00:00:00"},
        "stop": {"value": 375.0, "source": "calculated", "timestamp": "2026-04-14T00:00:00"},
        "target": {"value": 430.0, "source": "calculated", "timestamp": "2026-04-14T00:00:00"},
        "beta": {"value": 1.1, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "pe": {"value": 28.4, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "forward_pe": {"value": 26.2, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "sma50": {"value": 401.0, "source": "calculated", "timestamp": "2026-04-14T00:00:00"},
        "sma200": {"value": 404.0, "source": "calculated", "timestamp": "2026-04-14T00:00:00"},
        "week52_high": {"value": 468.0, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "week52_low": {"value": 344.0, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "market_cap": {"value": 3.1e12, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
        "div_yield": {"value": 0.008, "source": "cache", "timestamp": "2026-04-14T00:00:00"},
    }


def test_freeze_prevents_further_modification():
    import pytest

    snapshot = ReportSnapshot(_snapshot_fixture())
    snapshot.freeze()

    with pytest.raises(RuntimeError):
        snapshot.set("price", {"value": 500.0})


def test_get_before_freeze_raises_runtime_error():
    import pytest

    snapshot = ReportSnapshot(_snapshot_fixture())

    with pytest.raises(RuntimeError):
        snapshot.get("price")


def test_missing_required_field_raises_value_error():
    import pytest

    raw = _snapshot_fixture()
    raw.pop("price")

    with pytest.raises(ValueError, match="Missing required field: price"):
        ReportSnapshot(raw)


def test_cached_field_is_flagged_correctly():
    snapshot = ReportSnapshot(_snapshot_fixture())
    snapshot.freeze()

    assert snapshot.is_cached("beta") is True
    assert snapshot.is_cached("price") is False


def test_audit_log_records_freeze_event_with_timestamp():
    snapshot = ReportSnapshot(_snapshot_fixture())
    snapshot.freeze()

    audit_log = snapshot.get_audit_log()
    assert len(audit_log) == 1
    assert audit_log[0]["event"] == "snapshot_frozen"
    assert "timestamp" in audit_log[0]
    assert audit_log[0]["fields_count"] == len(_snapshot_fixture())
