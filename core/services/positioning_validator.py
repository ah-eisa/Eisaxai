from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationResult:
    is_valid: bool
    flags: list[str] = field(default_factory=list)
    suppressed: bool = False
    audit: dict = field(default_factory=dict)


def validate_positioning(
    entry: float,
    stop: float,
    target: float,
    side: str = "long",
) -> ValidationResult:
    flags: list[str] = []
    checks: list[dict] = []
    normalized_side = (side or "long").lower()

    if entry is None or stop is None or target is None:
        flags.append("MISSING_POSITIONING_INPUT")
        checks.append(
            {
                "rule": "all positioning inputs present",
                "result": "FAIL",
                "values": {"entry": entry, "stop": stop, "target": target},
            }
        )
    elif normalized_side == "long":
        if stop >= entry:
            flags.append("STOP_ABOVE_ENTRY")
            checks.append(
                {
                    "rule": "stop < entry",
                    "result": "FAIL",
                    "values": {"stop": stop, "entry": entry},
                }
            )
        else:
            checks.append({"rule": "stop < entry", "result": "PASS"})

        if target <= entry:
            flags.append("TARGET_BELOW_ENTRY")
            checks.append(
                {
                    "rule": "target > entry",
                    "result": "FAIL",
                    "values": {"target": target, "entry": entry},
                }
            )
        else:
            checks.append({"rule": "target > entry", "result": "PASS"})
    elif normalized_side == "short":
        if stop <= entry:
            flags.append("STOP_BELOW_ENTRY")
            checks.append(
                {
                    "rule": "stop > entry",
                    "result": "FAIL",
                    "values": {"stop": stop, "entry": entry},
                }
            )
        else:
            checks.append({"rule": "stop > entry", "result": "PASS"})

        if target >= entry:
            flags.append("TARGET_ABOVE_ENTRY")
            checks.append(
                {
                    "rule": "target < entry",
                    "result": "FAIL",
                    "values": {"target": target, "entry": entry},
                }
            )
        else:
            checks.append({"rule": "target < entry", "result": "PASS"})
    else:
        flags.append("UNSUPPORTED_SIDE")
        checks.append(
            {
                "rule": "side in {'long', 'short'}",
                "result": "FAIL",
                "values": {"side": normalized_side},
            }
        )

    is_valid = len(flags) == 0
    audit = {
        "entry": entry,
        "stop": stop,
        "target": target,
        "side": normalized_side,
        "flags": list(flags),
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }

    return ValidationResult(
        is_valid=is_valid,
        flags=flags,
        suppressed=not is_valid,
        audit=audit,
    )


def test_long_stop_above_entry_fails_and_suppresses():
    result = validate_positioning(entry=382.0, stop=384.0, target=405.0, side="long")

    assert result.is_valid is False
    assert result.suppressed is True
    assert "STOP_ABOVE_ENTRY" in result.flags


def test_long_valid_setup_passes_without_suppression():
    result = validate_positioning(entry=382.0, stop=378.0, target=405.0, side="long")

    assert result.is_valid is True
    assert result.suppressed is False
    assert result.flags == []


def test_target_below_entry_adds_flag():
    result = validate_positioning(entry=382.0, stop=378.0, target=380.0, side="long")

    assert result.is_valid is False
    assert "TARGET_BELOW_ENTRY" in result.flags


def test_short_position_stop_below_entry_fails():
    result = validate_positioning(entry=100.0, stop=95.0, target=90.0, side="short")

    assert result.is_valid is False
    assert result.suppressed is True
    assert "STOP_BELOW_ENTRY" in result.flags


def test_all_checks_are_logged_in_audit_dict():
    result = validate_positioning(entry=382.0, stop=378.0, target=405.0, side="long")

    assert len(result.audit["checks"]) == 2
    assert result.audit["checks"][0]["rule"] == "stop < entry"
    assert result.audit["checks"][1]["rule"] == "target > entry"
    assert "timestamp" in result.audit
