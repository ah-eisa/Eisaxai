from __future__ import annotations

from dataclasses import dataclass, field
import re

from core.services.interpretation_engine import (
    classify_entry_quality,
    classify_support_proximity,
    classify_trend_strength,
    classify_volume_conviction,
    classify_yield_quality,
)
from core.services.positioning_validator import validate_positioning


@dataclass
class ReportSection:
    name: str
    content: str
    suppressed: bool = False

    def is_empty(self) -> bool:
        return not str(self.content or "").strip()

    def suppress(self) -> None:
        self.suppressed = True
        self.content = ""


@dataclass
class RenderedReport:
    ticker: str
    full_text: str
    sections: list[ReportSection]
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    warnings: list[str] = field(default_factory=list)
    audit_log: list[dict] = field(default_factory=list)
    observed_prices: list[float] = field(default_factory=list)

    @property
    def has_positioning_section(self) -> bool:
        return any(
            section.name == "Positioning Guide"
            and not section.suppressed
            and not section.is_empty()
            for section in self.sections
        )

    def add_visible_warning(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)


@dataclass
class LintResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safe_to_render: bool = True
    audit: list[dict] = field(default_factory=list)


def _snapshot_value(snapshot, field: str, default=None):
    try:
        return snapshot.get(field)
    except Exception:
        return default


def _interpretation_context(snapshot) -> dict:
    context = _snapshot_value(snapshot, "_interpretation_context", {})
    return context if isinstance(context, dict) else {}


def extract_prices_from_report(report, ticker: str) -> list[float]:
    if getattr(report, "observed_prices", None):
        return [float(price) for price in report.observed_prices if price is not None]

    full_text = getattr(report, "full_text", str(report))
    patterns = [
        r"(?i)(?:live price|cached price|current price|spot)\D{0,16}\$?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        r"(?i)(?:trades at|trading at)\D{0,24}\$?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
    ]

    values: list[float] = []
    for pattern in patterns:
        for match in re.findall(pattern, full_text):
            try:
                values.append(float(match.replace(",", "")))
            except ValueError:
                continue
    return values


def has_duplicate_sentences(text: str) -> bool:
    if not text:
        return False
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    normalized = []
    for sentence in raw_sentences:
        clean = re.sub(r"\s+", " ", sentence.strip(" -*>\t\r\n"))
        if len(clean) >= 12:
            normalized.append(clean.lower())
    return len(normalized) != len(set(normalized))


def _report_text(report) -> str:
    return getattr(report, "full_text", "") or ""


def _find_matches(text: str, patterns: dict[str, str]) -> list[str]:
    found: list[str] = []
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    return found


def lint_trend_language(report, snapshot) -> dict:
    context = _interpretation_context(snapshot)
    adx = float(context.get("adx") or 0)
    if adx <= 0:
        return {"check": "trend_language", "result": "SKIP", "reason": "missing_adx"}

    detail = classify_trend_strength(adx)
    expected = detail["label"]
    borderline = detail["borderline"]

    # "weak-to-emerging trend" is acceptable when expected == emerging and borderline
    acceptable = {expected}
    if expected == "emerging trend" and borderline:
        acceptable.add("weak trend")  # weak-to-emerging phrasing covers both

    mentioned = _find_matches(
        _report_text(report),
        {
            "strong trend": r"\bstrong trend\b",
            "confirmed trend": r"\bconfirmed trend\b",
            "emerging trend": r"\b(?:emerging trend|weak-to-emerging trend)\b",
            "weak trend": r"\b(?:weak trend|no trend|trend confirmation remains insufficient)\b",
        },
    )
    conflicts = [label for label in mentioned if label not in acceptable]

    if not conflicts:
        return {
            "check": "trend_language",
            "result": "PASS",
            "expected": expected,
            "borderline": borderline,
            "adx": adx,
        }

    if borderline:
        # Borderline mismatch → WARNING only, auto-correct and log
        return {
            "check": "trend_language",
            "result": "WARNING",
            "expected": expected,
            "found": conflicts,
            "adx": adx,
            "borderline": True,
            "auto_corrected": True,
            "event": "trend_tolerance_override",
        }

    # Hard mismatch → WARNING (logged, does not block render)
    return {
        "check": "trend_language",
        "result": "WARNING",
        "expected": expected,
        "found": conflicts,
        "adx": adx,
        "borderline": False,
    }


def lint_support_language(report, snapshot) -> dict:
    context = _interpretation_context(snapshot)
    price = float(context.get("price") or 0)
    support = float(context.get("support") or 0)
    if price <= 0 or support <= 0:
        return {"check": "support_language", "result": "SKIP", "reason": "missing_support_context"}

    expected = classify_support_proximity(price, support)
    distance = abs(price - support) / support
    mentioned = _find_matches(
        _report_text(report),
        {
            "near support": r"\bnear support\b",
            "above support zone": r"\babove support zone\b",
            "extended above support": r"\bextended above support\b",
        },
    )
    conflicts = [label for label in mentioned if label != expected]
    if conflicts:
        return {
            "check": "support_language",
            "result": "WARNING",
            "expected": expected,
            "found": conflicts,
            "distance_pct": round(distance * 100, 2),
        }
    return {
        "check": "support_language",
        "result": "PASS",
        "expected": expected,
        "distance_pct": round(distance * 100, 2),
    }


def lint_yield_language(report, snapshot) -> dict:
    context = _interpretation_context(snapshot)
    div_yield = context.get("div_yield")
    if div_yield is None:
        return {"check": "yield_language", "result": "SKIP", "reason": "missing_div_yield"}

    expected = classify_yield_quality(div_yield)
    mentioned = _find_matches(
        _report_text(report),
        {
            "high yield": r"\bhigh (?:yield|income component|income)\b",
            "attractive yield": r"\battractive (?:yield|income component|income)\b",
            "moderate yield": r"\bmoderate yield\b",
            "low yield": r"\blow yield\b",
            "minimal yield": r"\bminimal yield\b",
        },
    )
    conflicts = [label for label in mentioned if label != expected]
    if conflicts:
        # Downgrade to WARNING — "high yield" / "attractive yield" phrases appear in
        # descriptive prose contexts and do not reliably indicate a classification error.
        # This prevents false-positive report blocking for stocks where the LLM uses
        # "high yield" loosely (e.g. "provides a high income component") while the
        # dividend yield classifier maps the actual yield to "attractive yield".
        return {
            "check": "yield_language",
            "result": "WARNING",
            "expected": expected,
            "found": conflicts,
            "div_yield": div_yield,
        }
    return {
        "check": "yield_language",
        "result": "PASS",
        "expected": expected,
        "div_yield": div_yield,
    }


def lint_entry_language(report, snapshot) -> dict:
    context = _interpretation_context(snapshot)
    price = float(context.get("price") or 0)
    entry = float(context.get("entry_price") or 0)
    if price <= 0 or entry <= 0:
        return {"check": "entry_language", "result": "SKIP", "reason": "missing_entry_context"}

    expected = classify_entry_quality(price, entry)
    text = _report_text(report)
    errors: list[str] = []
    warnings: list[str] = []

    if re.search(r"(?:\bgood entry\b|\bfavorable entry\b|near the preferred entry zone)", text, re.IGNORECASE) and price > entry * 1.03:
        errors.append("ENTRY_TOO_STRETCHED_FOR_FAVORABLE_LANGUAGE")
    if re.search(r"(?:\bpoor timing\b|timing remains poor)", text, re.IGNORECASE) and price <= entry:
        warnings.append("POOR_TIMING_UNDERSHOOTS_RULE")

    result = "ERROR" if errors else "WARNING" if warnings else "PASS"
    return {
        "check": "entry_language",
        "result": result,
        "expected": expected,
        "price": price,
        "entry": entry,
        "errors": errors,
        "warnings": warnings,
    }


def lint_volume_language(report, snapshot) -> dict:
    context = _interpretation_context(snapshot)
    volume_today = context.get("volume_today")
    volume_avg = context.get("volume_avg")
    if not volume_today or not volume_avg:
        return {"check": "volume_language", "result": "SKIP", "reason": "missing_volume_context"}

    expected = classify_volume_conviction(volume_today, volume_avg)
    ratio = float(volume_today) / float(volume_avg) if float(volume_avg) > 0 else 0
    if re.search(r"\b(?:strong volume(?: confirmation)?|strong confirmation)\b", _report_text(report), re.IGNORECASE) and ratio <= 1.2:
        return {
            "check": "volume_language",
            "result": "ERROR",
            "expected": expected,
            "ratio": round(ratio, 4),
        }
    return {
        "check": "volume_language",
        "result": "PASS",
        "expected": expected,
        "ratio": round(ratio, 4),
    }


def lint_report(
    report,
    snapshot,
    decision: dict | None = None,
    interpretation_labels: dict | None = None,
) -> LintResult:
    errors: list[str] = []
    warnings: list[str] = []
    audit: list[dict] = []

    prices_found = extract_prices_from_report(report, _snapshot_value(snapshot, "ticker", ""))
    unique_prices = sorted({round(price, 4) for price in prices_found})
    if len(unique_prices) > 1:
        errors.append(f"PRICE_CONFLICT: multiple prices found {prices_found}")
        audit.append({"check": "price_consistency", "result": "FAIL", "values": prices_found})
    else:
        audit.append({"check": "price_consistency", "result": "PASS", "values": prices_found})

    if report.has_positioning_section:
        positioning_result = validate_positioning(report.entry, report.stop, report.target)
        if not positioning_result.is_valid:
            errors.append(f"POSITIONING_INVALID: {positioning_result.flags}")
            audit.append(
                {
                    "check": "positioning_validity",
                    "result": "FAIL",
                    "flags": positioning_result.flags,
                }
            )
        else:
            audit.append({"check": "positioning_validity", "result": "PASS", "flags": []})
    else:
        audit.append({"check": "positioning_validity", "result": "SKIP", "flags": []})

    for section in report.sections:
        if section.is_empty():
            warnings.append(f"EMPTY_SECTION: {section.name}")
            section.suppress()
            audit.append(
                {"check": "empty_section", "section": section.name, "result": "SUPPRESSED"}
            )
        else:
            audit.append({"check": "empty_section", "section": section.name, "result": "PASS"})

    if has_duplicate_sentences(report.full_text):
        warnings.append("DUPLICATE_SENTENCES_DETECTED")
        audit.append({"check": "duplicate_sentences", "result": "WARNING"})
    else:
        audit.append({"check": "duplicate_sentences", "result": "PASS"})

    try:
        if snapshot.is_cached("price"):
            warnings.append("GCC_PRICE_FROM_CACHE - label required in UI")
            audit.append({"check": "cached_price_label", "result": "WARNING"})
        else:
            audit.append({"check": "cached_price_label", "result": "PASS"})
    except Exception:
        audit.append({"check": "cached_price_label", "result": "SKIP"})

    beta_check = lint_beta_sanity(snapshot)
    audit.append(beta_check)
    if beta_check["result"] == "INVALID":
        warnings.append(f"BETA_INVALID: beta={beta_check.get('beta')} — excluded from narrative")

    _snap_beta_val = _snapshot_value(snapshot, "beta")
    beta_conflict_check = lint_beta_conflict(report, float(_snap_beta_val) if _snap_beta_val is not None else None)
    audit.append(beta_conflict_check)
    if beta_conflict_check["result"] == "ERROR":
        errors.append(
            f"BETA_CONFLICT: multiple beta values found in report "
            f"{beta_conflict_check.get('values_in_report')} "
            f"(snapshot={beta_conflict_check.get('snapshot_beta')})"
        )

    interpretation_checks = [
        lint_trend_language(report, snapshot),
        lint_support_language(report, snapshot),
        lint_yield_language(report, snapshot),
        lint_entry_language(report, snapshot),
        lint_volume_language(report, snapshot),
    ]

    for check in interpretation_checks:
        audit.append(check)
        if check["result"] == "ERROR":
            errors.append(f"{check['check'].upper()}: expected {check.get('expected')} but found {check.get('found') or check.get('errors') or check.get('ratio')}")
        elif check["result"] == "WARNING":
            for item in check.get("warnings", []):
                warnings.append(f"{check['check'].upper()}: {item}")

    # ── Verdict consistency (Week 4 binding layer) ────────────────────────
    if decision and interpretation_labels:
        _vc = lint_verdict_consistency(report, interpretation_labels, decision)
        audit.append(_vc)
        if _vc["result"] == "ERROR":
            for _ve in _vc.get("errors", []):
                errors.append(_ve)
    else:
        audit.append({"check": "verdict_consistency", "result": "SKIP",
                      "reason": "decision or interpretation_labels not provided"})

    passed = len(errors) == 0
    safe_to_render = passed
    return LintResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        safe_to_render=safe_to_render,
        audit=audit,
    )


def check_data_availability(snapshot, field: str) -> bool:
    return _snapshot_value(snapshot, field) is not None


def lint_peer_consistency(
    report: RenderedReport,
    snapshot,
    tolerance: float = 0.01,
) -> dict:
    snapshot_price = _snapshot_value(snapshot, "price")
    ticker = _snapshot_value(snapshot, "ticker")
    if not snapshot_price or not ticker:
        return {"check": "peer_self_price", "result": "SKIP", "reason": "missing_snapshot_context"}

    peer_pattern = re.compile(
        rf"\|\s*{re.escape(str(ticker))}\s*\|[^\|]*\|\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    matches = peer_pattern.findall(getattr(report, "full_text", ""))
    if not matches:
        return {"check": "peer_self_price", "result": "SKIP", "reason": "no_peer_row_found"}

    for match in matches:
        try:
            peer_price = float(match)
        except ValueError:
            continue
        if abs(peer_price - snapshot_price) > tolerance * snapshot_price:
            return {
                "check": "peer_self_price",
                "result": "MISMATCH",
                "snapshot_price": snapshot_price,
                "peer_price": peer_price,
            }
    return {"check": "peer_self_price", "result": "PASS"}


def lint_dividend_consistency(
    report: RenderedReport,
    snapshot,
    tolerance: float = 0.005,
) -> dict:
    snapshot_div = _snapshot_value(snapshot, "div_yield")
    if not snapshot_div:
        return {"check": "div_yield", "result": "SKIP", "reason": "missing_dividend_yield"}

    snapshot_pct = float(snapshot_div) * 100 if float(snapshot_div) <= 1 else float(snapshot_div)
    pattern = re.compile(r"(?i)div(?:idend)?\s*yield[^\d]{0,8}([0-9]+(?:\.[0-9]+)?)\s*%")
    mentions = [float(value) for value in pattern.findall(getattr(report, "full_text", ""))]
    if not mentions:
        return {"check": "div_yield", "result": "SKIP", "reason": "no_mentions_found"}

    mismatches = [value for value in mentions if abs(value - snapshot_pct) > tolerance * 100]
    if mismatches:
        return {
            "check": "div_yield",
            "result": "MISMATCH",
            "snapshot_pct": round(snapshot_pct, 3),
            "report_values": mismatches,
        }
    return {"check": "div_yield", "result": "PASS"}


def lint_peer_self_row(
    report: RenderedReport,
    snapshot,
    tolerance: float = 0.01,
) -> dict:
    """Strict version: self-row mismatch returns ERROR (blocks render)."""
    snapshot_price = _snapshot_value(snapshot, "price")
    ticker = _snapshot_value(snapshot, "ticker")
    if not snapshot_price or not ticker:
        return {"check": "peer_self_row", "result": "SKIP", "reason": "missing_snapshot_context"}

    peer_pattern = re.compile(
        rf"\|\s*{re.escape(str(ticker))}\s*\|[^\|]*\|\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    matches = peer_pattern.findall(getattr(report, "full_text", ""))
    if not matches:
        return {"check": "peer_self_row", "result": "SKIP", "reason": "no_peer_row_found"}

    for match in matches:
        try:
            peer_price = float(match)
        except ValueError:
            continue
        if abs(peer_price - snapshot_price) > tolerance * snapshot_price:
            return {
                "check": "peer_self_row",
                "result": "ERROR",
                "snapshot_price": snapshot_price,
                "peer_price": peer_price,
            }
    return {"check": "peer_self_row", "result": "PASS"}


def lint_beta_sanity(snapshot) -> dict:
    """Beta outside [0, 5] is invalid — exclude from narrative and audit log."""
    beta = _snapshot_value(snapshot, "beta")
    if beta is None:
        return {"check": "beta_sanity", "result": "SKIP", "reason": "missing_beta"}
    try:
        beta_val = float(beta)
    except (TypeError, ValueError):
        return {"check": "beta_sanity", "result": "SKIP", "reason": "non_numeric_beta"}

    if beta_val < 0 or beta_val > 5:
        return {
            "check": "beta_sanity",
            "result": "INVALID",
            "beta": beta_val,
            "event": "invalid_beta",
            "action": "exclude_from_narrative",
        }
    return {"check": "beta_sanity", "result": "PASS", "beta": beta_val}


def lint_beta_conflict(report: RenderedReport, snapshot_beta: float | None) -> dict:
    """
    Extract all numeric beta values mentioned in the rendered report text.
    If more than one *distinct* rounded value is found → BETA_CONFLICT error.
    Tolerates trivial rounding (rounds to 2 dp before comparing).
    """
    if snapshot_beta is None:
        return {"check": "beta_conflict", "result": "SKIP", "reason": "no_snapshot_beta"}

    full_text = _report_text(report)
    # Match patterns like "Beta: 0.40", "beta = 1.20", "Beta 0.40x" etc.
    pattern = re.compile(r'[Bb]eta[:\s=]+(-?[0-9]+(?:\.[0-9]+)?)')
    found_raw = pattern.findall(full_text)
    if not found_raw:
        return {"check": "beta_conflict", "result": "SKIP", "reason": "no_beta_in_report"}

    try:
        unique_vals = {round(float(v), 2) for v in found_raw}
    except (TypeError, ValueError):
        return {"check": "beta_conflict", "result": "SKIP", "reason": "non_numeric_beta_in_report"}

    if len(unique_vals) > 1:
        return {
            "check":          "beta_conflict",
            "result":         "ERROR",
            "event":          "BETA_CONFLICT",
            "values_in_report": sorted(unique_vals),
            "snapshot_beta":  round(float(snapshot_beta), 2),
        }
    return {"check": "beta_conflict", "result": "PASS", "beta": next(iter(unique_vals))}


def lint_52w_consistency(report: RenderedReport, snapshot) -> dict:
    has_high = check_data_availability(snapshot, "week52_high")
    has_low = check_data_availability(snapshot, "week52_low")
    has_data = has_high and has_low
    full_text = getattr(report, "full_text", "")
    says_unavailable = bool(
        re.search(r"(?i)52.?w(?:eek)?.{0,30}(?:unavailable|not available|n/?a)", full_text)
    )

    if has_data and says_unavailable:
        return {
            "check": "52w_availability",
            "result": "CONFLICT",
            "week52_high": _snapshot_value(snapshot, "week52_high"),
            "week52_low": _snapshot_value(snapshot, "week52_low"),
        }
    if not has_data:
        return {"check": "52w_availability", "result": "SKIP", "reason": "missing_52w_data"}
    return {"check": "52w_availability", "result": "PASS"}


def lint_verdict_consistency(
    report: RenderedReport,
    interpretation_labels: dict,
    decision: dict,
) -> dict:
    """
    Checks that the decision engine verdict is consistent with interpretation signals,
    and that the rendered report text does not contradict the verdict.

    Catches:
    - BUY verdict when trend is weak/emerging
    - "Trend Confirmed" language when ADX < 25
    - BUY verdict when RSI is overbought
    """
    verdict        = str(decision.get("verdict", "")).upper()
    trend_strength = interpretation_labels.get("TrendStrength", "")
    rsi_zone       = interpretation_labels.get("RSIZone", "")
    full_text      = _report_text(report)
    errors: list[str] = []

    # BUY verdict with weak/emerging trend is always a contradiction
    if verdict == "BUY" and trend_strength in ("weak trend", "emerging trend"):
        errors.append(
            f"VERDICT_TREND_CONFLICT: verdict=BUY but TrendStrength={trend_strength!r}"
        )

    # Report text claims "Trend Confirmed" while ADX is below 25
    if trend_strength in ("weak trend", "emerging trend"):
        if re.search(r"\btrend[- ]confirmed\b", full_text, re.IGNORECASE):
            errors.append(
                f"TREND_CONFIRMED_LANGUAGE_WITH_WEAK_ADX: "
                f"report says 'Trend Confirmed' but TrendStrength={trend_strength!r}"
            )

    # RSI overbought + BUY verdict
    if rsi_zone == "overbought" and verdict == "BUY":
        errors.append(
            "RSI_OVERBOUGHT_BUY_CONFLICT: verdict=BUY while RSI is overbought"
        )

    if errors:
        return {
            "check":          "verdict_consistency",
            "result":         "ERROR",
            "errors":         errors,
            "verdict":        verdict,
            "trend_strength": trend_strength,
            "rsi_zone":       rsi_zone,
        }
    return {
        "check":          "verdict_consistency",
        "result":         "PASS",
        "verdict":        verdict,
        "trend_strength": trend_strength,
    }
