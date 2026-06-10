from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from core.services.decision_policy import (
    canonical_evidence,
    canonical_execution,
    canonical_timing,
    canonical_verdict,
)

VerdictType = Literal["Buy", "Hold", "Reduce", "Sell"]
ActionType = Literal["Scale In", "Wait", "Hold Steady", "Reduce Exposure"]
RiskType = Literal["Low", "Moderate", "High"]
TimingType = Literal["Attractive", "Neutral", "Extended"]
EvidenceType = Literal["Limited", "Moderate", "Strong"]

logger = logging.getLogger("eisax.decision_state")

_VERDICTS = {"Buy", "Hold", "Reduce", "Sell"}
_ACTIONS = {"Scale In", "Wait", "Hold Steady", "Reduce Exposure"}
_RISKS = {"Low", "Moderate", "High"}
_TIMINGS = {"Attractive", "Neutral", "Extended"}
_EVIDENCE_LEVELS = {"Limited", "Moderate", "Strong"}


@dataclass(frozen=True)
class DecisionState:
    """Immutable authoritative decision record. All layers MUST read from this."""

    verdict: VerdictType
    action: ActionType
    risk: RiskType
    timing: TimingType
    evidence: EvidenceType
    calibration_pct: int = field(metadata={"range": "30-85"})
    score: int = field(metadata={"range": "0-100"})
    rationale: str
    emoji: str
    ticker: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", _coerce_verdict(self.verdict))
        object.__setattr__(self, "action", _coerce_action(self.action))
        object.__setattr__(self, "risk", _coerce_risk(self.risk))
        object.__setattr__(self, "timing", _coerce_timing(self.timing))
        object.__setattr__(self, "evidence", _coerce_evidence(self.evidence))
        object.__setattr__(self, "calibration_pct", _clamp_int(self.calibration_pct, 30, 85))
        object.__setattr__(self, "score", _clamp_int(self.score, 0, 100))
        object.__setattr__(self, "rationale", str(self.rationale or "")[:199])
        object.__setattr__(self, "emoji", str(self.emoji or ""))
        object.__setattr__(self, "ticker", str(self.ticker or ""))

    def to_dict(self) -> dict:
        return asdict(self)

    def axes_line(self) -> str:
        """Render the canonical 3-axis line: Risk · Evidence · Timing."""
        return f"**Risk:** {self.risk} · **Evidence:** {self.evidence} · **Timing:** {self.timing}"

    def headline(self) -> str:
        """Render the canonical verdict headline."""
        return f"**Verdict: {self.verdict} {self.emoji} · Score: {self.score}/100**"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _casefold_match(value: Any, allowed: set[str], label: str) -> str:
    raw = _text(value)
    for item in allowed:
        if raw == item or raw.casefold() == item.casefold():
            return item
    raise ValueError(f"Invalid {label}: {value!r}")


def _coerce_verdict(value: Any) -> VerdictType:
    return _casefold_match(canonical_verdict(value), _VERDICTS, "verdict")  # type: ignore[return-value]


def _coerce_timing(value: Any) -> TimingType:
    raw = _text(value)
    upper = raw.upper()
    if "BUY NOW" in upper:
        return "Attractive"
    if "REDUCE" in upper or "WAIT" in upper or "WATCHLIST" in upper:
        return "Extended"
    if "ADD ON DIP" in upper or "ACCUMULATE" in upper:
        return "Neutral"
    return _casefold_match(canonical_timing(value), _TIMINGS, "timing")  # type: ignore[return-value]


def _coerce_evidence(value: Any) -> EvidenceType:
    return _casefold_match(canonical_evidence(value), _EVIDENCE_LEVELS, "evidence")  # type: ignore[return-value]


def _coerce_risk(value: Any) -> RiskType:
    return _casefold_match(value, _RISKS, "risk")  # type: ignore[return-value]


def _coerce_action(value: Any) -> ActionType:
    return _casefold_match(value, _ACTIONS, "action")  # type: ignore[return-value]


def _clamp_int(value: Any, low: int, high: int) -> int:
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _action_from_verdict_timing(verdict: VerdictType, timing: TimingType) -> ActionType:
    return _coerce_action(canonical_execution(verdict, timing))


def _emoji_for(verdict: VerdictType) -> str:
    if verdict == "Buy":
        return "🟢"
    if verdict == "Hold":
        return "🟡"
    if verdict in {"Reduce", "Sell"}:
        return "🔴"
    return "⚪"


def _rationale(verdict: VerdictType, evidence: EvidenceType, score: int) -> str:
    if verdict in {"Reduce", "Sell"}:
        return "Risk/reward profile no longer compensates exposure."
    if score >= 70 and evidence == "Strong":
        return "Confluence of strong fundamentals and momentum."
    if score >= 65 and evidence == "Moderate":
        return "Moderate evidence supports a measured stance."
    if score < 60 or evidence == "Limited":
        return "Limited fundamental visibility; thesis is technical-led."
    return "Moderate evidence supports a measured stance."


def _calibration_pct(scorecard: dict, score: int) -> int:
    raw = _first_present(
        scorecard.get("calibration_pct"),
        scorecard.get("calibration"),
        scorecard.get("confidence_pct"),
        scorecard.get("conviction_score"),
    )
    return _clamp_int(score if raw is None else raw, 30, 85)


def build_decision_state(
    scorecard: dict,
    summary: dict | None = None,
    allow_list: dict | None = None,
    ticker: str = "",
) -> DecisionState:
    """
    Builds DecisionState from canonical scorecard dict and supporting data.

    scorecard dict expected keys (already canonicalized by scorecard.py):
        tax_verdict     -> "Buy"/"Hold"/"Reduce"/"Sell"
        tax_timing      -> "Attractive"/"Neutral"/"Extended"
        tax_evidence    -> "Limited"/"Moderate"/"Strong"
        tax_execution   -> action label
        tax_risk        -> "High"/"Moderate"/"Low"
        score           -> int 0-100
        emoji           -> str

    If any canonical field is missing, fall back to legacy (verdict, conviction)
    and coerce via core.services.decision_policy.canonical_* helpers.
    """
    if not isinstance(scorecard, dict):
        raise TypeError("scorecard must be a dict")

    summary = summary or {}
    allow_list = allow_list or {}
    _ = allow_list

    verdict = _coerce_verdict(_first_present(scorecard.get("tax_verdict"), scorecard.get("verdict")))
    timing = _coerce_timing(
        _first_present(
            scorecard.get("tax_timing"),
            scorecard.get("timing_en"),
            scorecard.get("timing"),
            "Neutral",
        )
    )
    evidence = _coerce_evidence(
        _first_present(
            scorecard.get("tax_evidence"),
            scorecard.get("evidence"),
            scorecard.get("conviction"),
            "Moderate",
        )
    )
    risk = _coerce_risk(
        _first_present(
            scorecard.get("tax_risk"),
            scorecard.get("risk"),
            summary.get("risk"),
            "Moderate",
        )
    )

    raw_action = scorecard.get("tax_execution")
    action = _coerce_action(raw_action) if raw_action not in (None, "") else _action_from_verdict_timing(verdict, timing)
    score = _clamp_int(scorecard.get("score", 0), 0, 100)
    ds = DecisionState(
        verdict=verdict,
        action=action,
        risk=risk,
        timing=timing,
        evidence=evidence,
        calibration_pct=_calibration_pct(scorecard, score),
        score=score,
        rationale=_rationale(verdict, evidence, score),
        emoji=_text(scorecard.get("emoji")) or _emoji_for(verdict),
        ticker=ticker,
    )

    logger.info(
        "[DecisionState] %s: verdict=%s action=%s risk=%s evidence=%s timing=%s score=%d",
        ticker,
        ds.verdict,
        ds.action,
        ds.risk,
        ds.evidence,
        ds.timing,
        ds.score,
    )
    return ds


def from_scorecard_decision(
    decision_dict: dict,
    summary: dict | None = None,
    ticker: str = "",
) -> DecisionState:
    """
    Convenience wrapper used by analytics handler -- bridges the
    self._last_scorecard_decision dict directly.
    """
    return build_decision_state(decision_dict, summary=summary, ticker=ticker)


if __name__ == "__main__":
    scenarios = [
        {
            "tax_verdict": "Buy",
            "tax_timing": "Attractive",
            "tax_evidence": "Strong",
            "tax_risk": "Low",
            "tax_execution": "Scale In",
            "score": 78,
            "emoji": "🟢",
        },
        {
            "tax_verdict": "Hold",
            "tax_timing": "Extended",
            "tax_evidence": "Limited",
            "tax_risk": "Moderate",
            "score": 60,
            "emoji": "🟡",
        },
        {
            "verdict": "REDUCE",
            "conviction": "Medium",
            "score": 42,
            "emoji": "🔴",
        },
    ]

    for scenario in scenarios:
        state = build_decision_state(scenario)
        print(state.to_dict())
        print(state.axes_line())

    print("CODEX_DONE")
