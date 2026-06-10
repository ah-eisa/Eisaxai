"""
Controlled Variability — EisaX editorial enhancement.

Problem: rule-based normalization makes every report read identically, producing
"template smell" — same cadence, same openers, same lexical density across all
tickers. Bloomberg-tier reports vary syntax while keeping the same policy and
vocabulary.

Solution: a small library of semantically-equivalent phrase variants, selected
deterministically by hash(ticker). Same ticker → same phrasing every session;
different tickers → different phrasings → no template smell.

How it works:
    Each PHRASE_BANK entry is (regex_pattern, [variant_a, variant_b, variant_c]).
    For every match in the text, pick variant[hash(ticker) mod N] and substitute.

Determinism guarantee:
    - Identical ticker + identical phrase → same variant always.
    - Different tickers → likely different variants (uniform distribution).
    - No randomness; no LLM call; no latency cost.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


# ── Phrase banks ───────────────────────────────────────────────────────────────
# Each entry: regex pattern (case-INSENSITIVE) → list of variant replacements.
# Variants must preserve the same semantic meaning. Order does NOT matter for
# selection — but list[0] is the "canonical" used as the regex anchor.

_PHRASE_BANKS: list[tuple[str, list[str]]] = [
    # ── Verdict justification openers ─────────────────────────────────────────
    # Match BOTH pre-normalized ("Maintain position") AND post-normalized ("Hold") forms.
    # Earlier normalization pass converts "Maintain position" → "Hold".
    (r"\b(?:Maintain\s+position|Hold)\s*[—\-]\s*await\s+confirmation\.?", [
        "Maintain exposure — pending breakout confirmation.",
        "Hold current allocation — awaiting catalyst.",
        "Stay flat — entry conditions not yet aligned.",
        "Await confirmation before extending position.",
    ]),
    (r"\b(?:Hold|maintain)\s+steady\s*[—\-]\s*monitor\.?", [
        "Hold — monitor for catalyst.",
        "Maintain allocation — watch for trigger.",
        "Stay flat — monitor key levels.",
    ]),

    # ── Risk / reward framing ─────────────────────────────────────────────────
    (r"\brisk[\-/]reward\s+(?:remains|is)\s+unattractive\b[^\.]*\.", [
        "Risk/reward remains unattractive at current levels.",
        "Current pricing leaves limited margin of safety.",
        "Upside appears constrained relative to downside risk.",
        "Reward-to-risk profile is unfavorable here.",
    ]),
    (r"\b(?:favorable|attractive)\s+risk[\-/]reward\b[^\.]*\.", [
        "Risk/reward is favorable at current levels.",
        "Upside meaningfully exceeds the downside scenario.",
        "Reward profile compensates for the underlying risk.",
        "Asymmetry tilts toward the upside.",
    ]),

    # ── Cash flow strength ────────────────────────────────────────────────────
    (r"\b(?:strong|robust)\s+cash[\-\s]generation\b[^\.]*\.", [
        "Strong cash generation supports the thesis.",
        "Cash flow generation underpins the dividend.",
        "Operating cash flow remains a key structural strength.",
        "Free cash flow profile is a defining feature.",
    ]),

    # ── Defensive characteristics ─────────────────────────────────────────────
    (r"\bdefensive\s+(?:core\s+)?holding\b[^\.]*\.", [
        "Defensive profile relative to sector.",
        "Lower-beta exposure within the sector.",
        "Stability features outweigh growth dynamics.",
        "Suited for capital-preservation allocations.",
    ]),

    # ── Catalyst / trigger language ───────────────────────────────────────────
    (r"\bWatch\s+for\s+([^\.]{5,80})\.", [
        "Watch for {1}.",
        "Monitor {1} as a confirmation signal.",
        "Look to {1} for thesis validation.",
        "{1} is the near-term trigger to track.",
    ]),

    # ── Awaiting confirmation phrasing ────────────────────────────────────────
    (r"\bAwait(?:ing)?\s+confirmation\s+before\s+adding\s+exposure\.?", [
        "Await confirmation before adding exposure.",
        "Defer additions until momentum confirms.",
        "Position sizing holds pending technical confirmation.",
    ]),

    # ── Valuation language ────────────────────────────────────────────────────
    (r"\btrades?\s+at\s+a\s+(?:slight\s+)?premium\s+to\s+(?:its\s+)?peers\b", [
        "trades at a premium to peers",
        "is priced above peer-group multiples",
        "carries a valuation premium versus the sector",
    ]),
    (r"\btrades?\s+at\s+a\s+(?:slight\s+)?discount\s+to\s+(?:its\s+)?peers\b", [
        "trades at a discount to peers",
        "is priced below peer-group multiples",
        "shows a valuation discount versus the sector",
    ]),

    # ── Action language for low-conviction holds ──────────────────────────────
    (r"\bawaiting\s+(?:catalyst|trigger|breakout)\s+confirmation\b", [
        "awaiting catalyst confirmation",
        "watching for trigger confirmation",
        "monitoring for breakout confirmation",
    ]),
]
# Note: margin-of-safety bank was removed — overlapped with risk-reward variants
# and produced cascade artifacts. Risk-reward bank already covers the concept.


# ── Deterministic selection ────────────────────────────────────────────────────
def _seed_int(ticker: str) -> int:
    """Hash ticker to a stable integer (same ticker → same seed forever)."""
    h = hashlib.md5(ticker.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _pick(variants: list[str], ticker: str, salt: int = 0) -> str:
    """Pick one variant deterministically from ticker hash (+ optional salt)."""
    if not variants:
        return ""
    idx = (_seed_int(ticker) + salt) % len(variants)
    return variants[idx]


# ── Public API ─────────────────────────────────────────────────────────────────
def apply_controlled_variability(text: str, ticker: str = "") -> str:
    """
    Walk through phrase banks and substitute canonical phrases with a
    deterministically-rotated variant. Returns the modified text.

    Args:
        text:   The full report markdown.
        ticker: Used for deterministic variant selection. Empty → no variation.

    Returns:
        Modified text. Idempotent for the same (text, ticker) pair.
    """
    if not text or not ticker:
        return text

    import os as _os_sv
    if _os_sv.getenv("EISAX_DISABLE_VARIATION", "").strip().lower() in {"1","true","yes","on"}:
        return text

    ticker_norm = ticker.upper().strip()
    salt = 0
    for pattern, variants in _PHRASE_BANKS:
        if not variants:
            continue
        choice = _pick(variants, ticker_norm, salt=salt)

        def _sub(match: re.Match) -> str:
            replacement = choice
            # Support {1} {2} placeholders for captured groups
            for i, g in enumerate(match.groups(), start=1):
                replacement = replacement.replace("{" + str(i) + "}", g or "")
            return replacement

        text = re.sub(pattern, _sub, text, flags=re.IGNORECASE)
        salt += 1   # different phrase categories rotate independently
    return text


# ── Self-test (CLI) ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Verdict: Hold\n"
        "Maintain position — await confirmation.\n"
        "Risk/reward remains unattractive at current levels.\n"
        "Strong cash generation supports the thesis.\n"
        "Defensive core holding within the energy sector.\n"
        "Watch for Brent breaking $80.\n"
    )
    for ticker in ["ADNOCGAS.AE", "2222.SR", "AAPL", "NVDA"]:
        print(f"\n=== {ticker} ===")
        print(apply_controlled_variability(sample, ticker))
