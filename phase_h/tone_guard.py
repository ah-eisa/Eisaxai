"""
Phase H tone-guard.

Final-stage scrubber that removes forbidden retail wording and
neutralises hype before markdown is appended to the institutional
report. Idempotent; safe to call multiple times.

Forbidden phrases per PHASE_H_SPEC.md. Replacements use the
institutional equivalents already accepted elsewhere in the report.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .feature_flags import PHASE_H_TONE_GUARD


# Ordered (longer phrases first so they match before shorter substrings)
FORBIDDEN: List[tuple[str, str]] = [
    (r"\bhigh[- ]conviction trade\b", "elevated-conviction position"),
    (r"\bmassive upside\b",            "material upside potential"),
    (r"\bmoonshot\b",                  "high-dispersion outcome"),
    (r"\breturn enhancer\b",           "return contributor"),
    (r"\btop risk\b",                  "principal risk"),
    (r"\bAI momentum\b",               "AI-related factor exposure"),
    (r"\bstrong timing\b",             "favourable entry context"),
    (r"\bgood setup\b",                "constructive configuration"),
    (r"\bto the moon\b",               "outsized upside scenario"),
    (r"\bgame[- ]?changer\b",          "structural shift"),
    (r"\bno[- ]?brainer\b",            "low-controversy decision"),
    (r"\bsupercharge[ds]?\b",          "materially increase"),
    (r"\bcrushing it\b",               "performing strongly"),
    (r"\b🚀\b",                        ""),
    (r"\b💎\b",                        ""),
    (r"\b🔥\b",                        ""),
    (r"\b🌙\b",                        ""),
]

# Emojis are stripped from analytical sections only. Section A (Executive
# Summary) may keep neutral typographic glyphs like · — • that are not emojis.
_EMOJI_PATTERN = re.compile(
    "["                       # broad emoji range
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def scrub_text(text: str) -> str:
    """
    Apply forbidden-phrase substitutions and (optionally) emoji removal.

    Honours the EISAX_PHASE_H_TONE_GUARD flag. If the flag is off,
    returns the text unchanged so the existing behaviour is preserved
    during emergency rollback.
    """
    if not PHASE_H_TONE_GUARD or not text:
        return text

    out = text
    for pattern, replacement in FORBIDDEN:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    # collapse any double-spaces introduced by emoji-strip patterns
    out = re.sub(r"  +", " ", out)
    return out


def scrub_block(block: str, allow_emoji: bool = False) -> str:
    """
    Stronger scrub for analytical blocks: also strip emojis.
    Use for Phase H analytical subsections; pass `allow_emoji=True`
    if the upstream block is a user-facing Executive Summary line.
    """
    out = scrub_text(block)
    if not allow_emoji and out:
        out = _EMOJI_PATTERN.sub("", out)
    return out


def audit_block(text: str) -> Dict[str, int]:
    """Return a hit count for each forbidden pattern. For audit/tests."""
    counts: Dict[str, int] = {}
    if not text:
        return counts
    for pattern, _ in FORBIDDEN:
        hits = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if hits:
            counts[pattern] = hits
    emoji_hits = len(_EMOJI_PATTERN.findall(text))
    if emoji_hits:
        counts["__emoji__"] = emoji_hits
    return counts


__all__ = ["scrub_text", "scrub_block", "audit_block", "FORBIDDEN"]
