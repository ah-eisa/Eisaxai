from dataclasses import dataclass
import logging
import re


logger = logging.getLogger("eisax.evidence_tone_governor")


@dataclass
class GovernorResult:
    text: str
    edits_made: int
    edit_summary: list[str]


_THEATRICAL_RULES = [
    (r"\bThesis\s+Kill\s+Shot\b", "Primary Thesis Risk", "theatrical phrasing"),
    (r"\bThesis\s+Kill\b", "Primary Thesis Risk", "theatrical phrasing"),
    (r"\bKill\s+Shot\b", "Primary Risk", "theatrical phrasing"),
    (r"\bmoonshot\b", "high-upside scenario", "theatrical phrasing"),
    (r"\bdiamond\s+hands\b", "high-conviction holders", "theatrical phrasing"),
    (r"\bFOMO\b", "momentum-driven buying", "theatrical phrasing"),
    (r"\bsmart\s+money\b", "institutional investors", "theatrical phrasing"),
]


_MODERATE_RULES = [
    (r"\bWe forecast\b", "Analysis indicates", "deterministic certainty"),
    (r"\bWe project\b", "Our estimate is", "deterministic certainty"),
    (r"\bis destined\b", "trends toward", "deterministic certainty"),
    (r"\bguaranteed\b", "expected", "deterministic certainty"),
    (r"\bcertain\b", "likely", "deterministic certainty"),
]


_LIMITED_RULES = [
    (
        r"\bfair value of \$\d+\.\d{2}\b",
        "fair value range (precise target withheld in low-data mode)",
        "precise target withheld",
    ),
    (
        r"\bimplied price target of \$\d+\.\d{2}\b",
        "implied price range",
        "precise target withheld",
    ),
    (
        r"\bintrinsic value of \$\d+\.\d{2}\b",
        "intrinsic value range",
        "precise target withheld",
    ),
    (
        r"\btarget of \$\d+\.\d{2}\b",
        "target range",
        "precise target withheld",
    ),
    (
        r"\b((?:downside|upside) risk of )(\d+(?:\.\d+)?)%",
        r"\1modest",
        "deterministic percentage softened",
    ),
    (
        r"\b(downside|upside) of (\d+(?:\.\d+)?)%",
        r"potential \1",
        "deterministic percentage softened",
    ),
    (
        r"\bthe market is pricing peak\s+([A-Za-z0-9][^.,;:]*)",
        r"current pricing reflects an \1 scenario",
        "macro-thematic language downgraded",
    ),
    (
        r"\bthe market is pricing in\b",
        "the market reflects",
        "macro-thematic language downgraded",
    ),
    (
        r"\bis a structural valuation concern\b",
        "appears extended",
        "macro-thematic language downgraded",
    ),
    (
        r"\ba structural valuation concern\b",
        "valuation appears extended",
        "macro-thematic language downgraded",
    ),
    (
        r"\bstructural valuation concern\b",
        "valuation appears extended",
        "macro-thematic language downgraded",
    ),
    (
        r"\bdeterministic valuation\b",
        "valuation indication",
        "macro-thematic language downgraded",
    ),
    (
        r"\bDCF suggests\b",
        "fundamental coverage is insufficient for DCF",
        "macro-thematic language downgraded",
    ),
    (
        r"\bfundamental fair value\b",
        "estimated value range",
        "macro-thematic language downgraded",
    ),
    (
        r"\btrades meaningfully above fair value\b",
        "trades above the estimated range",
        "macro-thematic language downgraded",
    ),
    (
        r"\btrades meaningfully below fair value\b",
        "trades below the estimated range",
        "macro-thematic language downgraded",
    ),
    # ── Phase 5: stricter language for Limited evidence ──────────────────────
    (
        r"\bthe market is correctly pricing\b",
        "current pricing is consistent with",
        "macro-thematic language downgraded",
    ),
    (
        r"\bmarket is correctly pricing\b",
        "current pricing is consistent with",
        "macro-thematic language downgraded",
    ),
    (
        r"\btailwind already (?:partially\s+)?(?:priced|discounted)\b",
        "tailwind partly reflected in price",
        "macro-thematic language downgraded",
    ),
    (
        r"\bcould\s+compress\s+margins\s+and\s+pressure\s+the\s+stock\b",
        "may weigh on margins",
        "directional certainty softened",
    ),
    (
        r"\bpressure\s+the\s+stock\b",
        "weigh on the share price",
        "directional certainty softened",
    ),
    (
        r"\bcompress\s+margins\b",
        "weigh on margins",
        "directional certainty softened",
    ),
    (
        r"\bcorrectly\s+pricing\b",
        "consistent with current pricing",
        "macro-thematic language downgraded",
    ),
    (
        r"\bnear[\-\s]peak\b",
        "elevated",
        "macro-thematic language downgraded",
    ),
    (
        r"\bpeak\s+oil[- ]cycle\b",
        "oil-cycle",
        "macro-thematic language downgraded",
    ),
    (
        r"\bstable\s+income\s+franchise\b",
        "income-focused holding",
        "macro-thematic language downgraded",
    ),
    (
        r"\bgas\s+franchise\b",
        "gas operations",
        "macro-thematic language downgraded",
    ),
]


def _apply_rules(
    out: str,
    rules: list[tuple[str, str, str]],
    categories: set[str],
) -> tuple[str, int]:
    edits_made = 0
    for pat, repl, cat in rules:
        new = re.sub(pat, repl, out, flags=re.IGNORECASE)
        if new != out:
            categories.add(cat)
            edits_made += 1
        out = new
    return out, edits_made


def govern_tone(
    text: str,
    evidence: str = "Moderate",
    full_fundamental: bool = True,
) -> GovernorResult:
    """
    Adjust prose to match evidence quality. If evidence is "Limited" or
    full_fundamental is False, strip precise targets and downgrade strong
    macro-thematic language.

    Strong-evidence inputs pass through unchanged except for theatrical phrasing.
    """
    if not text:
        return GovernorResult(text="", edits_made=0, edit_summary=[])

    edits_made = 0
    categories: set[str] = set()
    out = text

    out, edits = _apply_rules(out, _THEATRICAL_RULES, categories)
    edits_made += edits

    if evidence in ("Moderate", "Limited") or not full_fundamental:
        out, edits = _apply_rules(out, _MODERATE_RULES, categories)
        edits_made += edits

    if evidence == "Limited" or not full_fundamental:
        out, edits = _apply_rules(out, _LIMITED_RULES, categories)
        edits_made += edits

    logger.info(
        "[EvidenceToneGovernor] evidence=%s full_fund=%s edits=%d cats=%s",
        evidence,
        full_fundamental,
        edits_made,
        sorted(categories),
    )
    return GovernorResult(text=out, edits_made=edits_made, edit_summary=sorted(categories))


def _smoke() -> None:
    scenarios = [
        (
            "Strong evidence, clean prose",
            "The company trades near its estimated range, with risks balanced by resilient margins.",
            "Strong",
            True,
        ),
        (
            "Strong evidence, theatrical phrasing",
            "Thesis Kill Shot: leverage could pressure equity returns.",
            "Strong",
            True,
        ),
        (
            "Moderate evidence, deterministic phrasing",
            "Thesis Kill Shot: We forecast margin recovery, but execution risk remains certain.",
            "Moderate",
            True,
        ),
        (
            "Limited evidence, full governor",
            (
                "Thesis Kill Shot: The market is pricing peak oil-cycle gains. "
                "DCF suggests fair value of $4.50, implying downside of 12.3%. "
                "This is a structural valuation concern."
            ),
            "Limited",
            True,
        ),
    ]

    for name, before, evidence, full_fundamental in scenarios:
        result = govern_tone(
            before,
            evidence=evidence,
            full_fundamental=full_fundamental,
        )
        print(f"=== {name} ===")
        print(f"Before: {before}")
        print(f"After:  {result.text}")
        print(f"Edits:  {result.edits_made}")
        print(f"Summary: {result.edit_summary}")
        print()


if __name__ == "__main__":
    _smoke()
