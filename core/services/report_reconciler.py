"""
Report Reconciler — single-pass cleaner that aligns report markdown to a
FactSheet (SSOT). Replaces hallucinated numbers, recomputes percentages,
strips invalid FV math equations, filters irrelevant news, and flags
verdict/score mismatches.

Consumes:
    • report_text (the LLM-generated markdown after editorial pipeline)
    • FactSheet (from core.services.fact_sheet — already authoritative)

Returns:
    (cleaned_text, ReconciliationAudit)
        — audit.blocked  → caller MUST NOT publish the report
        — audit.warnings → human-reviewable but non-blocking
        — audit.corrections → log every swap (old/new/rule)

Design constraints (per architecture review):
    • Reconciler reads NOTHING but FactSheet (no I/O, no LLM calls)
    • Idempotent: re-running on already-clean text is a no-op
    • Single pass: walks the text in a fixed rule order
    • Never invents data: if FactSheet lacks a value, original text is kept
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.services.fact_sheet import FactSheet, SectorSubtype

logger = logging.getLogger("eisax.reconciler")


@dataclass
class Correction:
    field_name: str
    old: str
    new: str
    rule: str


@dataclass
class ReconciliationAudit:
    corrections: list[Correction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None

    def summary(self) -> str:
        parts = []
        if self.blocked:
            parts.append(f"BLOCKED({self.block_reason})")
        parts.append(f"corrections={len(self.corrections)}")
        if self.warnings:
            parts.append(f"warnings={len(self.warnings)}")
        return " ".join(parts)


# ── Public entry point ────────────────────────────────────────────────────
def reconcile_report(text: str, fs: FactSheet) -> tuple[str, ReconciliationAudit]:
    """
    Walk through report text and apply SSOT corrections from the FactSheet.

    Order of rules:
        0. Blocking check — FactSheet.blocking_errors → audit.blocked = True
        1. Price reconciliation (Live Price rows + top card)
        2. SMA200 / SMA50 numeric value swaps
        3. SMA200 row swap in S/R ladder
        4. Percentage comparison sentence rebuild
        5. Fair-Value math equation integrity check
        6. Verdict consistency (diagnostic — flags only, may block)
        7. Score label disambiguation (Fundamental Quality vs EisaX Score)
        8. News filter by sector subtype
        9. Currency forcing for MENA tickers
       10. Sector-subtype-specific scrubs (oil for RE, banking for energy, etc.)

    Returns:
        (text, audit)
    """
    audit = ReconciliationAudit()
    if not text or not fs:
        return text, audit

    # ── 0. Blocking pre-check ───────────────────────────────────────────
    if fs.blocking_errors:
        audit.blocked = True
        audit.block_reason = ", ".join(fs.blocking_errors)
        logger.error("[Reconciler] %s BLOCKED: %s", fs.ticker, audit.block_reason)
        # Still apply non-blocking rules so the (un-published) text is at least clean
        # but caller will refuse to ship it.

    cur = fs.currency_symbol or ""

    # ── 1. Price ─────────────────────────────────────────────────────────
    if fs.price is not None and fs.price > 0:
        text = _swap_n(
            text,
            r"(\|\s*Live\s+Price\s*\|\s*)(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d,]+\.\d{1,4}",
            f"\\g<1>{cur}{fs.price:.2f}",
            audit, "price", "live_price_factcheck_row",
        )
        text = _swap_n(
            text,
            r"(\*\*🔴\s+Live Price:\*\*\s*)[\d,]+\.\d{1,4}\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?",
            f"\\g<1>{fs.price:.2f} {cur}".strip(),
            audit, "price", "topcard_live_price",
        )

    # ── 2. SMA200 paren value ───────────────────────────────────────────
    # First detect the LLM's hallucinated SMA200 value (if any) so we can
    # find every echo of it (entry-zone references, etc.) and rewrite them.
    hallucinated_sma200 = _extract_first_sma_value(text, "200", ssot_value=fs.sma200)
    hallucinated_sma50  = _extract_first_sma_value(text, "50",  ssot_value=fs.sma50)

    if fs.sma200 and fs.sma200 > 0:
        _cur_pat = r"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)"
        # Paren format: SMA200 (3.19 د.إ)  or  SMA200 (د.إ3.19)
        text = _swap_n(
            text,
            rf"SMA\s*200\s*\(\s*{_cur_pat}?\s*[\d,]+\.\d{{1,4}}\s*{_cur_pat}?\s*\)",
            f"SMA200 ({cur}{fs.sma200:.2f})",
            audit, "sma200", "sma200_paren_swap",
        )
        # Colon format: "SMA200: 3.19 د.إ"  (LLM sometimes uses "- SMA50: X | SMA200: Y")
        text = _swap_n(
            text,
            rf"(SMA\s*200\s*:\s*){_cur_pat}?\s*[\d,]+\.\d{{1,4}}\s*{_cur_pat}?",
            f"\\g<1>{cur}{fs.sma200:.2f}",
            audit, "sma200", "sma200_colon_swap",
        )
        # Bold-colon format: "**SMA200:** د.إ3.19" (markdown bold variant)
        text = _swap_n(
            text,
            rf"(\*\*SMA\s*200:\*\*\s*){_cur_pat}?\s*[\d,]+\.\d{{1,4}}\s*{_cur_pat}?",
            f"\\g<1>{cur}{fs.sma200:.2f}",
            audit, "sma200", "sma200_bold_swap",
        )
        # 3. SMA200 row in S/R ladder
        text = _swap_n(
            text,
            r"(\|\s*S\d\s*\|\s*)(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d,]+\.\d{1,4}"
            r"(\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*\|\s*Support\s*\|\s*SMA200\s*\|)",
            f"\\g<1>{cur}{fs.sma200:.2f}\\g<2>",
            audit, "sma200", "sr_ladder_sma200_row",
        )

        # 3b. If the LLM picked a wrong SMA200 value and reused it
        # everywhere ("entry zone 3.19", "support 3.19", etc.), swap each
        # occurrence in level-like contexts.
        if hallucinated_sma200 and abs(hallucinated_sma200 - fs.sma200) > 0.01:
            text = _swap_hallucinated_level(
                text,
                wrong=hallucinated_sma200,
                right=fs.sma200,
                currency=cur,
                audit=audit,
                rule="sma200_echo_swap",
                field_name="sma200_echo",
            )

        # 3c. S/R ladder "not computable" — inject SSOT SMA200 inline
        # When LLM omits SMA data entirely (52-week range unavailable in its
        # context), the SSOT SMA200 from the TV cache is still authoritative.
        # Patch the "not computable" sentence to include it.
        if fs.price and fs.price_vs_sma200_pct is not None:
            pct = fs.price_vs_sma200_pct
            pos = "above" if pct >= 0 else "below"
            injection = (
                f"**Technical S/R Ladder (SSOT):**\n"
                f"| Level | Price | Type | Basis |\n"
                f"|-------|-------|------|-------|\n"
                f"| S1 | {cur}{fs.sma200:.2f} | Support | SMA200 (TV cache) |\n\n"
                f"*Price at {cur}{fs.price:.2f} is {abs(pct):.1f}% {pos} SMA200.*"
            )
            new_text, n = re.subn(
                r"\*\*Technical S/R Ladder:\*\*\s*Not computable[^\n]*\n",
                injection + "\n",
                text,
            )
            if n:
                audit.corrections.append(
                    Correction(
                        "sma200_ladder_inject",
                        "S/R ladder not computable",
                        f"injected SMA200={cur}{fs.sma200:.2f} ({abs(pct):.1f}% {pos})",
                        "sr_ladder_sma200_inject",
                    )
                )
                text = new_text

    if fs.sma50 and fs.sma50 > 0:
        text = _swap_n(
            text,
            r"SMA\s*50\s*\(\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d,]+\.\d{1,4}\s*"
            r"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*\)",
            f"SMA50 ({cur}{fs.sma50:.2f})",
            audit, "sma50", "sma50_paren_swap",
        )
        if hallucinated_sma50 and abs(hallucinated_sma50 - fs.sma50) > 0.01:
            text = _swap_hallucinated_level(
                text,
                wrong=hallucinated_sma50,
                right=fs.sma50,
                currency=cur,
                audit=audit,
                rule="sma50_echo_swap",
                field_name="sma50_echo",
            )

    # ── 4. Percentage comparison rebuilds ────────────────────────────────
    if fs.price and fs.sma200 and fs.price_vs_sma200_pct is not None:
        pct = fs.price_vs_sma200_pct
        pos = "above" if pct >= 0 else "below"
        # Form A: "Price is above SMA200 (X) by Y%" — solo SMA200
        text = _swap_n(
            text,
            r"Price\s+is\s+(?:above|below)\s+SMA\s*200\s*\([^)]+\)\s+by\s+\+?\d+\.\d+%",
            f"Price is {pos} SMA200 ({cur}{fs.sma200:.2f}) by {abs(pct):.1f}%",
            audit, "pct_sma200", "rebuild_is_X_by_Y",
        )
        # Form B: "Price at <price> is +Y% above SMA200 (V)" — solo, prefix-anchored
        text = _swap_n(
            text,
            r"Price\s+at\s+(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d,]+\.\d{1,4}"
            r"\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s+is\s+[+\-]?\d+\.\d+%\s+"
            r"(?:above|below)\s+SMA\s*200\s*\([^)]+\)",
            f"Price at {cur}{fs.price:.2f} is {abs(pct):.1f}% {pos} SMA200 ({cur}{fs.sma200:.2f})",
            audit, "pct_sma200_solo", "rebuild_price_at_solo_sma200",
        )
        # Form C: "Price at X is Y% above SMA200 (V) and Z% below SMA50 (W)"
        if fs.sma50 and fs.price_vs_sma50_pct is not None:
            pct50 = fs.price_vs_sma50_pct
            pos50 = "above" if pct50 >= 0 else "below"
            text = _swap_n(
                text,
                r"Price\s+at\s+(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*[\d,]+\.\d{1,4}"
                r"\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s+is\s+[+\-]?\d+\.\d+%\s+"
                r"(?:above|below)\s+SMA\s*200\s*\([^)]+\)\s+and\s+"
                r"[+\-]?\d+\.\d+%\s+(?:above|below)\s+SMA\s*50\s*\([^)]+\)",
                (
                    f"Price at {cur}{fs.price:.2f} is {abs(pct):.1f}% {pos} "
                    f"SMA200 ({cur}{fs.sma200:.2f}) and {abs(pct50):.1f}% "
                    f"{pos50} SMA50 ({cur}{fs.sma50:.2f})"
                ),
                audit, "pct_sma_combo", "rebuild_full_comparison",
            )

    # ── 5. Fair-Value math equation integrity check ─────────────────────
    text = _validate_fv_math(text, fs, audit)

    # ── 6. Verdict consistency ──────────────────────────────────────────
    if fs.verdict:
        body_verdicts = set()
        # Pattern A: "Verdict: Buy", "verdict: **Hold**", "Verdict 🟢 Buy" etc.
        for m in re.finditer(
            r"\bVerdict[:\s]+(?:🟢|🔴|⚪|🟡)?\s*\*{0,2}(Buy|Hold|Reduce|Sell)\b",
            text,
            re.IGNORECASE,
        ):
            body_verdicts.add(m.group(1).title())
        # Pattern B: narrative "our verdict is a Reduce" / "verdict is Reduce"
        for m in re.finditer(
            r"\bverdict\s+is\s+(?:a\s+)?\*{0,2}(Buy|Hold|Reduce|Sell)\b",
            text,
            re.IGNORECASE,
        ):
            body_verdicts.add(m.group(1).title())
        if body_verdicts and body_verdicts != {fs.verdict}:
            # Auto-correct: swap any non-SSOT verdict label to SSOT
            for wrong in body_verdicts - {fs.verdict}:
                # Swap Pattern A: "Verdict: <wrong>"
                text = _swap_n(
                    text,
                    rf"(\bVerdict[:\s]+(?:🟢|🔴|⚪|🟡)?\s*\*{{0,2}}){re.escape(wrong)}\b",
                    f"\\g<1>{fs.verdict}",
                    audit, "verdict", f"verdict_swap_{wrong}_to_{fs.verdict}",
                )
                # Swap Pattern B: "verdict is a <wrong>" / "verdict is <wrong>"
                # Replace only the verdict word — do NOT add bold markers so
                # we don't break surrounding bold spans (e.g. "**Our verdict
                # is a Reduce**" → "**Our verdict is a Buy**").
                text = _swap_n(
                    text,
                    rf"(\bverdict\s+is\s+(?:a\s+)?){re.escape(wrong)}\b",
                    f"\\g<1>{fs.verdict}",
                    audit, "verdict", f"verdict_narrative_swap_{wrong}_to_{fs.verdict}",
                )
            audit.warnings.append(
                f"verdict_mismatch_corrected: body had {body_verdicts}, "
                f"SSOT={fs.verdict}"
            )

    # ── 7. Score label disambiguation ───────────────────────────────────
    # Catch ANY standalone "Score: N/100" line where the trailing text talks
    # about business quality (LLM uses many phrasings: "Score reflects business
    # quality", "Business quality is adequate", "reflects company quality",
    # ...). Accept em-dash, en-dash, hyphen, or colon as the separator.
    text = _swap_n(
        text,
        r"(?m)^(\s*)\*{0,2}\s*Score:\s*\*{0,2}(\d{1,3})/100\*{0,2}\s*"
        r"[—–\-:]\s*"
        r"(?:Score\s+reflects|Reflects|Business\s+quality|"
        r"Reflects\s+company|Company\s+quality|"
        r"Fundamental\s+quality)[^.\n]*\.?\s*\*{0,2}",
        (
            r"\g<1>**Fundamental Quality Score: \g<2>/100** — reflects "
            r"business quality only (independent of the top-card EisaX Score, "
            r"which is a blended composite)."
        ),
        audit, "score_label", "fundamental_quality_disambig",
    )

    # ── 8. News filter ──────────────────────────────────────────────────
    if fs.news_required_keywords:
        text = _filter_news(text, fs, audit)

    # ── 9. Currency forcing for MENA ────────────────────────────────────
    if cur and cur != "$" and fs.is_mena:
        before = text
        text = re.sub(
            r"(\|\s*Live\s+Price\s*\|\s*)\$([\d,]+\.\d+)",
            rf"\1\2 {cur}",
            text,
        )
        if text != before:
            audit.corrections.append(
                Correction("currency", "$", cur, "currency_swap_factcheck")
            )

    # ── 10. Sector-specific scrubs ──────────────────────────────────────
    text = _sector_scrub(text, fs, audit)

    # ── 11. Missing-value placeholder normalization ─────────────────────
    # When a metric (e.g. SMA50) is unavailable, the builder renders a
    # "<currency>0.00 (N/A)" placeholder, which is misleading. Normalise to
    # plain "N/A". The (?<![\d.]) guard + [^\d\s|] currency cluster ensure we
    # never touch a real number like "30.00 (N/A)" — only a standalone 0.00.
    new_na, n_na = re.subn(
        r"(?<![\d.])[^\d\s|]{0,4}0\.00\s*\(N/?A\)",
        "N/A",
        text,
    )
    if n_na:
        text = new_na
        audit.corrections.append(
            Correction(
                "formatting",
                f"{n_na} '0.00 (N/A)' placeholder(s)",
                "N/A",
                "missing_value_na_fix",
            )
        )

    # ── 12. Orphan markdown bold marker cleanup ─────────────────────────
    # v4-flash occasionally emits a stray closing ** without a matching
    # opener on the same line (e.g. "Reduce** —" at the end of a long
    # interpolated sentence). Lines with an odd count of ** markers are
    # the signal; fix by removing the last (unpaired) ** on such lines.
    fixed_lines = []
    n_orphan = 0
    for _line in text.split("\n"):
        _stars = re.findall(r"\*{2}", _line)
        if len(_stars) % 2 != 0:
            # Remove the last ** on this line (it has no opening partner)
            _last = _line.rfind("**")
            if _last >= 0:
                _line = _line[:_last] + _line[_last + 2:]
                n_orphan += 1
        fixed_lines.append(_line)
    if n_orphan:
        text = "\n".join(fixed_lines)
        audit.corrections.append(
            Correction(
                "formatting",
                f"{n_orphan} line(s) with orphaned **",
                "removed stray closing **",
                "orphan_bold_fix",
            )
        )

    logger.info(
        "[Reconciler] %s subtype=%s %s",
        fs.ticker, fs.sector_subtype.value, audit.summary(),
    )
    return text, audit


# ── Helpers ───────────────────────────────────────────────────────────────
def _swap_n(
    text: str,
    pattern: str,
    replacement: str,
    audit: ReconciliationAudit,
    field_name: str,
    rule: str,
) -> str:
    """re.sub with logging. Returns new text; logs a Correction iff changed."""
    try:
        new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
    except re.error as e:
        logger.warning("[Reconciler] regex error in %s: %s", rule, e)
        return text
    if n > 0 and new_text != text:
        audit.corrections.append(
            Correction(field_name, f"[{n} mention(s)]", "[updated]", rule)
        )
    return new_text


def _extract_first_sma_value(
    text: str,
    period: str,
    ssot_value: float | None = None,
) -> float | None:
    """
    Find the numeric value the LLM internally uses for SMA<period>.
    If ssot_value is provided, prefer a candidate that differs from it
    (i.e. find the LLM's hallucinated value, not the already-correct one).

    Formats handled:
      • Paren  : "SMA200 (3.19 د.إ)"      → 3.19
      • Colon  : "SMA200: 3.19 د.إ"        → 3.19
      • Prose  : "SMA200 at د.إ3.19"       → 3.19
      • Reverse: "(د.إ3.19 SMA200)"        → 3.19
    Returns None if no divergent value found.
    """
    _cur = r"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)"
    _bold = r"(?:\*\*?)?"   # optional markdown bold markers
    patterns = [
        # Paren: SMA200 (3.19 د.إ)  or  SMA200 (د.إ3.19)
        re.compile(
            rf"{_bold}SMA\s*{period}{_bold}\s*\(\s*{_cur}?\s*([\d,]+\.\d{{1,4}})\s*{_cur}?\s*\)",
            re.IGNORECASE,
        ),
        # Colon/bold-colon: "SMA200: 3.19", "**SMA200:** 3.19", "SMA200: د.إ3.19"
        re.compile(
            rf"{_bold}SMA\s*{period}{_bold}\s*:\s*{_bold}\s*{_cur}?\s*([\d,]+\.\d{{1,4}})",
            re.IGNORECASE,
        ),
        # Prose: "SMA200 at X.XX" or "SMA200 = X.XX"
        re.compile(
            rf"SMA\s*{period}\s+(?:at|of|=)\s*{_cur}?\s*([\d,]+\.\d{{1,4}})",
            re.IGNORECASE,
        ),
        # Reverse paren: "(د.إ3.19 SMA200)"  or "(3.19 SMA200)"
        re.compile(
            rf"\(\s*{_cur}?\s*([\d,]+\.\d{{1,4}})\s*{_cur}?\s*SMA\s*{period}\s*\)",
            re.IGNORECASE,
        ),
    ]
    # Collect (position, value) from all patterns
    hits: list[tuple[int, float]] = []
    for pat in patterns:
        for m in pat.finditer(text):
            try:
                hits.append((m.start(), float(m.group(1).replace(",", ""))))
            except (TypeError, ValueError):
                pass
    if not hits:
        return None

    if ssot_value is not None:
        # Prefer a value that diverges from SSOT by more than 1%
        divergent = [
            (pos, val) for pos, val in hits
            if abs(val - ssot_value) > max(0.01, ssot_value * 0.005)
        ]
        if divergent:
            # Return the earliest divergent value
            return min(divergent, key=lambda x: x[0])[1]

    # Fallback: earliest overall
    return min(hits, key=lambda x: x[0])[1]


def _swap_hallucinated_level(
    text: str,
    wrong: float,
    right: float,
    currency: str,
    audit: ReconciliationAudit,
    rule: str,
    field_name: str,
) -> str:
    """
    Replace echoes of a hallucinated price level (e.g. SMA200=3.19) with
    the SSOT value (3.41) wherever it appears in "level-like" contexts:
        • "entry zone" / "preferred zone"
        • "support" / "resistance"
        • "SMA<period> zone"
        • "above/below 3.19"
        • table rows where the wrong value sits alone

    Only matches when the wrong number appears as a price level (next to a
    currency symbol OR in known support/entry contexts). We DON'T blanket-
    replace every "3.19" because it might legitimately appear in unrelated
    contexts (percentages, identifiers, etc.).
    """
    # Build a regex for the wrong value, allowing ±0.01 tolerance via
    # exact match on the printed form. Use two decimals as canonical.
    wrong_2dp = f"{wrong:.2f}"
    wrong_pat = re.escape(wrong_2dp)
    # Also handle the 3-decimal form if present
    wrong_3dp = f"{wrong:.3f}"
    wrong_pat_3 = re.escape(wrong_3dp)

    replacements = [
        # "(<wrong> د.إ)" — currency-anchored value in parens
        (
            rf"\(\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*(?:{wrong_pat}|{wrong_pat_3})"
            rf"\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*\)",
            f"({currency}{right:.2f})",
        ),
        # "د.إ<wrong>" or "<wrong> د.إ" — currency-anchored bare
        (
            rf"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)\s*(?:{wrong_pat}|{wrong_pat_3})",
            f"{currency}{right:.2f}",
        ),
        (
            rf"(?:{wrong_pat}|{wrong_pat_3})\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)",
            f"{right:.2f} {currency}".strip(),
        ),
        # "entry zone of <wrong>" / "support at <wrong>" / "below <wrong>"
        (
            rf"\b(?:entry\s+zone|preferred\s+(?:zone|entry)|support(?:\s+at)?|"
            rf"resistance(?:\s+at)?|below|above|near|at)\s+(?:of\s+)?"
            rf"(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*(?:{wrong_pat}|{wrong_pat_3})"
            rf"\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?",
            lambda m: m.group(0).replace(wrong_2dp, f"{right:.2f}")
                                  .replace(wrong_3dp, f"{right:.2f}"),
        ),
    ]
    n_total = 0
    for pat, repl in replacements:
        try:
            if callable(repl):
                new_text, n = re.subn(pat, repl, text, flags=re.IGNORECASE)
            else:
                new_text, n = re.subn(pat, repl, text, flags=re.IGNORECASE)
        except re.error as e:
            logger.warning("[Reconciler] regex error in echo swap: %s", e)
            continue
        if n > 0:
            n_total += n
            text = new_text
    if n_total:
        audit.corrections.append(
            Correction(
                field_name,
                f"{wrong_2dp} (LLM)",
                f"{right:.2f} (TV SSOT)",
                f"{rule}_n{n_total}",
            )
        )
    return text


def _validate_fv_math(text: str, fs: FactSheet, audit: ReconciliationAudit) -> str:
    """
    Any visible "Forward EPS X × Y x sector P/E = Z" equation must satisfy
    Z ≈ X × Y (within 5%). If not, strip the "= Z" tail so the displayed
    target stands as a proprietary methodology output, not a derived equation.
    """
    pat = re.compile(
        r"(Forward|TTM)\s+EPS\s*\(?\s*(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*"
        r"(?P<eps>[\d.]+)\s*\)?\s*[×x]\s*"
        r"(?P<mult>\d+(?:\.\d+)?)\s*x?\s+(?:sector|peer)\s*P/E\s*=\s*"
        r"\*?\*?(?:د\.إ|﷼|ج\.م|ر\.ق|\$)?\s*(?P<z>[\d.]+)\*?\*?",
        re.IGNORECASE,
    )

    def _check(m: re.Match) -> str:
        try:
            eps = float(m.group("eps"))
            mult = float(m.group("mult"))
            z = float(m.group("z"))
        except (TypeError, ValueError):
            return m.group(0)
        if eps <= 0 or mult <= 0 or z <= 0:
            return m.group(0)
        calc = eps * mult
        if abs(calc - z) / z < 0.05:
            # Math holds — keep as-is
            return m.group(0)
        # Math does NOT hold — strip the equation tail
        audit.corrections.append(
            Correction(
                "fv_math",
                f"{eps}×{mult}={z} (calc {calc:.2f})",
                "stripped",
                "fv_math_mismatch_strip",
            )
        )
        return (
            f"the EisaX Fair Value Estimate is based on a proprietary "
            f"forward-earnings × peer multiple methodology (calculation "
            f"detail omitted; displayed target = {fs.currency_symbol or ''}{z:.2f})"
        )

    new_text = pat.sub(_check, text)
    return new_text


def _filter_news(text: str, fs: FactSheet, audit: ReconciliationAudit) -> str:
    """Keep only news items matching required keywords; drop excluded ones."""
    req_re = re.compile("|".join(fs.news_required_keywords), re.IGNORECASE)
    exc_re = (
        re.compile("|".join(fs.news_excluded_keywords), re.IGNORECASE)
        if fs.news_excluded_keywords
        else None
    )
    m = re.search(
        r"(?ms)^📰\s*\*\*Latest News\*\*[^\n]*\n(?P<body>.+?)(?=^---|\n## |\Z)",
        text,
    )
    if not m:
        return text
    body = m.group("body")
    new_lines: list[str] = []
    dropped = 0
    for line in body.split("\n"):
        stripped = line.strip()
        is_news_item = bool(re.match(r"^[-*]\s*(?:⚪\s*)?\[", stripped))
        if is_news_item:
            keep = req_re.search(stripped) is not None
            if keep and exc_re and exc_re.search(stripped):
                keep = False
            if keep:
                new_lines.append(line)
            else:
                dropped += 1
        else:
            new_lines.append(line)
    new_body = "\n".join(new_lines)
    if not re.search(r"^[-*]\s*(?:⚪\s*)?\[", new_body, re.MULTILINE):
        new_body = "\n_No directly relevant news items in the current feed window._\n"
    if dropped:
        audit.corrections.append(
            Correction("news", f"{dropped} unrelated items", "dropped", "news_filter")
        )
    return text[: m.start("body")] + new_body + text[m.end("body"):]


def _sector_scrub(text: str, fs: FactSheet, audit: ReconciliationAudit) -> str:
    """
    Sector-subtype-aware sentence scrubs. E.g., for Real-Estate tickers,
    any sentence mentioning Brent/OPEC/crude is off-thesis and stripped.
    For Banks, any oil/gas commodity sentence is off-thesis.
    Also fixes the Sector/Industry metadata line when TV sends wrong sector.
    """
    # Real-estate or Bank tickers: strip any commodity/oil cycle prose
    if fs.is_real_estate or fs.is_financial:
        commodity_phrases = [
            r"cyclical\s+commodity\s+exposure",
            r"commodity\s+price\s+cycle",
            r"oil[- ]?\s*price\s+(?:sensitivity|decline|drop|fall|swing|spike|volatility|impact|risk)",
            r"oil\s+prices?\s+(?:fall|rise|drop|spike|decline|surge|crash)",
            r"hydrocarbon\s+(?:price\s+)?exposure",
            r"\bcrude[- ]oil\b",
            r"OPEC[+\- ]?\s*(?:supply|production|cuts?|decisions?)?",
            r"\bOPEC\b",
            r"\bBrent\b",
            r"\bhydrocarbon\b",
            r"energy[- ]price\s+(?:volatility|sensitivity)",
            r"correlation\s+to\s+oil",
            r"lower\s+crude\s+pressures",
            r"oil\s+price\s+(?:correlation|exposure|sensitivity|decline|drop)",
        ]
        total = 0
        for p in commodity_phrases:
            # Sentence-level
            new_text, n1 = re.subn(
                rf"[^.\n]*\b{p}\b[^.\n]*\.\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            # Bullet-level
            new_text, n2 = re.subn(
                rf"(?m)^[\-\*]\s*\*?\*?[^\n]*?\b{p}\b[^\n]*\n",
                "",
                new_text,
                flags=re.IGNORECASE,
            )
            # Table-row level
            new_text, n3 = re.subn(
                rf"(?m)^\|[^\n]*\b{p}\b[^\n]*\|\s*\n",
                "",
                new_text,
                flags=re.IGNORECASE,
            )
            total += n1 + n2 + n3
            text = new_text
        if total:
            label = "real_estate" if fs.is_real_estate else "bank"
            audit.corrections.append(
                Correction(
                    "sector_scrub",
                    f"{total} off-thesis commodity mentions",
                    "stripped",
                    f"{label}_strip_oil_lang",
                )
            )
        # Collapse blank-line cascades the scrub leaves behind
        text = re.sub(r"\n{3,}", "\n\n", text)

    # Real-estate developer tickers: remove cross-sector bank-peer leakage.
    # The LLM sometimes compares RE developers to UAE banks (FAB, Emirates NBD,
    # etc.) because they share the same exchange. Narrative sentences and peer
    # table rows referencing bank-specific metrics are off-thesis for RE and
    # confuse readers. News link lines are preserved (external headlines are
    # not peer analysis — they are editorial context).
    if fs.sector_subtype == SectorSubtype.REAL_ESTATE_DEVELOPER:
        # Detect the specific bank/finance-peer terms we want to scrub from
        # RE reports. Deliberately narrow so we don't over-strip:
        #  • Named UAE bank tickers/names used as peers: FAB, First Abu Dhabi
        #    Bank, DIB, Emirates NBD, ENBD.
        #  • Banking-specific metrics: banking margin, NIM, banking peer, etc.
        _BANK_PEER_TERMS = [
            r"\bFAB\b",                              # First Abu Dhabi Bank ticker
            r"First\s+Abu\s+Dhabi\s+Bank",
            r"\bDIB\b",                              # Dubai Islamic Bank ticker
            r"Emirates\s+NBD",
            r"\bENBD\b",
            r"banking\s+(?:margin|average|peer)",
            r"finance[\-\s]sector\s+peer",
            r"finance[\-\s]sector\s+average",
        ]
        # News-link pattern: lines starting with a bullet + emoji + link
        _NEWS_LINK_RE = re.compile(
            r"^\s*[-*]\s*[⚪🔴🟢🟡⚠️🔵🟠]\s*\[", flags=re.UNICODE
        )
        _TABLE_ROW_RE = re.compile(r"^\s*\|")
        _TABLE_DIV_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

        _bank_removed = 0
        _bank_kept_lines = []
        for _bline in text.split("\n"):
            # Check if this line contains any bank-peer term
            _matched = any(
                re.search(_tp, _bline, re.IGNORECASE) for _tp in _BANK_PEER_TERMS
            )
            if not _matched:
                _bank_kept_lines.append(_bline)
                continue
            # News links — always keep (they are external headlines, not analysis)
            if _NEWS_LINK_RE.search(_bline):
                _bank_kept_lines.append(_bline)
                continue
            # Table divider rows (---|---) — always keep
            if _TABLE_DIV_RE.match(_bline):
                _bank_kept_lines.append(_bline)
                continue
            # Peer-table data rows and narrative lines → remove
            _bank_removed += 1
            # (line is dropped by not appending)

        if _bank_removed:
            text = "\n".join(_bank_kept_lines)
            # Collapse any triple-blank-line cascades left behind
            text = re.sub(r"\n{3,}", "\n\n", text)
            audit.corrections.append(
                Correction(
                    "peer_scrub",
                    f"{_bank_removed} bank-peer reference(s) in RE report",
                    "removed",
                    "re_bank_peer_scrub",
                )
            )

    # Real-estate tickers: fix wrong "Finance / Real Estate" sector label
    # TV sometimes classifies RE developers under Finance sector upstream.
    # The FactSheet has the correct sector; patch the metadata line.
    if fs.is_real_estate:
        sector_display = fs.sector if fs.sector and fs.sector != "Unknown" else "Real Estate"
        # Pattern: **Sector/Industry:** Finance / Real Estate Operations
        new_text, n = re.subn(
            r"(\*\*Sector/Industry:\*\*\s*)Finance\s*/\s*(Real\s+Estate[^\n]*)",
            rf"\1{sector_display} / \2",
            text,
        )
        if n:
            audit.corrections.append(
                Correction(
                    "sector_label",
                    "Finance / Real Estate (wrong TV classification)",
                    f"{sector_display} / ...",
                    "re_sector_label_fix",
                )
            )
            text = new_text

        # Also fix the plain "**Sector:** <wrong>" report header.
        # analytics_builder emits "**Sector:** {fund.sector}", which upstream
        # data mislabels as "Finance" for RE developers (EMAAR, ALDAR). The
        # FactSheet has the authoritative sector — rewrite the header value
        # only (leave the rest of the header line, e.g. Industry, untouched).
        def _fix_sector_header(m: "re.Match") -> str:
            current = m.group(2).strip()
            if re.search(r"real\s*estate", current, re.IGNORECASE):
                return m.group(0)  # already correct — no change
            return f"{m.group(1)}{sector_display}{m.group(3)}"

        new_text2, n2 = re.subn(
            r"(\*\*Sector:\*\*\s*)([^\n|]+?)(\s*(?:\||\n|$))",
            _fix_sector_header,
            text,
        )
        if n2 and new_text2 != text:
            audit.corrections.append(
                Correction(
                    "sector_label",
                    "Sector header (wrong upstream classification)",
                    sector_display,
                    "re_sector_header_fix",
                )
            )
            text = new_text2

    return text


__all__ = [
    "Correction",
    "ReconciliationAudit",
    "reconcile_report",
]
