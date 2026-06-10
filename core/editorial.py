"""
core/editorial.py
─────────────────
Two-tier editorial cleanup for EisaX reports.

light_cleanup(text)       → fast pass before UI delivery
full_editorial_pass(text) → institutional restructure for PDF/export

Guardrails (both passes):
  - numeric_guard   : verdict / significant numbers must not change → fallback
  - truncation_guard: output < 55 % of input length → fallback
  - latency metric  : logged as editorial_light_ms / editorial_full_ms

Layer 2 (brain / internal JSON) is never touched here — only the reply string.
"""
import os
import re
import time
import logging
import requests
from typing import Optional, Tuple

log = logging.getLogger("editorial")

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_DS_URL = "https://api.deepseek.com/v1/chat/completions"

# ── Prompts ────────────────────────────────────────────────────────────────────

_LIGHT_SYSTEM = """\
You are a concise editorial cleanup tool for institutional investment reports.
Your ONLY job: remove redundancy and fix weak phrasing. Never rewrite analysis.

Rules:
- Remove duplicate paragraphs or sections that repeat the same conclusion
- Remove LLM filler openers: "Based on the analysis", "It is worth noting",
  "In conclusion", "To summarize", "As mentioned earlier"
- Remove raw debug markers, "Raw Report" sections, or score block repetitions
- Fix these exact phrases:
    "near support and resistance"  →  "range-bound"
    "stretched entry"              →  "entry conditions unfavorable"
    "weak confirmation"            →  "trend lacks confirmation"
- CRITICAL: Keep ALL numbers, verdicts, targets, scores, probabilities UNCHANGED
- Do NOT shorten the report significantly — every section must survive
- Do NOT add new content
- Return ONLY the cleaned report text, no commentary, no notes."""

_FULL_SYSTEM = """\
You are the final institutional quality-control and editorial layer for EisaX.
Transform the raw report into a clean, production-grade institutional report.

CRITICAL RULES — DO NOT BREAK:
1. DO NOT change: verdict, scores, numbers, targets, scenarios, probabilities, triggers
2. DO NOT add new analysis
3. DO NOT remove important insight — only compress repetition
4. DO NOT keep raw/debug/LLM artifacts

STRUCTURE — rebuild into this EXACT order, no other sections:

### 1. Executive Summary
4-6 lines: what the asset is · current positioning · core opportunity · core risk

### 2. Investment Verdict
Verdict (HOLD/BUY/SELL/REDUCE) + 1-2 line justification
Fundamental View: [one line]
Timing Signal: [one line]
Conviction: [High/Medium/Low + reason]

### 3. Key Drivers
3-5 strongest only, no repetition from other sections

### 4. Fundamental Analysis
Profitability · Growth · Valuation · Balance Sheet
Clean tight paragraphs, no repetition

### 5. Technical Context
Trend: · Momentum: · Volume: · Key Levels (S/R):

### 6. Risk Framework
Format each risk as:
[Risk Name] (Severity)
Cause → Impact → Implication

### 7. Catalysts
Short list: earnings · macro triggers · sector drivers

### 8. Valuation & Scenarios
Keep tables or structured ranges, remove duplicate explanations

### 9. Portfolio Role
Why this asset exists in a portfolio · income/growth/defensive role

### 10. Timing & Why Now
MAX 2 short paragraphs: positioning · timing edge · catalyst proximity
No repetition from summary

### 11. Action Framework
Upgrade Trigger: [one sentence]
Downgrade Trigger: [one sentence]
Invalidation: [one sentence]
No-Action Case: [one sentence]

CLEANUP RULES:
- Remove: emojis, duplicated sections, repeated explanations, UI artifacts, debug text
- Fix weak phrasing (see Light rules above)
- Tone: calm, professional, neutral — no hype, no over-explanation

Return ONLY the final cleaned report. No commentary, no notes."""

# ── Polish-condensed system prompt ─────────────────────────────────────────────
# Used by polish_condensed() — extracts & condenses into 13-section institutional
# report. Target: 8,000–12,000 chars. Does NOT try to reproduce the full report.

_POLISH_CONDENSED_SYSTEM = """\
You are EisaX Institutional Report Compiler.

Your job: read the raw analysis and produce a complete, dense institutional report.
Do NOT reproduce the raw report verbatim. EXTRACT the key facts and WRITE a professional version.

LENGTH REQUIREMENTS:
- Total output: 4,000–8,000 characters (institutional density — high signal/noise)
- Each bullet: maximum 2 sentences. Each paragraph: 3–5 sentences max.
- Lead every bullet with the fact, finding, or trigger — never with narrative setup.
- Use declarative voice. Cut hedging filler ("it is worth noting", "as we can see", "based on the analysis").
- One idea per sentence. No multi-clause sentences with three connectors.

INSTITUTIONAL WRITING STYLE (mandatory):
- GOOD: "Primary risk is momentum extension near resistance with RSI > 70."
- BAD:  "The primary risk currently is the stock's price extension above the preferred entry zone, which suggests caution may be warranted."
- GOOD: "Revenue declined 2.9% YoY; earnings grew 21% on margin expansion."
- BAD:  "It is worth noting that revenue showed a decline of approximately 2.9% year-over-year, while at the same time earnings managed to grow by 21%."

CALIBRATED LANGUAGE — preserve uncertainty without sounding weak:
- "Evidence suggests..." / "Data supports..." / "Signal indicates..." (calibrated declarative)
- "Limited Fundamental Visibility" — when financial coverage is partial
- "Awaiting Confirmation" — when momentum/breakout unconfirmed
- "Technical-Led Thesis" — when fundamentals are sparse but technicals support
- "Moderate Evidence Strength" — when partial confluence exists
- "Pullback Preferred" — when entry conditions are extended
NEVER use: "Low Confidence", "Low Conviction", "Weak Thesis", "We believe", "It is clear that".

DECISION HIERARCHY — strict consistency rule:
The raw report contains a scorecard with the canonical Verdict (Buy/Hold/Reduce/Sell),
Timing (Attractive/Neutral/Extended), Evidence (Limited/Moderate/Strong), and Action.
- COPY the Verdict from the scorecard EXACTLY in every section that references it.
- DO NOT invent a different verdict in Executive Summary, Investment Verdict, or Portfolio Role.
- DO NOT introduce a phrase like "Fundamental Verdict: BUY" when the scorecard says HOLD.
- Use "Fundamental View" to describe business quality narrative — but never override the Verdict.
- Action / "what to do now" must match the canonical Action (Scale In / Wait / Reduce Exposure / Hold Steady).

BANNED WALL-STREET CLICHÉS (do not write these):
- "the market is pricing in", "structural resilience", "asymmetric risk/reward"
- "lowest-cost integrated producer", "modestly higher cycle", "compelling entry point"
- "significant upside optionality", "structural cash-flow resilience"
Use direct, concrete sentences: "Aramco's margins are industry-leading at current oil prices."
Not: "Aramco offers compelling exposure to structurally resilient cash flows."

BANNED THEATRICAL / TWITTER-STYLE FRAMING (institutional tone violation):
- "Thesis Kill Shot", "Kill Shot", "Conviction Buy"
- "smart money", "diamond hands", "paper hands", "moonshot", "to the moon"
- "FOMO", "YOLO", "rocket trade", "destined to", "guaranteed to"
- "crushing earnings", "blowout quarter"
Replace dramatic framing with neutral institutional phrasing:
- "Primary Thesis Risk:" not "Thesis Kill Shot:"
- "Strong quarter" not "blowout quarter"
- "Expected to" not "guaranteed to"
The target tone is institutional buy-side memo — NOT hedge-fund Twitter.

DUPLICATION GUARD: Say each idea ONCE.
Bad (3 ways to say the same thing):
  "RSI is overbought. The setup is stretched. Timing is mixed awaiting pullback."
Good (one declarative sentence):
  "RSI at 73 signals extended technicals; await pullback for entry."

SENTENCE VARIATION: Avoid identical openers and rhythms across sections.
Do not start every paragraph with "The company..." or "Aramco's...". Vary sentence
length and structure. Mix short declaratives with longer analytical sentences.
- Bad rhythm: "Aramco generates strong cash flow. Aramco maintains low debt. Aramco pays a high dividend."
- Good rhythm: "Cash generation is industry-leading. With near-zero leverage, the dividend remains well-covered through cycle bottoms."

CANONICAL TONE WITHOUT TEMPLATE SMELL:
- Same vocabulary across reports (Verdict / Timing / Evidence / Risk axes).
- Different syntax for similar ideas (e.g. "Risk/reward is unattractive" vs "Upside is constrained vs downside" vs "Margin of safety is limited").
- The post-process layer (controlled_variability) will rotate canonical phrases automatically — your job is to write distinct prose, not template prose.

NO MECHANISTIC RULES: Avoid retail-TA-YouTube style pseudo-precision.
Bad: "If RSI drops below 30 AND price closes below SMA200 for 2 consecutive days, the bearish timing call fails."
Good: "Sustained weakness below long-term trend support would invalidate the timing framework."
The institutional reader does not want stochastic triggers — they want the principle.

NO PARALLEL VERDICT VOCABULARY: Use ONLY these labels in the long report:
- Verdict: Buy / Hold / Reduce / Sell  (matches the scorecard exactly)
- Timing: Attractive / Neutral / Extended
- Evidence: Limited / Moderate / Strong
- Risk: High / Moderate / Low
Banned in long report (will be auto-rewritten if you use them):
- "Tactical Underweight", "Tactical Overweight", "Underweight", "Overweight"
- "ACCUMULATE", "AVOID", "Hold Steady", "Maintain position"
- "Conviction:", "Confidence:", "Fundamental Conviction:"
- "Fundamental Verdict: X" when X differs from canonical Verdict

RISK FRAMEWORK HIERARCHY: When discussing risks, label them by category:
- Market Beta Risk: Low/Moderate/High
- Commodity / Cycle Risk: Low/Moderate/High
- Timing Risk: Low/Moderate/High
- Portfolio Risk: Low/Moderate/High
Never write "Risk: High" then "low beta risk" then "defensive holding" — pick a category and be specific.

CRITICAL PRESERVATION RULES — these must survive exactly:
- Verdict (BUY/SELL/HOLD/REDUCE/ACCUMULATE/AVOID)
- Ticker symbol
- Current price and price target
- EisaX Score (e.g. 65/100)
- Conviction level
- Valuation multiples (P/E, forward P/E)
- Scenario prices (bear/base/bull)
- Key percentages (dividend yield, revenue growth, upside/downside)
- Support and resistance levels

Do NOT: invent numbers, add analysis not in the source, start with "EisaX Intelligence Report",
include preamble phrases like "Institutional version ready", "Prepared for institutional use",
"EisaX Institutional Report", or any meta-commentary about the report type or format.
Return ONLY the report content — no wrapper text before or after.

REMOVE ENTIRELY:
- ASCII price charts (lines of ░█▓▒ or $xx |...)
- Scorecard bar charts with Unicode blocks (████░░)
- Raw debug markers, "FACT-CHECK" tables, "Raw Report" headers
- Duplicate paragraphs, LLM filler openers ("Based on the analysis...", "It is worth noting...")
- Peer comparison tables with more than 4 peers (keep the 3–4 most relevant)
- Score Trend sparklines and cache data sections

DO NOT INVENT MISSING DATA: If a field appears as "Not reliable", "N/A", "—", or is absent
from the source, do NOT generate a numeric value for it. Write "unavailable" or omit the field.
This applies especially to: beta, short interest, float %, insider ownership.

TONE: Calm, institutional, professional. No hype. No over-explanation.

OUTPUT STRUCTURE — write all 13 sections in this exact order:

### Signal Snapshot
One line using this exact labeled format with " | " separators (not "·"):
Ticker: {TICKER} | Verdict: {VERDICT} | Price: {PRICE} {CURRENCY} | EisaX Score: {SCORE}/100 | Evidence Strength: {LEVEL}

The Evidence Strength label must use canonical taxonomy:
- "Strong" — verified fundamentals, multi-signal confluence
- "Moderate" — partial fundamentals, mixed signals
- "Limited" — low fundamental coverage, technical-led thesis
NEVER write "Conviction: Low/Medium/High" — that vocabulary is deprecated.
Currency: SAR for Saudi/ADX stocks, USD for NYSE/NASDAQ, AED/KWD/QAR/OMR as appropriate.
Write {TICKER} as plain text only — do NOT wrap it in a markdown link or add any URL around it.
Correct: Ticker: 2222.SR | ...
Wrong:   Ticker: [2222.SR](http://2222.SR) | ...
Wrong:   Ticker: [[2222.SR](http://2222.SR)](http://2222.SR) | ...

### Executive Summary
3–4 sentences max. State: what the asset is, current positioning, the single most material opportunity, the single most material risk. Be specific with numbers. No setup, no closing line.

### Investment Verdict
Verdict: [BUY/SELL/HOLD/REDUCE] — [1 sentence justification, declarative]
Fundamental View: [one line — lead with fact]
Timing Signal: [one line — lead with fact]
Evidence Strength: [High/Moderate/Limited Fundamental Visibility/Awaiting Confirmation] — [one short sentence reason]

### Key Drivers
3–4 bullets. Each bullet: max 2 sentences, lead with the driver/finding. No repetition from other sections.

### Fundamental Analysis
3 short paragraphs (3–4 sentences each):
Para 1 — Revenue/earnings: growth rates, EPS, trajectory
Para 2 — Valuation: P/E, forward P/E, fair value, premium/discount
Para 3 — Balance sheet or dividend: yield, leverage, capital position

### Technical Context
Trend: [one complete sentence with price vs SMA]
Momentum: [one complete sentence with RSI and MACD values]
Volume: [one sentence]
Key Levels (S/R): list exact support and resistance prices

### Risk Framework
Cover 2–4 risks. Each block:
[Risk Name] (Severity: High/Medium/Low)
Cause → Impact → Implication
[2–3 sentences elaborating on the risk]

### Catalysts
5–8 bullet points covering: earnings dates, macro triggers, sector-specific drivers, company-specific events.

### Valuation & Scenarios
Bear / Base / Bull scenario table with: multiple, implied price, upside/downside, probability.
Keep all original numbers. Add 1 sentence of context per scenario.

### Portfolio Role
3–4 sentences: income vs growth vs defensive role, correlation characteristics, who this suits.

### Timing & Why Now
Paragraph 1: Current entry conditions, distance from support, technical setup.
Paragraph 2: Catalyst timing, what would change the timing thesis.
No repetition from Executive Summary.

### Action Framework
Upgrade Trigger: [one complete sentence — what specific event would prompt upgrade]
Downgrade Trigger: [one complete sentence — what event would prompt downgrade]
Invalidation: [one complete sentence — what fully breaks the thesis]
No-Action Case: [one complete sentence — when to hold and do nothing]

### Audit Trail
EisaX AI | Live market data | For informational purposes only — not financial advice.

Return ONLY the formatted report. No preamble, no notes, no commentary after the report."""


# ── Weak-phrase map (applied by rule in both passes) ──────────────────────────

_WEAK_PHRASES = [
    # ── Terminology calibration (replace weak labels with institutional language) ──
    (r"\bLow Conviction\b",                           "Limited Fundamental Visibility"),
    (r"\bLow Confidence\b",                           "Awaiting Confirmation"),
    (r"\bWeak Thesis\b",                              "Technical-Led Thesis"),
    (r"\bWeak Conviction\b",                          "Limited Fundamental Visibility"),
    (r"\bTiming\s+LOW\b",                             "Pullback Preferred"),
    (r"\bTiming:\s*LOW\b",                            "Timing: Pullback Preferred"),
    (r"\bConviction:\s*Low\b",                        "Evidence Strength: Limited"),
    (r"\bConviction:\s*Medium\b",                     "Evidence Strength: Moderate"),
    (r"\bConviction:\s*High\b",                       "Evidence Strength: High"),
    (r"\blow confidence\b(?!\s*\()",                  "limited fundamental visibility"),
    # ── Verbosity removal ──────────────────────────────────────────────────────
    (r"\bnear support and resistance\b",              "range-bound"),
    (r"\bstretched entry\b",                          "entry conditions unfavorable"),
    (r"\bweak confirmation\b",                        "trend lacks confirmation"),
    (r"\bBased on (?:the |this |our )?analysis[,.]?\s*", ""),
    (r"\bIt is worth noting that\s*",                 ""),
    (r"\bIt is important to note that\s*",            ""),
    (r"\bIn conclusion[,.]?\s*",                      ""),
    (r"\bTo summarize[,.]?\s*",                       ""),
    (r"\bAs mentioned (?:above|earlier|previously)[,.]?\s*", ""),
    (r"\bAs we can see[,.]?\s*",                      ""),
    (r"\bIt should be noted that\s*",                 ""),
    (r"\bWe believe that\s*",                         "Evidence indicates "),
    (r"\bWe think that\s*",                           "Analysis suggests "),
    (r"\bIt is clear that\s*",                        ""),
    (r"\bIn order to\b",                              "To"),
    (r"\bdue to the fact that\b",                     "because"),
    (r"\bat this point in time\b",                    "currently"),
    # ── Pseudo-Wall-Street prose (hedge-fund cosplay) ──────────────────────
    (r"\bthe market is pricing in\b",                 "the market reflects"),
    (r"\bstructural cash-flow resilience\b",          "strong cash generation"),
    (r"\bstructural resilience\b",                    "resilience"),
    (r"\bworld[-’']s lowest-cost integrated producer\b", "low-cost integrated producer"),
    (r"\basymmetric risk[/-]reward\b",                "favorable risk/reward"),
    (r"\bmodestly higher cycle\b",                    "modest upward cycle"),
    (r"\bmodestly[ -]higher\s+oil\s+cycle\b",         "modestly higher oil prices"),
    (r"\bunderestimating\s+the\s+structural\b",       "underestimating the"),
    (r"\bsignificant\s+(?:cash[ -]flow)?\s*generation\s+capacity\b", "strong cash generation"),
    (r"\bcompelling\s+entry\s+(?:point|opportunity)\b","attractive entry"),
    (r"\bsignificant\s+upside\s+optionality\b",       "meaningful upside"),
    # Synonym chain dedup (same idea repeated three different ways)
    (r"\boverbought\s+near\s+resistance\b",           "RSI elevated near resistance"),
    (r"\bstretched\s+setup\b",                        "extended technicals"),
    (r"\btiming\s+mixed\b",                           "timing unclear"),
    # ── Theatrical / Twitter-hedge-fund cosplay (Phase 4) ──────────────────
    # Target tone is institutional buy-side memo. Strip theatrical framing.
    (r"\*{0,2}Thesis\s+Kill\s+Shot\*{0,2}\s*:",       "**Primary Thesis Risk:**"),
    (r"\bKill\s+Shot\b",                              "Primary Risk"),
    (r"\bConviction\s+Buy\b",                         "Buy"),
    (r"\bsmart\s+money\b",                            "institutional flow"),
    (r"\bdumb\s+money\b",                             "retail flow"),
    (r"\bdiamond\s+hands\b",                          "long-duration conviction"),
    (r"\bpaper\s+hands\b",                            "short-duration positioning"),
    (r"\bmoon(?:shot)?\b",                            "outsized return"),
    (r"\bto\s+the\s+moon\b",                          "materially higher"),
    (r"\brocket\s+(?:ship\s+)?(?:trade|setup|move)\b","high-momentum setup"),
    (r"\bYOLO\b",                                     "concentrated bet"),
    (r"\bFOMO\b",                                     "momentum chasing"),
    (r"\bhedge[- ]fund\s+(?:bro|cowboy)\b",           "institutional manager"),
    # Banned dramatic framing
    (r"\bdestined\s+to\b",                            "likely to"),
    (r"\bguaranteed\s+to\b",                          "expected to"),
    (r"\bcrushing\s+(?:results|earnings)\b",          "strong results"),
    (r"\bblow\s*out\s+(?:results|earnings|quarter)\b","strong quarter"),
    (r"(?m)^---\n#{1,3}\s*(?:Raw Report|Debug Info|Source Data)[^\n]*\n[\s\S]*?(?=\n---|\Z)", ""),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def rule_based_clean(text: str, ticker: str = "") -> str:
    """
    Instant rule-based editorial pass — no LLM, safe for every analyze request.

    Flow: raw → _rule_based(text, ticker) → numeric_guard(raw, rule)
      fail → raw (numbers would change)
      pass → rule_cleaned

    The ticker arg drives controlled variability — same ticker always gets the
    same phrasing variants, different tickers get different ones.

    Logs: editorial_mode=rule_based_only, raw_len, rule_clean_len, llm_skipped=true.
    """
    if not text or len(text) < 200:
        return text

    _t0 = time.monotonic()
    _original = text

    _rule_text = _rule_based(text, ticker=ticker)
    _r_ok, _r_msg = _numeric_guard(_original, _rule_text)
    if not _r_ok:
        log.warning("[editorial.rule] numeric guard failed — returning raw: %s", _r_msg)
        return text

    _latency_ms = int((time.monotonic() - _t0) * 1000)
    log.info(
        "[editorial] editorial_mode=rule_based_only raw_len=%d rule_clean_len=%d "
        "delta=%d llm_skipped=true latency_ms=%d",
        len(_original), len(_rule_text), len(_original) - len(_rule_text), _latency_ms,
    )
    return _rule_text


def light_cleanup(text: str) -> str:
    """
    Fast editorial pass — rule-based (instant) + short DeepSeek call (5-8s).

    Flow:
      raw → rule_based → numeric_guard(raw, rule)
        fail → raw               (rule-based itself changed numbers)
        pass → safe_rule_text
             → LLM edit → numeric_guard(raw, llm)
               fail → safe_rule_text
               pass → llm_text

    Logs: editorial_light_ms, changed_chars, numeric_guard_passed, fallback_reason.
    """
    if not text or len(text) < 200:
        return text

    _t0 = time.monotonic()
    _original = text
    _model = "rule-based"
    _fallback_reason: Optional[str] = None
    _guard_passed = True

    # Stage 1 — rule-based + guard against raw
    _rule_text = _rule_based(text)
    _r_ok, _r_msg = _numeric_guard(_original, _rule_text)
    if not _r_ok:
        _fallback_reason = f"rule_numeric: {_r_msg}"
        log.warning("[editorial.light] rule-based guard failed — returning raw: %s", _r_msg)
        text = _original
    else:
        text = _rule_text  # safe rule-based confirmed

        # Stage 2 — optional LLM pass + guard against raw
        if DEEPSEEK_KEY:
            try:
                _model = "deepseek-v4-flash"
                _edited = _llm_call(text, _LIGHT_SYSTEM, max_tokens=2500, timeout=25)

                _tok, _tmsg = _truncation_guard(text, _edited)
                if not _tok:
                    _fallback_reason = f"truncation: {_tmsg}"
                else:
                    _nok, _nmsg = _numeric_guard(_original, _edited)
                    if not _nok:
                        _guard_passed = False
                        _fallback_reason = f"llm_numeric: {_nmsg}"
                        text = _rule_text      # fallback to verified rule-based
                    else:
                        text = _edited         # LLM output accepted

            except Exception as exc:
                _fallback_reason = f"exception: {exc}"
                # text stays as verified _rule_text

    _latency_ms = int((time.monotonic() - _t0) * 1000)
    log.info(
        "[editorial] pass=light model=%s latency_ms=%d changed_chars=%d "
        "numeric_guard_passed=%s fallback_reason=%s",
        _model, _latency_ms, abs(len(text) - len(_original)),
        _guard_passed, _fallback_reason or "none",
    )
    return text


def full_editorial_pass(text: str) -> str:
    """
    Full institutional editorial pass — used for PDF/export only.
    Gated by caller (polish request flag). Latency not critical (~20-30s).

    Flow:
      raw → rule_based → numeric_guard(raw, rule)
        fail → raw
        pass → safe_rule_text → LLM → numeric_guard(raw, llm)
          fail → safe_rule_text
          pass → llm_text
    """
    if not text or len(text) < 200:
        return text

    _t0 = time.monotonic()
    _original = text
    _fallback_reason: Optional[str] = None
    _guard_passed = True

    # Stage 1 — rule-based + guard
    _rule_text = _rule_based(text)
    _r_ok, _r_msg = _numeric_guard(_original, _rule_text)
    if not _r_ok:
        _fallback_reason = f"rule_numeric: {_r_msg}"
        log.warning("[editorial.full] rule-based guard failed — returning raw: %s", _r_msg)
        text = _original
    else:
        text = _rule_text

        # Stage 2 — LLM full pass + guard (only runs when Stage 1 passed)
        if DEEPSEEK_KEY:
            try:
                _edited = _llm_call(text, _FULL_SYSTEM, max_tokens=4000, timeout=60)

                _tok, _tmsg = _truncation_guard(text, _edited)
                if not _tok:
                    _fallback_reason = f"truncation: {_tmsg}"
                else:
                    _nok, _nmsg = _numeric_guard(_original, _edited)
                    if not _nok:
                        _guard_passed = False
                        _fallback_reason = f"llm_numeric: {_nmsg}"
                        text = _rule_text      # fallback to verified rule-based
                    else:
                        text = _edited         # LLM output accepted

            except Exception as exc:
                _fallback_reason = f"exception: {exc}"
                # text stays as verified _rule_text

    _latency_ms = int((time.monotonic() - _t0) * 1000)

    log.info(
        "[editorial] pass=full model=deepseek-chat latency_ms=%d changed_chars=%d "
        "numeric_guard_passed=%s fallback_reason=%s",
        _latency_ms, abs(len(text) - len(_original)),
        _guard_passed, _fallback_reason or "none",
    )

    return text


# ── Guards ─────────────────────────────────────────────────────────────────────

def _truncation_guard(original: str, edited: str) -> Tuple[bool, str]:
    """Fail if edited is less than 55% of original length."""
    if not original:
        return True, ""
    ratio = len(edited) / len(original)
    if ratio < 0.55:
        return False, f"{len(edited)}/{len(original)} chars ({ratio:.0%})"
    return True, ""


def _numeric_guard(original: str, edited: str) -> Tuple[bool, str]:
    """
    Fail if verdict changed, or if more than 2 significant numbers
    from the original are absent from the edited text.
    """
    # Verdict check
    _VERDICT_RE = re.compile(
        r'\b(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|REDUCE|ACCUMULATE|AVOID)\b', re.I
    )
    orig_v = _VERDICT_RE.search(original)
    edit_v = _VERDICT_RE.search(edited)
    if orig_v and edit_v:
        if orig_v.group(0).upper() != edit_v.group(0).upper():
            return False, f"verdict {orig_v.group(0).upper()} → {edit_v.group(0).upper()}"

    # Significant number check (skip obvious year/date values)
    _SKIP_VALS = {str(y) for y in range(2020, 2031)}
    orig_nums = {
        m.group(0) for m in re.finditer(r'\b\d{1,6}(?:\.\d{1,2})?\b', original)
        if m.group(0) not in _SKIP_VALS
    }
    missing = [n for n in orig_nums if n not in edited]
    if len(missing) > 2:
        return False, f"{len(missing)} numbers missing: {missing[:5]}"

    return True, ""


def _check_unreliable_fields(original: str, polished: str) -> Tuple[bool, list]:
    """
    Verify that the polished output does not invent numeric values for fields
    that are missing or explicitly marked unreliable in the original text.

    Returns (passed: bool, violated_fields: list[str]).

    To add new fields: append a tuple (name, unreliable_re, invented_re) to
    _FIELD_SPECS below.  unreliable_re matches the field in original when it is
    flagged as bad/absent; invented_re matches a fabricated numeric value in polished.
    """
    _FIELD_SPECS = [
        (
            "beta",
            # unreliable_re — marks beta as bad in source
            re.compile(
                r'\bbeta[^a-z\n]{0,30}'
                r'(?:not\s+reliable|n/?a|unavailable|not\s+available|–|—|\bunknown\b)',
                re.I,
            ),
            # invented_re — any form of LLM-invented numeric beta value.
            # Must stay in sync with _INVENTED_BETA_RE used by _strip_invented_beta.
            _INVENTED_BETA_RE,
            # present_re — any mention of beta in source
            re.compile(r'\bbeta\b', re.I),
        ),
    ]

    violations: list = []
    for field_name, unreliable_re, invented_re, present_re in _FIELD_SPECS:
        field_in_original = bool(present_re.search(original))
        field_unreliable = (not field_in_original) or bool(unreliable_re.search(original))
        if field_unreliable and invented_re.search(polished):
            violations.append(field_name)

    return len(violations) == 0, violations


def _is_beta_unreliable(text: str) -> bool:
    """
    Return True if beta should be treated as unavailable in this report.

    Unreliable when:
      - the word "beta" is completely absent from the source text, OR
      - beta appears alongside an unreliability marker (N/A, Not reliable,
        unavailable, not available, — , –, unknown)
    """
    _BETA_PRESENT = re.compile(r'\bbeta\b', re.I)
    _BETA_BAD = re.compile(
        r'\bbeta[^a-z\n]{0,30}'
        r'(?:not\s+reliable|n/?a|unavailable|not\s+available|–|—|\bunknown\b)',
        re.I,
    )
    if not _BETA_PRESENT.search(text):
        return True   # absent entirely
    return bool(_BETA_BAD.search(text))


# Patterns that indicate the LLM invented a beta numeric value
_INVENTED_BETA_RE = re.compile(
    r'\blow[- ]beta\s+(?:of|at|:)\s*[-+]?\d+(?:\.\d+)?'        # "low beta of 0.26" — number required
    r'|\bbeta\s*(?:of|at|:)\s*[-+]?\d+(?:\.\d+)?'             # "beta of 0.26", "beta: 0.26"
    r'|\bbeta\s*\(\s*[-+]?\d+(?:\.\d+)?\s*\)'                 # "beta (0.26)"
    r'|\bBeta:\s*[-+]?\d+(?:\.\d+)?',                           # "Beta: 0.26" — colon required
    re.I,
)


def _strip_invented_beta(text: str, language: str = "en") -> str:
    """Replace invented beta numeric phrases; also clean up malformed LLM suffix tokens."""
    replacement = "بيانات بيتا غير متاحة" if language == "ar" else "beta data is unavailable"
    text = _INVENTED_BETA_RE.sub(replacement, text)
    # Clean up LLM artifacts like "unavailablex" appended directly after the phrase
    text = re.sub(r'\bbeta data is unavailable\w+', 'beta data is unavailable', text, flags=re.I)
    return text


def _critical_number_guard(original: str, polished: str) -> Tuple[bool, str]:
    """
    Check that numbers associated with key financial metrics survive condensation.

    Strategy
    --------
    Only numbers on lines that contain financial-context keywords are eligible.
    Lines about peers, market context, cache, news, volume, and timestamps are
    skipped entirely — those numbers are legitimately dropped in a condensed report.
    At most 20 critical numbers are collected (first 20 by document order).
    At least 70 % of those must appear in the polished output.

    Comma normalisation: "1,211" in polished is treated as "1211" so
    thousand-separator formatting differences don't count as missing.

    Logged as: [polish_guard] critical_numbers passed=true|false retained=X/Y missing=[...]
    """
    _KEYWORD_RE = re.compile(
        r'\b(?:price|target|fair.{0,5}value|upside|downside|yield|'
        r'support|resistance|bear|base|bull|score|eisax|'
        r'brent|crude|wti|forward\s+p/?e|p/?e\b|eps|revenue|ebitda|'
        r'conviction|signal|timing|dcf|consensus)\b',
        re.I,
    )
    # Skip lines whose primary subject is something other than this asset.
    # \bpeer (no trailing \b) matches: peer, peers, peer1, peer2, peer company …
    _SKIP_LINE_RE = re.compile(
        r'\bpeer'
        r'|\bmarket\s+context\b'
        r'|\bcache\b'
        r'|\bnews\s+title\b'
        r'|\bvolume\b'
        r'|\btimestamp\b'
        r'|\bunrelated\b'
        r'|\bticker\s+ref\b',
        re.I,
    )
    _NUM_RE   = re.compile(r'\b\d{1,6}(?:\.\d{1,4})?\b')
    _YEAR_SET = {str(y) for y in range(2019, 2032)}

    seen: set  = set()
    critical: list = []
    for line in original.splitlines():
        if len(critical) >= 20:
            break
        if _SKIP_LINE_RE.search(line):
            continue
        if not _KEYWORD_RE.search(line):
            continue
        for m in _NUM_RE.finditer(line):
            n = m.group(0)
            if n not in _YEAR_SET and n not in seen:
                seen.add(n)
                critical.append(n)
                if len(critical) >= 20:
                    break

    if not critical:
        log.info("[polish_guard] critical_numbers passed=true retained=0/0 missing=[]")
        return True, ""

    # Normalise thousand-separator commas before substring check
    _pol_norm = re.sub(r'(?<=\d),(?=\d{3}\b)', '', polished)

    missing  = [n for n in critical if n not in _pol_norm]
    retained = len(critical) - len(missing)
    retention = retained / len(critical)
    _passed  = retention >= 0.70

    log.info(
        "[polish_guard] critical_numbers passed=%s retained=%d/%d missing=%s",
        "true" if _passed else "false",
        retained, len(critical), missing[:5],
    )

    if not _passed:
        return False, (
            f"critical numbers: {retention:.0%} retained "
            f"({retained}/{len(critical)}) — missing {missing[:5]}"
        )
    return True, ""


# ── Strip markdown link syntax from a ticker string ───────────────────────────
# e.g. [ADNOCGAS.AE](http://ADNOCGAS.AE)  →  ADNOCGAS.AE
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]*\)')


def _clean_ticker(ticker: str) -> str:
    """Return the ticker with all markdown link wrappers removed (loops until stable)."""
    prev = None
    while prev != ticker:
        prev = ticker
        ticker = _MD_LINK_RE.sub(r'\1', ticker).strip()
    return ticker


# ── Phrases the LLM must never emit in the report body ────────────────────────
_META_PHRASE_RE = re.compile(
    r'(?im)^(?:institutional version ready|prepared for institutional use'
    r'|eisax institutional report|this is an institutional version'
    r'|النسخة المؤسسية جاهزة)\s*[\n\r]*',
)


def _strip_meta_phrases(text: str) -> str:
    """Remove any meta-commentary the LLM may have prepended/appended."""
    return _META_PHRASE_RE.sub("", text).strip()


def _strip_ticker_autolinks(text: str) -> str:
    """
    Clean up two classes of malformed markdown ticker links that LLMs generate:

    Rule 1 — Double-wrapped links:
        [[label](url)](url)  →  [label](url)
        Only unwraps when outer URL == inner URL (both are the same fake autolink).
        Real nested links with distinct URLs are left untouched.

    Rule 2 — Plain-ticker autolinks (no exchange-suffix dot in label):
        [NVDA](http://NVDA)  →  NVDA
        Matches when URL is exactly http://{label} (case-insensitive) and the
        label contains no "." — i.e. it is a plain US-style ticker, not a
        Saudi/ADX ticker like 2222.SR that legitimately has a dot.

    Real news/source URLs (domains, https://, paths) are never touched.
    """
    # Rule 1: unwrap [[inner_text](inner_url)](outer_url) when inner_url == outer_url
    def _unwrap_double(m: re.Match) -> str:
        label, inner_url, outer_url = m.group(1), m.group(2), m.group(3)
        if inner_url == outer_url:
            return f"[{label}]({inner_url})"
        return m.group(0)

    text = re.sub(
        r'\[\[([^\]\n]+)\]\(([^)\n]+)\)\]\(([^)\n]+)\)',
        _unwrap_double,
        text,
    )

    # Rule 2: strip [TICKER](http://TICKER) → TICKER when label has no "."
    def _strip_plain_autolink(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if "." not in label and url.lower() == f"http://{label.lower()}":
            return label
        return m.group(0)

    text = re.sub(
        r'\[([^\]\n]+)\]\((http://[^)\s]+)\)',
        _strip_plain_autolink,
        text,
    )

    return text


def _polish_guard(
    original: str,
    polished: str,
    ticker: str = "",
    official_verdict: str = "",
    language: str = "en",
) -> Tuple[bool, str]:
    """
    Guard for condensed polish output.

    Unlike _truncation_guard (output >= 55% of input), this checks that the
    output lands within the target institutional range AND preserves key facts.

    Pass conditions:
      - 5,000 ≤ len(polished) ≤ 12,000 chars  (soft bounds)
      - Length check bypassed (down to absolute floor 2,500) if ≥ 9/12 required
        sections are present AND verdict + ticker + numbers pass
      - Verdict preserved: when official_verdict is supplied it is used as ground
        truth; otherwise extracted from original (legacy fallback)
      - Ticker present in output (if provided)
      - ≥ 40% of significant numbers from original appear in polished
      - Section count ≥ 9/12 checked and logged (not a hard fail above soft min)
    """
    _POLISH_MIN = 5_000
    _POLISH_MAX = 12_000
    _ABSOLUTE_MIN = 2_500   # floor — no output this short is ever acceptable
    _MIN_SECTIONS = 9

    _REQUIRED_SECTIONS = [
        "Signal Snapshot",
        "Executive Summary",
        "Investment Verdict",
        "Key Drivers",
        "Fundamental Analysis",
        "Technical Context",
        "Risk Framework",
        "Catalysts",
        "Valuation & Scenarios",
        "Portfolio Role",
        "Timing & Why Now",
        "Action Framework",
    ]
    _AR_REQUIRED_SECTIONS = [
        "لقطة الإشارات",
        "الملخص التنفيذي",
        "قرار الاستثمار",
        "المحركات الرئيسية",
        "التحليل الأساسي",
        "السياق الفني",
        "إطار المخاطر",
        "المحفزات",
        "التقييم والسيناريوهات",
        "دور الأصل في المحفظة",
        "التوقيت ولماذا الآن",
        "إطار التصرف",
    ]

    out_len = len(polished)
    _sections_to_check = _AR_REQUIRED_SECTIONS if language == "ar" else _REQUIRED_SECTIONS
    section_count = sum(1 for s in _sections_to_check if s in polished)

    def _fail(reason: str) -> Tuple[bool, str]:
        log.warning(
            "[polish_guard] passed=false language=%s length=%d sections=%d/12 reason=%s",
            language, out_len, section_count, reason,
        )
        return False, reason

    # Arabic output must not contain English section headings
    if language == "ar":
        _EN_HEADING_RE = re.compile(
            r'\b(?:Signal Snapshot|Executive Summary|Investment Verdict|Key Drivers|'
            r'Fundamental Analysis|Technical Context|Risk Framework|Catalysts|'
            r'Valuation & Scenarios|Portfolio Role|Timing & Why Now|Action Framework)\b',
            re.I,
        )
        if _EN_HEADING_RE.search(polished):
            return _fail("english_headings_in_arabic_output")

    # Hard upper bound
    if out_len > _POLISH_MAX:
        return _fail(f"output too long: {out_len} chars (max {_POLISH_MAX})")

    # Absolute floor — no bypass possible
    if out_len < _ABSOLUTE_MIN:
        return _fail(f"output too short: {out_len} chars (absolute floor {_ABSOLUTE_MIN})")

    # Soft minimum — bypassed when structure + facts are preserved
    length_bypassed = False
    if out_len < _POLISH_MIN:
        if section_count < _MIN_SECTIONS:
            return _fail(
                f"output too short: {out_len} chars (min {_POLISH_MIN}) "
                f"and only {section_count}/12 sections (need {_MIN_SECTIONS})"
            )
        length_bypassed = True  # sections present — proceed to fact checks

    # Verdict must survive — prefer official_verdict (decision-engine ground truth)
    # over extracting from potentially ambiguous raw text.
    _VERDICT_RE = re.compile(
        r'\b(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|REDUCE|ACCUMULATE|AVOID)\b', re.I
    )
    _AR_VERDICTS = {
        "HOLD": "احتفاظ", "BUY": "شراء", "REDUCE": "تخفيض", "WAIT": "انتظار",
        "SELL": "بيع", "ACCUMULATE": "تراكم",
        "STRONG BUY": "شراء قوي", "STRONG SELL": "بيع قوي", "AVOID": "تجنب",
    }
    if language == "ar":
        if official_verdict:
            _ov = official_verdict.strip().upper()
            _ar_v = _AR_VERDICTS.get(_ov)
            if _ar_v:
                if _ar_v not in polished:
                    return _fail(f"Arabic verdict missing: {_ar_v} (for {_ov})")
            else:
                # No translation known — fall back to English check
                if not _VERDICT_RE.search(polished):
                    return _fail(f"verdict missing: {_ov}")
    else:
        edit_v = _VERDICT_RE.search(polished)
        if official_verdict:
            _ov = official_verdict.strip().upper()
            if not edit_v:
                return _fail(f"verdict missing: {_ov}")
            if edit_v.group(0).upper() != _ov:
                return _fail(f"verdict changed: {_ov} → {edit_v.group(0).upper()}")
        else:
            # Legacy fallback: extract from original text
            orig_v = _VERDICT_RE.search(original)
            if orig_v and not edit_v:
                return _fail(f"verdict missing: {orig_v.group(0).upper()}")
            if orig_v and edit_v and orig_v.group(0).upper() != edit_v.group(0).upper():
                return _fail(f"verdict changed: {orig_v.group(0).upper()} → {edit_v.group(0).upper()}")

    # Ticker must appear (strict)
    # Strip markdown link syntax defensively — ticker may arrive as [SYM](URL)
    if ticker:
        _t = _clean_ticker(ticker)
        _ticker_base = re.sub(r'\.[A-Z]{2}$', '', _t.upper())  # strip exchange suffix
        if _ticker_base not in polished.upper() and _t.upper() not in polished.upper():
            return _fail(f"ticker {_t} missing from polished output")

    # Critical number guard — replaces the broad numeric-retention check.
    # Only validates numbers associated with key financial keywords; ignores
    # peer tables, historical rows, market-context cache data, and news.
    _cn_ok, _cn_msg = _critical_number_guard(original, polished)
    if not _cn_ok:
        return _fail(_cn_msg)

    # Unreliable field check — must not invent values for absent/flagged fields
    _uf_ok, _uf_violations = _check_unreliable_fields(original, polished)
    log.info(
        "[polish_guard] unreliable_field_check passed=%s fields=%s",
        "true" if _uf_ok else "false",
        _uf_violations if _uf_violations else [],
    )
    if not _uf_ok:
        return _fail(f"invented values for unreliable fields: {_uf_violations}")

    _pass_reason = "length_bypassed_by_sections" if length_bypassed else "none"
    log.info(
        "[polish_guard] passed=true language=%s sections=%d/12 fallback=false reason=%s",
        language, section_count, _pass_reason,
    )
    return True, ""


def polish_condensed(
    text: str,
    ticker: str = "",
    verdict: str = "",
    language: str = "en",
) -> Tuple[str, bool, str]:
    """
    Generate a condensed 13-section institutional report from the raw/rule-based text.

    Args:
        text:    Rule-based clean report (source material).
        ticker:  Ticker symbol — used by the guard to confirm it survives.
        verdict: Official decision-engine verdict (e.g. "HOLD").  When supplied,
                 it is injected as a hard rule into the system prompt and used as
                 the ground truth in _polish_guard.  Prevents the LLM from
                 inferring a different verdict from bullish/bearish signal language.

    Returns: (result_text, fallback_bool, reason_str)
      - fallback=False → result_text is the polished condensed report
      - fallback=True  → result_text is the original text unchanged; reason explains why

    Flow:
      source → LLM(system_prompt + optional verdict lock, max_tokens=3500, timeout=90s)
             → strip_meta_phrases + strip_ticker_autolinks
             → _polish_guard (range + verdict + ticker + number retention)
               pass  → (polished, False, "")
               fail  → (original, True, reason)

    Logs: editorial_mode=polish_condensed, latency_ms, fallback_reason, ticker.
    """
    if not text or len(text) < 500:
        return text, True, "source_too_short"

    if not DEEPSEEK_KEY:
        return text, True, "no_llm_key"

    _t0 = time.monotonic()
    _fallback_reason = ""

    # Strip markdown link syntax from ticker so guard/log/prompt get a clean symbol
    # e.g. [ADNOCGAS.AE](http://ADNOCGAS.AE)  →  ADNOCGAS.AE
    _ticker = _clean_ticker(ticker) if ticker else ""

    log.info("[polish] language=%s ticker=%s", language, _ticker or "?")

    # ── Detect beta reliability from source text ──────────────────────────────
    _beta_unreliable = _is_beta_unreliable(text)
    log.info("[polish] beta_reliable=%s", "false" if _beta_unreliable else "true")

    # ── Build system prompt — inject hard verdict lock + beta rule ────────────
    _system = _POLISH_CONDENSED_SYSTEM
    _official_verdict = verdict.strip().upper() if verdict else ""
    if _official_verdict:
        _system += (
            f"\n\nHARD RULE — VERDICT LOCK (non-negotiable):\n"
            f"The official top-level verdict for this report is: {_official_verdict}\n"
            f"You MUST write exactly \"{_official_verdict}\" as the verdict in the "
            f"Investment Verdict section.\n"
            f"Do NOT infer, upgrade, or downgrade the verdict based on:\n"
            f"  - Technical signals (e.g. 'Weak Buy', 'bullish momentum', 'oversold')\n"
            f"  - Analyst consensus ratings\n"
            f"  - Catalysts or upgrade triggers in the report\n"
            f"These are supporting context only. The official verdict is "
            f"{_official_verdict} and must remain {_official_verdict} throughout "
            f"the polished output."
        )
    _AR_V_MAP = {
        "HOLD": "احتفاظ", "BUY": "شراء", "REDUCE": "تخفيض", "WAIT": "انتظار",
        "SELL": "بيع", "ACCUMULATE": "تراكم",
        "STRONG BUY": "شراء قوي", "STRONG SELL": "بيع قوي", "AVOID": "تجنب",
    }

    if _beta_unreliable:
        if language == "ar":
            _system += (
                "\n\nقاعدة صارمة — بيانات بيتا:\n"
                "بيانات بيتا غير متاحة أو غير موثوقة في المصدر. "
                "لا تذكر أي قيمة رقمية لبيتا. "
                "إذا احتجت للإشارة إليها، قل 'بيانات بيتا غير متاحة'."
            )
        else:
            _system += (
                "\n\nHARD RULE — BETA DATA:\n"
                "Beta is unavailable or not reliable in the source data. "
                "Do not mention any numeric beta value. "
                "If needed, say 'beta data is unavailable'."
            )

    if language == "ar":
        _ar_verdict = _AR_V_MAP.get(_official_verdict, "") if _official_verdict else ""
        _system += (
            "\n\nقاعدة صارمة — اللغة العربية (غير قابلة للتفاوض):\n"
            "يجب كتابة التقرير بالكامل باللغة العربية الفصحى.\n"
            "عناوين الأقسام يجب أن تكون بالعربية حصراً وبهذا الترتيب:\n"
            "1. لقطة الإشارات\n"
            "2. الملخص التنفيذي\n"
            "3. قرار الاستثمار\n"
            "4. المحركات الرئيسية\n"
            "5. التحليل الأساسي\n"
            "6. السياق الفني\n"
            "7. إطار المخاطر\n"
            "8. المحفزات\n"
            "9. التقييم والسيناريوهات\n"
            "10. دور الأصل في المحفظة\n"
            "11. التوقيت ولماذا الآن\n"
            "12. إطار التصرف\n"
            "13. سجل التحليل\n"
            "لا تكتب عناوين الأقسام بالإنجليزية. "
            "لا تخلط النثر الإنجليزي — احتفظ فقط برموز الأسهم وأسماء الشركات "
            "والنسب المالية (P/E، ROE، EV/EBITDA، إلخ) والأسعار والنسب المئوية "
            "والاختصارات السوقية المتعارف عليها بالحروف اللاتينية.\n"
            + (f"قرار الاستثمار يجب كتابته بالعربية كالتالي: {_ar_verdict}\n" if _ar_verdict else "")
        )

    # ── Cap input length to prevent finish_reason=length for large reports ──────
    # Reports > 10 000 chars exceed DeepSeek-chat's practical output window at
    # max_tokens=6000.  Truncate at the last newline before the limit so we
    # never cut mid-sentence.  A 10 000-char report is already comprehensive.
    _MAX_POLISH_INPUT = 10_000
    if len(text) > _MAX_POLISH_INPUT:
        _cut = text.rfind("\n", 0, _MAX_POLISH_INPUT)
        if _cut < int(_MAX_POLISH_INPUT * 0.75):   # no newline in last 25% — hard cut
            _cut = _MAX_POLISH_INPUT
        log.warning("[polish] input truncated %d → %d chars for LLM", len(text), _cut)
        text = text[:_cut]

    # Prepend language instruction to user message for Arabic — double-reinforcement
    # ensures the LLM does not default to English regardless of source text language.
    _user_text = text
    if language == "ar":
        _user_text = (
            "اكتب التقرير النهائي بالكامل باللغة العربية. "
            "لا تستخدم الإنجليزية في العناوين أو الشرح. "
            "الاستثناءات الوحيدة: رموز الأسهم، أسماء الشركات، النسب المالية، والعملات.\n\n"
            + text
        )

    try:
        _polished = _llm_call(_user_text, _system, max_tokens=6000, timeout=90)
    except RuntimeError as _rte:
        # finish_reason=length — output was cut mid-sentence
        _fallback_reason = f"llm_truncated: {_rte}"
        _polished = ""
    except Exception as _exc:
        _fallback_reason = f"llm_error: {_exc}"
        _polished = ""

    _latency_ms = int((time.monotonic() - _t0) * 1000)

    if _polished:
        # Strip any meta-commentary the LLM may have prepended/appended
        _polished = _strip_meta_phrases(_polished)
        # Fix double-wrapped and plain-ticker autolinks in Signal Snapshot
        _polished = _strip_ticker_autolinks(_polished)
        # Phrase repairs — the LLM occasionally rewrites the "avoid chasing"
        # template into "Sell chasing", which reads as a hidden verdict.
        # Restore the canonical wording so the No-Action sentence stays
        # action-neutral.
        _polished = re.sub(
            r"\b[Ss]ell\s+chasing\b",
            "avoid chasing",
            _polished,
        )
        # Likewise: occasionally "Sell adding" / "Sell entering" sneak in
        # from the same paraphrase pattern.
        _polished = re.sub(
            r"\b[Ss]ell\s+(adding|entering|extending)\b",
            r"avoid \1",
            _polished,
        )
        # Remove invented beta values when source data marks beta as unreliable
        if _beta_unreliable:
            _polished = _strip_invented_beta(_polished, language=language)
        _ok, _msg = _polish_guard(text, _polished, ticker=_ticker, official_verdict=_official_verdict, language=language)
        if _ok:
            log.info(
                "[editorial] editorial_mode=polish_condensed ticker=%s latency_ms=%d "
                "polished_len=%d fallback=false",
                _ticker or "?", _latency_ms, len(_polished),
            )
            return _polished, False, ""
        else:
            _fallback_reason = f"guard_failed: {_msg}"

    log.warning(
        "[editorial] editorial_mode=polish_condensed ticker=%s latency_ms=%d "
        "fallback=true reason=%s",
        _ticker or "?", _latency_ms, _fallback_reason,
    )
    return text, True, _fallback_reason


# ── Internal helpers ────────────────────────────────────────────────────────────

def _rule_based(text: str, ticker: str = "") -> str:
    for pattern, replacement in _WEAK_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)
    text = _fix_markdown_bleed(text)
    text = _normalize_verdict_vocab(text)
    text = _strip_legacy_evidence_labels(text)
    text = _flatten_over_explanation(text)
    # ── Controlled variability: deterministic phrase rotation per ticker ──
    # Same ticker → same phrasing every time; different tickers → varied output.
    # Prevents "template smell" without losing editorial governance.
    if ticker:
        try:
            from core.sentence_variation import apply_controlled_variability
            text = apply_controlled_variability(text, ticker)
        except Exception as _sv_err:
            log.debug("[editorial.variation] skipped: %s", _sv_err)
    text = _dedup_paragraphs(text)
    return text.strip()


def _fix_markdown_bleed(text: str) -> str:
    """
    Fix prompt-formatting leakage where headers split across lines.
    e.g. "**1.\nExecutive Summary**" → "**1. Executive Summary**"
    Also collapses orphan emphasis markers across line breaks.
    """
    # Pattern 1: **1.\nHeader Name** → **1. Header Name**
    text = re.sub(r"\*\*(\d+\.)\s*\n+\s*([A-Za-z][^\*\n]+)\*\*", r"**\1 \2**", text)
    # Pattern 2: orphan ** at end of line followed by blank → header continuation
    text = re.sub(r"\*\*\s*\n\s*([A-Z][a-z]+(?:\s+[A-Za-z]+){0,4})\*\*", r"**\1**", text)
    # Pattern 3: stripped of "## Section\n###" double-headers
    text = re.sub(r"(#{1,3})\s*\n+\s*(#{1,3})\s+", r"\2 ", text)
    return text


# ── Verdict canonicalization (across the entire report body) ─────────────────
# IMPORTANT: order matters — multi-word phrases must run BEFORE single-word
# substitutions so they don't get half-replaced.
_VERDICT_NORM = [
    # Multi-word combos first
    (r"\bTactical\s+Underweight\b",                 "Reduce"),
    (r"\bTactical\s+Overweight\b",                  "Buy"),
    (r"\bTactical\s+Reduce\b",                      "Reduce"),
    (r"\bTactical\s+Buy\b",                         "Buy"),
    (r"\bTactical\s+Hold\b",                        "Hold"),
    (r"\bHold\s+Steady\b",                          "Hold"),
    (r"\bMaintain\s+position\b",                    "Hold"),
    # Action phrasing
    (r"\bWAIT\s*/\s*NO\s+ACTION\b",                 "Wait"),
    (r"\bWATCHLIST\s*/\s*WAIT\s+FOR\s+ENTRY\b",     "Wait for Entry"),
    (r"\bREDUCE\s*/\s*RISK\s+CONTROL\b",            "Reduce Exposure"),
    (r"\bSCALE\s+IN\s+GRADUALLY\b",                 "Scale In"),
    # Single legacy verbose words
    (r"\bUnderweight\b",                            "Reduce"),
    (r"\bOverweight\b",                             "Buy"),
    (r"\bACCUMULATE\b",                             "Buy"),
    (r"\bAVOID\b",                                  "Sell"),
    # "Fundamental Verdict: X" — relabel as "Fundamental View" to avoid dual-verdict
    (r"\bFundamental\s+Verdict:\s*",                "Fundamental View: "),
    # ALL-CAPS canonical verdicts → Title case (institutional tone)
    # Word-boundary on both sides + standalone — won't break compounds like "BUY-AND-HOLD-PORTFOLIO"
    (r"(?<![A-Z-])\bBUY\b(?![A-Z-])",               "Buy"),
    (r"(?<![A-Z-])\bHOLD\b(?![A-Z-])",              "Hold"),
    (r"(?<![A-Z-])\bREDUCE\b(?![A-Z-])",            "Reduce"),
    (r"(?<![A-Z-])\bSELL\b(?![A-Z-])",              "Sell"),
]


def _normalize_verdict_vocab(text: str) -> str:
    """
    Unify all verdict mentions in the report to the 4-axis taxonomy.
    Catches: HOLD/REDUCE/UNDERWEIGHT/STEADY/ACCUMULATE/AVOID/WAIT-style phrases.

    Note: the ALL-CAPS → Title case patterns use lookarounds so they're
    case-SENSITIVE — applied without IGNORECASE. Other patterns are CI.
    """
    for pat, repl in _VERDICT_NORM:
        if any(part in pat for part in (r"(?<![A-Z-])", r"(?![A-Z-])")):
            # Case-sensitive: only ALL-CAPS standalone tokens get title-cased
            text = re.sub(pat, repl, text)
        else:
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


# ── Strip retail-style Confidence/Conviction lines ───────────────────────────
def _strip_legacy_evidence_labels(text: str) -> str:
    """
    Remove standalone "Confidence: Low" / "Conviction: Low" lines from any
    section of the long report — they bleed in from older prompt templates.
    """
    # Remove "Confidence: <Value>" anywhere in the report (line or inline)
    text = re.sub(r"\*{0,2}\s*Confidence:\s*\*{0,2}\s*(?:Low|Medium|High|Limited|Moderate|Strong)\s*\*{0,2}\s*[·\.|]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*{0,2}\s*Conviction:\s*\*{0,2}\s*(?:Low|Medium|High|Limited|Moderate|Strong)\s*\*{0,2}\s*[·\.|]?\s*", "", text, flags=re.IGNORECASE)
    # "Fundamental Conviction: Low" → "Fundamental Visibility: Limited"
    text = re.sub(r"\*{0,2}\s*Fundamental\s+Conviction:\s*Low\s*\*{0,2}", "Fundamental Visibility: Limited", text, flags=re.IGNORECASE)
    text = re.sub(r"\*{0,2}\s*Fundamental\s+Conviction:\s*Medium\s*\*{0,2}", "Fundamental Visibility: Partial", text, flags=re.IGNORECASE)
    text = re.sub(r"\*{0,2}\s*Fundamental\s+Conviction:\s*High\s*\*{0,2}", "Fundamental Visibility: Full", text, flags=re.IGNORECASE)
    # Clean up resulting double separators (spaces only — preserve newlines)
    text = re.sub(r" · +·", " ·", text)
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse runs of spaces/tabs ONLY
    text = re.sub(r" +(\n)", r"\1", text)     # strip trailing spaces before newlines
    return text


# ── Over-explanation flattener (kill retail-TA-YouTube prose) ─────────────────
_OVEREXPLAIN_PATTERNS = [
    # "If RSI drops below 30 AND price closes below SMA200 for 2 consecutive days, X fails"
    (r"If\s+RSI\s+(?:drops?|falls?)\s+below\s+\d+\s+AND\s+price\s+closes?\s+below\s+SMA[\d-]+\s+for\s+\d+\s+(?:consecutive\s+)?days?,?\s*(?:the\s+)?(?:current\s+)?[^\.]{5,80}\s+fails?\.?",
     "Sustained weakness below long-term trend support would invalidate the timing framework."),
    # "If price breaks above resistance X with volume Y% greater than 20-day average for 3 consecutive sessions"
    (r"If\s+price\s+breaks?\s+above\s+[\$\d.,]+\s+with\s+volume\s+[^\.]{5,100}\s+(?:for\s+\d+\s+(?:consecutive\s+)?sessions?,?\s*)?",
     "A confirmed breakout with volume expansion would "),
]


def _flatten_over_explanation(text: str) -> str:
    for pat, repl in _OVEREXPLAIN_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _dedup_paragraphs(text: str) -> str:
    """Remove consecutive duplicate paragraph blocks (120-char fingerprint)."""
    blocks = re.split(r'\n{2,}', text)
    seen: set = set()
    out: list = []
    for block in blocks:
        key = re.sub(r'\s+', ' ', block.strip().lower())[:120]
        if not key or key not in seen:
            out.append(block)
            if key:
                seen.add(key)
    return '\n\n'.join(out)


def _llm_call(text: str, system: str, max_tokens: int, timeout: int = 25) -> str:
    resp = requests.post(
        _DS_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # Detect hard truncation via finish_reason
    finish = data.get("choices", [{}])[0].get("finish_reason", "")
    if finish == "length":
        log.warning("[editorial] finish_reason=length — output truncated by token limit")
        raise RuntimeError("LLM output truncated (finish_reason=length)")
    return data["choices"][0]["message"]["content"].strip()
