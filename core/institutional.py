"""
EISAX Institutional Intelligence Module

Provides decision-grade response formatting, intent validation,
confidence calibration, and executive-level output structuring.
"""

from __future__ import annotations
from typing import Any, Literal
import re
import config
from core.llm import get_client

# ============================================================
# OUTPUT MODES
# ============================================================
OUTPUT_MODES = {
    "explain": "Detailed educational breakdown with context and rationale",
    "framework": "Structured conceptual model with components and relationships",
    "executive": "Concise decision-ready format with key conclusions first",
    "slides": "Presentation-ready bullet structure",
    "memo": "Formal written memo format for documentation",
}

DEFAULT_OUTPUT_MODE = "executive"


def detect_output_mode(text: str) -> str:
    """
    Detect explicit output mode from user request.
    Returns mode name or 'executive' (default).
    """
    low = text.lower()
    
    # Check for explicit mode requests
    if any(w in low for w in ["explain to me", "explain this", "help me understand", "walk me through"]):
        return "explain"
    if any(w in low for w in ["framework", "structure this", "conceptual model", "organize this"]):
        return "framework"
    if any(w in low for w in ["executive summary", "brief", "tldr", "bottom line", "quick summary"]):
        return "executive"
    if any(w in low for w in ["slides", "presentation format", "bullet points for presentation"]):
        return "slides"
    if any(w in low for w in ["memo", "formal memo", "write a memo", "memo format"]):
        return "memo"
    
    return DEFAULT_OUTPUT_MODE


# ============================================================
# INTENT GUARD
# ============================================================
INTENT_CATEGORIES = {
    "general": [
        "write", "draft", "edit", "proofread", "summarize text", "translate",
        "code", "debug", "programming", "python", "javascript", "sql",
        "math", "calculate", "equation", "explain concept",
        "research", "what is", "define", "history of",
    ],
    "investment": [
        "portfolio", "stocks", "bonds", "etf", "allocation", "diversif",
        "invest", "return", "volatility", "risk", "sharpe", "var ",
        "optimize", "rebalance", "asset", "equity", "fixed income",
        "market", "sector", "dividend", "yield", "performance",
        "crypto", "bitcoin", "gold", "commodit",
        "weights", "drawdown", "backtest", "metrics", "mutual fund", "hedge fund",
        "financial", "industry", "valuation", "pe ratio", "cagr", "value at risk",
    ],
    "research": [
        "analyze", "research", "compare", "evaluate", "assess",
        "study", "investigate", "deep dive", "comprehensive analysis",
    ],
}


def classify_intent_category(text: str) -> Literal["general", "investment", "research"]:
    """
    Classify user intent into one of three categories.
    Used by Intent Guard to route or redirect.
    """
    low = text.lower()
    
    # Count matches for each category
    scores = {cat: 0 for cat in INTENT_CATEGORIES}
    
    for category, keywords in INTENT_CATEGORIES.items():
        for kw in keywords:
            if kw in low:
                scores[category] += 1
    
    # Determine dominant category
    if scores["investment"] >= 2 or (scores["investment"] > scores["general"]):
        return "investment"
    if scores["research"] >= 2:
        return "research"
    return "general"


def intent_guard_check(user_mode: str, intent_category: str) -> dict | None:
    """
    Check if request is outside current agent's scope.
    Returns redirect suggestion if needed, None if OK to proceed.
    
    Args:
        user_mode: Current agent mode ('assistant' or 'investment')
        intent_category: Detected intent category
    """
    # General Assistant handling investment requests
    if user_mode == "assistant" and intent_category == "investment":
        return {
            "redirect": True,
            "target": "investment",
            "message": (
                "This request involves investment analysis, which would be better handled "
                "by the Investment Assistant.\n\n"
                "Would you like to switch to the Investment Assistant?"
            )
        }
    
    # Investment Assistant handling general tasks
    if user_mode == "investment" and intent_category == "general":
        return {
            "redirect": True,
            "target": "assistant",
            "message": (
                "This appears to be a general task outside investment analysis.\n\n"
                "Would you like to switch to the General Assistant?"
            )
        }
    

    return None


def needs_analysis_confirmation(text: str, has_confirmation: bool = False) -> bool:
    """
    Returns True if user is asking for institutional analysis that needs confirmation.
    """
    low = (text or "").lower()
    
    # These always need confirmation (heavy analysis)
    heavy_analysis_phrases = [
        "analyze", "analysis", "evaluate", "assess", "review", "study", "examine",
        "research", "investigate", "deep dive", "comprehensive", "detailed", "full"
    ]
    
    # These are direct commands that don't need confirmation
    direct_commands = [
        "optimize", "report", "metrics", "export", "irr", "var", "calculate",
        "build me", "create", "design", "make", "generate", "run", "execute"
    ]
    
    has_heavy = any(phrase in low for phrase in heavy_analysis_phrases)
    # Check if the message starts with a direct command
    has_direct = any(low.startswith(cmd) for cmd in direct_commands)
    
    # If it's a heavy analysis request and user hasn't confirmed yet
    if has_heavy and not has_direct and not has_confirmation:
        return True
    
    return False



# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================
CONFIDENCE_LEVELS = {
    "high": "High confidence",
    "medium": "Medium confidence (assumptions involved)",
    "exploratory": "Exploratory / directional",
}


def assess_confidence(
    has_real_data: bool = False,
    has_assumptions: bool = False,
    is_projection: bool = False,
    is_opinion: bool = False,
) -> str:
    """
    Determine appropriate confidence level for response.
    """
    if has_real_data and not has_assumptions and not is_projection:
        return "high"
    if is_projection or is_opinion:
        return "exploratory"
    return "medium"


def format_confidence_tag(level: str) -> str:
    """Format confidence level for display."""
    return f"*Confidence: {CONFIDENCE_LEVELS.get(level, level)}*"


# ============================================================
# EXECUTIVE SUMMARY GENERATOR
# ============================================================
def generate_executive_summary(content: str, max_bullets: int = 5) -> str:
    """
    Extract key conclusions from content for executive summary.
    Uses heuristics to identify conclusion-like statements.
    
    For LLM-powered summary, use generate_executive_summary_llm instead.
    """
    lines = content.split("\n")
    conclusions = []
    
    # Patterns that indicate conclusions
    conclusion_indicators = [
        r"^(?:therefore|thus|consequently|in conclusion|overall|key takeaway|bottom line)",
        r"(?:should|recommend|suggest|advise)(?:s|ed)?",
        r"(?:critical|essential|important|key|primary)",
        r"(?:opportunity|risk|challenge|advantage)",
        r"(?:\d+%|\d+x|\$\d+)",  # Contains numbers/percentages
    ]
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Score the line for conclusion-likelihood
        score = 0
        low_line = line.lower()
        
        for pattern in conclusion_indicators:
            if re.search(pattern, low_line):
                score += 1
        
        if score >= 1 and len(line) < 200:
            # Clean up bullet formatting
            clean = re.sub(r"^[-*•]\s*", "", line).strip()
            if clean and clean not in conclusions:
                conclusions.append(clean)
        
        if len(conclusions) >= max_bullets:
            break
    
    if not conclusions:
        # Fallback: take first substantive sentences
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 20:
                clean = re.sub(r"^[-*•]\s*", "", line).strip()
                if clean:
                    conclusions.append(clean[:150])
                    if len(conclusions) >= 3:
                        break
    
    return conclusions


def format_executive_summary(bullets: list[str]) -> str:
    """Format executive summary bullets."""
    if not bullets:
        return ""
    
    lines = ["## Executive Summary", ""]
    for bullet in bullets[:5]:
        lines.append(f"- {bullet}")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# ASSUMPTIONS & CONSTRAINTS BOX (Investment Only)
# ============================================================
def format_assumptions_box(
    assumptions: list[str] | None = None,
    data_limitations: list[str] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """
    Format the Assumptions & Constraints box for investment analysis.
    """
    sections = []
    
    if assumptions:
        sections.append("**Key Assumptions:**")
        for a in assumptions[:3]:
            sections.append(f"- {a}")
    
    if data_limitations:
        sections.append("\n**Data Limitations:**")
        for d in data_limitations[:3]:
            sections.append(f"- {d}")
    
    if constraints:
        sections.append("\n**Structural Constraints:**")
        for c in constraints[:3]:
            sections.append(f"- {c}")
    
    if not sections:
        return ""
    
    return "---\n### Assumptions & Constraints\n" + "\n".join(sections) + "\n---"


# ============================================================
# DECISION LENS (Investment Only)
# ============================================================
DECISION_SIGNALS = {
    "proceed": "PROCEED - Conditions support action",
    "hold": "HOLD - Maintain current position",
    "reevaluate": "RE-EVALUATE - Additional analysis recommended",
    "more_data": "REQUIRES MORE DATA - Insufficient information for signal",
}


def format_decision_lens(signal: str, rationale: str = "") -> str:
    """
    Format the Decision Lens section.
    """
    signal_text = DECISION_SIGNALS.get(signal.lower(), signal)
    
    result = f"\n---\n### Decision Lens\n**{signal_text}**"
    if rationale:
        result += f"\n_{rationale}_"
    result += "\n\n*This is a directional signal, not investment advice.*\n---"
    
    return result


# ============================================================
# EISAX INSIGHT SIGNATURE
# ============================================================
def format_eisax_insight(insight: str) -> str:
    """
    Format the EISAX Insight signature.
    Should be a sharp, CIO-level observation.
    """
    if not insight:
        return ""
    
    # Clean up the insight
    insight = insight.strip()
    if not insight.endswith("."):
        insight += "."
    
    return f"\n---\n**EISAX INSIGHT:** {insight}\n"


# ============================================================
# SCENARIO THINKING
# ============================================================
def format_scenario_analysis(
    base_case: list[str],
    upside_case: list[str] | None = None,
    downside_case: list[str] | None = None,
    base_prob: float | None = None,
    upside_prob: float | None = None,
    downside_prob: float | None = None,
    base_price: str | None = None,
    upside_price: str | None = None,
    downside_price: str | None = None,
    base_return: float | None = None,
    upside_return: float | None = None,
    downside_return: float | None = None,
) -> str:
    """
    Format scenario analysis section.
    RULE: All scenarios MUST include probability + expected price.
    Total probabilities must sum to 100%.
    Scenarios without a price are classified as Tail Risk Overlay (not core scenarios).
    Adds Expected Value = Σ(probability × return).
    """
    # ── Normalize probabilities to sum to 100% ────────────────────────────
    probs = [base_prob, upside_prob, downside_prob]
    cases = [base_case, upside_case, downside_case]
    prices = [base_price, upside_price, downside_price]
    returns = [base_return, upside_return, downside_return]
    names = ["Base Case", "Upside Case", "Downside Case"]

    # Filter to only provided cases
    active = [(n, c, p, pr, r) for n, c, p, pr, r in zip(names, cases, probs, prices, returns) if c]

    if not active:
        return ""

    # Default equal-weight probabilities if not provided
    n_cases = len(active)
    default_prob = round(100 / n_cases, 1)
    total_provided = sum(p for _, _, p, _, _ in active if p is not None)

    normalized = []
    for name, case, prob, price, ret in active:
        if prob is None:
            prob = default_prob if total_provided == 0 else round((100 - total_provided) / n_cases, 1)
        normalized.append((name, case, prob, price, ret))

    # Adjust to exactly 100%
    total = sum(p for _, _, p, _, _ in normalized)
    if total != 100 and normalized:
        diff = 100 - total
        last = normalized[-1]
        normalized[-1] = (last[0], last[1], round(last[2] + diff, 1), last[3], last[4])

    sections = ["### Scenario Analysis", ""]
    tail_risks = []

    for name, case, prob, price, ret in normalized:
        if price is None:
            # No price → classify as Tail Risk Overlay
            tail_risks.append((name, case, prob))
            continue

        sections.append(f"**{name} ({prob:.0f}% probability):**")
        if price:
            sections.append(f"- *Expected Price: {price}*")
        for bullet in case[:2]:
            sections.append(f"- {bullet}")
        sections.append("")

    # Expected Value calculation (core scenarios only — exclude Tail Risk Overlay)
    ev_parts = [(r, p) for _, _, p, pr, r in normalized if r is not None and p is not None and pr is not None]
    if ev_parts:
        ev = sum(r * p / 100 for r, p in ev_parts)
        conf_band = "high" if abs(ev) > 0.15 else ("medium" if abs(ev) > 0.05 else "low")
        sections.append(f"**Expected Value:** {ev:+.1f}% (Confidence Band: {conf_band})")
        sections.append("")

    # Tail Risk Overlay (scenarios without explicit price targets)
    if tail_risks:
        sections.append("**Tail Risk Overlay** *(not included in EV calculation)*")
        for name, case, prob in tail_risks:
            sections.append(f"- *{name} ({prob:.0f}%):* {'; '.join(case[:1])}")
        sections.append("")

    return "\n".join(sections)


def detect_market_context_contradiction(
    trend: str,
    breadth: str | None = None,
    macro_signal: str | None = None,
) -> str | None:
    """
    Detect and explain contradictions between trend direction and market context.
    Returns an explanation string if a contradiction is found, None otherwise.

    Example contradictions:
    - Bearish trend + strong market breadth → "Short-term rebound within broader downtrend"
    - Bullish trend + deteriorating breadth → "Narrow leadership — fragile rally"
    """
    if not trend:
        return None

    trend_lower = trend.lower()
    breadth_lower = (breadth or "").lower()
    macro_lower  = (macro_signal or "").lower()

    is_bearish = "bear" in trend_lower or "down" in trend_lower
    is_bullish = "bull" in trend_lower or "up" in trend_lower

    strong_breadth  = any(w in breadth_lower for w in ("strong", "broad", "expanding", "positive"))
    weak_breadth    = any(w in breadth_lower for w in ("weak", "narrow", "deteriorating", "negative"))
    risk_on         = any(w in macro_lower   for w in ("risk-on", "expansion", "rate cut", "tailwind"))
    risk_off        = any(w in macro_lower   for w in ("risk-off", "recession", "rate hike", "headwind"))

    contradictions = []

    if is_bearish and strong_breadth:
        contradictions.append(
            "**Context Note:** Bearish price trend conflicts with strong market breadth. "
            "This suggests a short-term rebound within a broader downtrend — "
            "not a confirmed reversal. Wait for trend structure to improve before adding exposure."
        )
    if is_bearish and risk_on:
        contradictions.append(
            "**Context Note:** Risk-on macro environment conflicts with bearish price trend. "
            "Possible sector-specific headwind overriding broader market tailwind."
        )
    if is_bullish and weak_breadth:
        contradictions.append(
            "**Context Note:** Bullish price trend supported by narrow market leadership. "
            "Fragile rally — concentration risk elevated. Monitor breadth for deterioration."
        )
    if is_bullish and risk_off:
        contradictions.append(
            "**Context Note:** Risk-off macro signal conflicts with bullish price action. "
            "Defensive positioning may limit upside despite technical strength."
        )

    if not contradictions:
        return None

    return "\n\n" + "\n\n".join(contradictions)


# ============================================================
# RESPONSE FORMATTER
# ============================================================
def format_institutional_response(
    content: str,
    mode: str = "executive",
    confidence: str = "medium",
    is_investment: bool = False,
    assumptions: list[str] | None = None,
    decision_signal: str | None = None,
    decision_rationale: str = "",
    eisax_insight: str | None = None,
    include_summary: bool = True,
    contradiction_note: str | None = None,
) -> str:
    """
    Master formatter for institutional-grade responses.
    
    Applies:
    - Executive summary (if content is substantial)
    - Confidence tag
    - Assumptions box (investment only)
    - Decision lens (investment only)
    - EISAX Insight signature
    """
    parts = []
    
    # 1. Executive Summary (for substantial content)
    if include_summary and len(content) > 500:
        summary_bullets = generate_executive_summary(content)
        if summary_bullets:
            parts.append(format_executive_summary(summary_bullets))
    
    # 2. Main Content
    parts.append(content)

    # 2b. Market Context Contradiction (if any)
    if contradiction_note:
        parts.append(contradiction_note)

    # 3. Confidence Tag
    parts.append(f"\n{format_confidence_tag(confidence)}")
    
    # 4. Assumptions Box (Investment Only)
    if is_investment and assumptions:
        parts.append(format_assumptions_box(assumptions=assumptions))
    
    # 5. Decision Lens (Investment Only)
    if is_investment and decision_signal:
        parts.append(format_decision_lens(decision_signal, decision_rationale))
    
    # 6. EISAX Insight Signature
    if eisax_insight:
        parts.append(format_eisax_insight(eisax_insight))
    
    return "\n".join(parts)


# ============================================================
# MODE-SPECIFIC PROMPTS
# ============================================================
def get_output_mode_instruction(mode: str) -> str:
    """
    Get LLM instruction for specific output mode.
    """
    instructions = {
        "explain": (
            "Provide a detailed educational explanation. "
            "Use clear language, provide context, explain the 'why' behind concepts. "
            "Include relevant examples."
        ),
        "framework": (
            "Present a structured conceptual framework. "
            "Identify key components, their relationships, and the overall architecture. "
            "Use headers to organize major sections."
        ),
        "executive": (
            "Provide a decision-ready executive response. "
            "Lead with conclusions, be concise, prioritize actionable information. "
            "Avoid lengthy explanations; state what matters and why."
        ),
        "slides": (
            "Structure the response as presentation-ready bullet points. "
            "Use clear headers for each slide topic. "
            "Each point should be concise (under 15 words). "
            "Aim for 3-5 bullets per section."
        ),
        "memo": (
            "Format as a formal business memo. "
            "Include: Subject, Summary, Background, Analysis, Recommendations. "
            "Maintain professional, objective tone throughout."
        ),
    }
    
    return instructions.get(mode, instructions["executive"])


# ============================================================
# MEMORY WITH PERMISSION
# ============================================================
def format_memory_consent_request(preference_type: str) -> str:
    """
    Generate a consent request for storing user preferences.
    """
    return (
        f"\nI noticed a pattern in your {preference_type} preferences. "
        "Would you like me to remember this for future sessions?"
    )


def should_request_memory_consent(
    new_preference: str,
    existing_preferences: list[str],
) -> bool:
    """
    Determine if we should ask user for memory consent.
    Only ask if this is a new, distinct preference.
    """
    if not new_preference:
        return False
    
    # Don't re-ask for preferences we already know
    low_pref = new_preference.lower()
    for existing in existing_preferences:
        if low_pref in existing.lower() or existing.lower() in low_pref:
            return False
    

    return True


# ============================================================
# CROSS-AGENT COLLABORATION
# ============================================================
def format_consultation(
    primary_agent: str,
    consulting_agent: str,
    consultation_content: str,
) -> str:
    """
    Format a response that involves consultation between agents.
    The goal is a unified response, not exposing the internal dialogue.
    """
    # Simply incorporate the content without meta-commentary
    # The user should receive a seamless answer
    
    return consultation_content

