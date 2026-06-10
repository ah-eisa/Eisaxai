"""
core/services/scorecard_engine.py
───────────────────────────────────
Entry / stop / target positioning logic, extracted from _handle_analytics.

The full scorecard *display* is built by FinanceAgent._build_scorecard_md()
(which already lives in finance.py as a method).  This module handles only
the *positioning* computation — the price levels shown in the Positioning Guide.

Public API
──────────
    classify_adx(adx_value) -> tuple[str, str]
        Deterministic ADX bucket. Returns (short_label, description).
        Single source of truth — must be used everywhere ADX is described.

    validate_positioning(ep, sp, rp) -> tuple[float|None, float|None, bool, str]
        Validates entry/stop/target levels for a long trade.
        Returns (ep_fixed, sp_fixed, was_fixed, fix_note).

    compute_positioning(
        real_price, sma200, h52, l52,
        display_target, currency_sym, currency_lbl, fv_label
    ) -> dict
        Returns {pre_entry, pre_stop, pre_target, ep, sp}.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── ADX Classification — SINGLE SOURCE OF TRUTH ──────────────────────────────

def classify_adx(adx_value: float) -> tuple[str, str]:
    """
    Deterministic ADX strength classification.

    Buckets
    ───────
    < 20     → Weak      "Weak/absent trend (ADX < 20) — range-bound price action"
    20 – <25 → Emerging  "Emerging trend (ADX 20–25) — watch for confirmation"
    25 – <30 → Confirmed "Trend confirmed (ADX 25–30)"
    ≥ 30     → Strong    "Strong trend (ADX ≥ 30) — high directional conviction"

    Returns
    ───────
    (short_label: str, description: str)
    Use short_label in scorecard/table cells, description in prose sections.
    """
    v = float(adx_value or 0)
    if v >= 30:
        return "Strong", "Strong trend (ADX >= 30)"
    elif v >= 25:
        return "Confirmed", "Trend confirmed (ADX 25-30)"
    elif v >= 20:
        return "Emerging", "Emerging/borderline trend (ADX 20-25)"
    else:
        return "Weak", "Weak/absent trend (ADX < 20)"


# ── Positioning Validator ─────────────────────────────────────────────────────

def validate_positioning(
    ep: float | None,
    sp: float | None,
    rp: float | None,
) -> tuple[float | None, float | None, bool, str]:
    """
    Validate entry / stop levels for a long trade and auto-fix invalid states.

    Rules
    ─────
    1. stop must be BELOW entry  (sp < ep)
    2. entry must be BELOW or AT current price  (ep <= rp)

    If rule 1 violated  → recalculate stop as ep * 0.93
    If rule 2 violated  → recalculate entry as rp * 0.96, stop as rp * 0.91

    Returns
    ───────
    (ep_out, sp_out, was_fixed: bool, fix_note: str)
    fix_note is empty when no fix needed.
    """
    if ep is None or sp is None:
        return ep, sp, False, ""

    fixed = False
    notes: list[str] = []

    # Rule 2: entry must not exceed current price
    if rp and ep > rp * 1.001:
        old_ep, old_sp = ep, sp
        ep = rp * 0.96
        sp = rp * 0.91
        fixed = True
        notes.append(
            f"entry was above price ({old_ep:.2f} > {rp:.2f}); "
            f"reset to {ep:.2f}/{sp:.2f}"
        )

    # Rule 1: stop must be below entry
    if sp >= ep:
        old_sp = sp
        sp = ep * 0.93
        fixed = True
        notes.append(
            f"stop ({old_sp:.2f}) was >= entry ({ep:.2f}); "
            f"corrected to {sp:.2f} (-7% from entry)"
        )

    if fixed:
        logger.warning("[Positioning] Auto-fix applied: %s", "; ".join(notes))

    return ep, sp, fixed, "; ".join(notes)


# ── compute_positioning ────────────────────────────────────────────────────────

def compute_positioning(
    real_price:    float | None,
    sma200:        float,
    h52:           float,
    l52:           float,
    display_target: float | None,
    currency_sym:  str   = "$",
    currency_lbl:  str   = "USD",
    fv_label:      str   = "Analyst Target",
) -> dict:
    """
    Compute entry / stop / target price levels.

    Strategy
    ────────
    Entry:
      1. Fibonacci: nearest level BELOW current price that is 1-15% pullback
      2. Fallback: SMA200 * 1.02
      3. If entry >= 98% of current price -> force -4% from current

    Stop:
      - SMA200 * 0.95 (5% below long-term support)
      - If price already deeply below SMA200: stop = price * 0.91

    Validation:
      - validate_positioning() is called before formatting to guarantee
        stop < entry <= price for every long setup rendered.

    Target:
      - Analyst consensus target (or EisaX Fair Value estimate)
      - N/A if unavailable

    Returns
    ───────
    dict with keys:
      pre_entry   : formatted string (e.g. "$142.50 *(Limit Order)*")
      pre_stop    : formatted string (e.g. "$129.75")
      pre_target  : formatted string (e.g. "$180.00 (+20.5%) — *Analyst Target*")
      ep          : raw float entry price (or None)
      sp          : raw float stop price  (or None)
    """
    _rp = real_price or 0.0
    _is_local = currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")

    # ── Fibonacci entry ──────────────────────────────────────────────────────
    ep: float | None = None
    if h52 and l52 and _rp:
        _fib_levels = [
            l52 + (h52 - l52) * ratio
            for ratio in (0.236, 0.382, 0.500, 0.618)
        ]
        # Nearest level that is BELOW current price (meaningful pullback)
        _below = [f for f in _fib_levels if f < _rp * 0.995]
        if _below:
            _fib_ep = max(_below)
            if _rp and 0.85 <= (_fib_ep / _rp) <= 0.99:
                ep = _fib_ep

    # ── SMA200 fallback entry ────────────────────────────────────────────────
    if ep is None and sma200:
        ep = sma200 * 1.02

    # ── Stop ─────────────────────────────────────────────────────────────────
    sp: float | None = sma200 * 0.95 if sma200 else None

    # ── Force entry below current if too close ────────────────────────────────
    if _rp and ep and ep >= _rp * 0.98:
        ep = _rp * 0.96
        sp = _rp * 0.91

    # ── If price already deeply below SMA200: stop from current, not SMA200 ─
    if _rp and sma200 and _rp < sma200 * 0.90:
        sp = _rp * 0.91

    # ── VALIDATE: stop must be below entry for a long trade ───────────────────
    ep, sp, _fixed, _fix_note = validate_positioning(ep, sp, _rp)

    # ── Format helpers ────────────────────────────────────────────────────────
    def _fmt(price: float | None) -> str:
        if price is None:
            return "N/A"
        return f"{price:,.2f} {currency_sym}" if _is_local else f"${price:.2f}"

    _entry_is_limit = bool(ep and _rp and ep < _rp * 0.985)
    _limit_note = (
        " *(Limit Order — wait for pullback)*"
        if _entry_is_limit else ""
    )

    pre_entry = (_fmt(ep) + _limit_note) if ep else "N/A"
    pre_stop  = _fmt(sp)

    # ── Target ───────────────────────────────────────────────────────────────
    if display_target and _rp:
        _upside = ((display_target / _rp) - 1) * 100
        _tgt_str = _fmt(display_target)
        pre_target = f"{_tgt_str} ({_upside:+.1f}%) — *{fv_label}*"
    else:
        pre_target = "N/A"

    logger.debug(
        "[Positioning] entry=%s stop=%s target=%s fixed=%s",
        pre_entry, pre_stop, pre_target, _fixed,
    )

    return {
        "pre_entry":  pre_entry,
        "pre_stop":   pre_stop,
        "pre_target": pre_target,
        "ep":         ep,
        "sp":         sp,
    }
