"""scorecard_verdict.py -- entry quality, tech score, verdict, decision type."""
import re
from typing import Optional

from core.scorecard_parser import sanitize_field, _NEAR_ZERO_BETA_ALLOWED

def compute_entry_quality(data: dict) -> tuple[int, str, str]:
    '''
    Entry Quality Score: rates the TIMING of an entry, not the stock quality.
    Returns: (score 0-100, label, explanation)
    Score >= 70 = Good Entry Timing
    Score 50-69 = Acceptable
    Score  < 50 = Poor Timing / Wait
    '''
    score = 50  # neutral baseline

    # 1. RSI: best entry when not overbought/oversold extremes (25 pts)
    rsi = float(data.get('rsi', 50) or 50)
    if 35 <= rsi <= 55:    score += 25  # ideal zone
    elif 30 <= rsi < 35:   score += 18  # near oversold — risky but opportunity
    elif 55 < rsi <= 65:   score += 12  # slightly elevated but ok
    elif rsi < 30:         score += 5   # oversold — mean reversion play only
    elif rsi > 70:         score -= 20  # overbought — avoid chasing

    # 2. ADX: strong trend = better entry reliability (20 pts)
    adx = float(data.get('adx', 20) or 20)
    if adx >= 30:   score += 20
    elif adx >= 25: score += 14
    elif adx >= 20: score += 8
    else:           score -= 10  # no trend = choppy = bad entry timing

    # 3. Price vs SMA200: entry near support is better (20 pts)
    price = float(data.get('price', 0) or 0)
    sma200 = float(data.get('sma200', 0) or 0)
    if price and sma200:
        dist_pct = (price - sma200) / sma200 * 100
        if -5 <= dist_pct <= 10:    score += 20  # near SMA200 support
        elif 10 < dist_pct <= 20:   score += 10  # above but not overextended
        elif dist_pct > 20:         score -= 10  # overextended above SMA200
        elif dist_pct < -10:        score += 5   # deeply below — mean reversion only

    # 4. Fear & Greed: contrarian entry better in fear zones (20 pts)
    fg = int(data.get('fear_greed', 50) or 50)
    if fg <= 25:              score += 20  # extreme fear = best entry
    elif fg <= 45:            score += 14  # fear zone = good
    elif fg <= 55:            score += 8   # neutral
    elif fg <= 70:            score -= 5   # greed = elevated risk
    else:                     score -= 15  # extreme greed = dangerous entry

    # 5. Volume: above-average volume on up moves = conviction (15 pts)
    vol = float(data.get('volume', 0) or 0)
    avg_vol = float(data.get('avg_volume', 0) or 0)
    trend = str(data.get('trend', '') or '')
    if avg_vol and vol:
        ratio = vol / avg_vol
        if ratio >= 1.5 and trend == 'Bullish':  score += 15  # strong volume + up trend
        elif ratio >= 1.2:                        score += 8
        elif ratio < 0.7:                         score -= 8   # low volume = weak conviction

    final_eq = max(0, min(100, score))

    if final_eq >= 70:
        label = 'Good Timing ✅'
        note = 'Entry conditions are favorable — risk/reward is well-positioned.'
    elif final_eq >= 55:
        label = 'Acceptable ⚠️'
        note = 'Entry is acceptable but not ideal — consider scaling in gradually.'
    elif final_eq >= 40:
        label = 'Caution 🟡'
        note = 'Timing is suboptimal — wait for a better setup or use a tight stop.'
    else:
        label = 'Poor Timing 🔴'
        note = 'Entry conditions are unfavorable — high risk of immediate drawdown.'

    return final_eq, label, note


def compute_tech_score(data: dict) -> int:
    """
    Converts raw technical signals into a 0-100 score.
    Used to blend with fundamental score in get_verdict().

    Components:
      Trend  (SMA200): ±25 pts  — highest weight, most reliable
      Momentum (MACD): ±15 pts  — confirmation signal
      Strength (ADX) :  0-15 pts — amplifier when trend is strong
      RSI            : ±10 pts  — contrarian signal at extremes
    Baseline: 50 (neutral)
    """
    trend    = str(data.get('trend', '') or '').strip()
    momentum = str(data.get('momentum', '') or '').strip()
    adx      = float(data.get('adx', 20) or 20)
    rsi      = float(data.get('rsi', 50) or 50)

    ts = 50  # neutral baseline

    # Trend: ±25
    if trend == 'Bullish':   ts += 25
    elif trend == 'Bearish': ts -= 25

    # Momentum: ±15
    if momentum == 'Bullish':   ts += 15
    elif momentum == 'Bearish': ts -= 15

    # ADX (trend strength): 0-15
    if adx >= 35:   ts += 15
    elif adx >= 25: ts += 10
    elif adx >= 20: ts += 5
    else:           ts -= 5   # weak trend reduces reliability

    # RSI contrarian: ±10 at extremes
    if rsi <= 30:               ts += 10  # oversold = potential reversal
    elif rsi >= 70:             ts -= 10  # overbought = caution
    elif 40 <= rsi <= 60:       ts += 3   # neutral zone = slight positive

    return max(0, min(100, int(ts)))


def get_verdict(score: int, data: dict) -> tuple[str, str, str]:
    """
    Dynamic Weighted Decision Engine v2.

    Blends fundamental score + technical score using asset-type weights:
      Equity:    Fundamentals 60% + Technical 40%
      Crypto:    Fundamentals 30% + Technical 70%
      ETF/Cmdty: Fundamentals 40% + Technical 60%

    Replaces binary LLM conflict resolution with soft gradient override (±1 tier max).
    Returns: (verdict_label, emoji, conviction)
    """
    beta         = float(data.get('beta') or 1.0)
    beta_penalty = beta > 2.0
    estimate_only = bool(data.get('target_is_estimate', False))
    is_crypto    = bool(data.get('is_crypto', False))
    is_etf       = bool(data.get('is_etf', False))

    # ── 1. Technical Score ────────────────────────────────────────────────────
    tech_score = compute_tech_score(data)

    # ── 2. Asset-Type Weights ─────────────────────────────────────────────────
    if is_crypto:
        w_fund, w_tech = 0.30, 0.70
    elif is_etf:
        w_fund, w_tech = 0.40, 0.60
    else:
        w_fund, w_tech = 0.60, 0.40

    blended = round(score * w_fund + tech_score * w_tech)

    # ── 3. Blended Score → Verdict Tiers ─────────────────────────────────────
    if blended >= 80:
        if beta_penalty:
            sv, se, sc = "BUY (High Risk)", "🟡", "Medium"
        elif estimate_only:
            sv, se, sc = "BUY", "🟢", "Medium"
        else:
            sv, se, sc = "STRONG BUY", "🟢", "High"
    elif blended >= 75:
        # estimate_only no longer suppresses BUY at this score level (Rule 8A territory)
        sv, se, sc = "BUY", "🟢", ("Medium" if beta_penalty else "Medium-High")
    elif blended >= 68:
        if estimate_only:
            sv, se, sc = "HOLD", "🟡", "Low"
        else:
            sv, se, sc = "BUY", "🟢", ("Medium" if beta_penalty else "Medium-High")
    elif blended >= 52:
        sv, se, sc = "HOLD", "🟡", "Low"
    elif blended >= 38:
        sv, se, sc = "REDUCE", "🟠", "Medium"
    else:
        sv, se, sc = "SELL", "🔴", "High"

    # ── 4. Upside Guard: BUY with no room = HOLD ─────────────────────────────
    if not data.get('target_is_estimate') and not data.get('target_is_sma'):
        _tgt = float(data.get('target', 0) or 0)
        _prc = float(data.get('price', 0) or 0)
        if _tgt and _prc:
            _upside_pct  = (_tgt - _prc) / _prc * 100
            _div_yield   = float(data.get('dividend_yield') or 0) * 100
            _total_return = _upside_pct + _div_yield
            if sv in ('BUY', 'STRONG BUY', 'BUY (High Risk)'):
                if _upside_pct < 10 and _total_return < 12:
                    sv, se, sc = "HOLD", "🟡", "Low"
                elif _upside_pct < 10 and _total_return >= 12:
                    sv, se, sc = "HOLD", "🟡", "Medium"

    # ── 5. Soft Technical Override (max ±1 tier, not binary flip) ────────────
    # Replaces old binary "conflict → HOLD" logic.
    # Strong technical divergence nudges verdict by ONE tier only.
    _TIERS = ["SELL", "REDUCE", "HOLD", "BUY", "STRONG BUY"]
    _EMOJIS = {"SELL": "🔴", "REDUCE": "🟠", "HOLD": "🟡", "BUY": "🟢", "STRONG BUY": "🟢", "BUY (High Risk)": "🟡"}

    sv_base = "BUY" if sv == "BUY (High Risk)" else sv
    if sv_base in _TIERS:
        idx = _TIERS.index(sv_base)
        # Strong technical bearish (tech_score < 35) AND fundamental verdict is BUY+
        if tech_score < 35 and sv in ('BUY', 'STRONG BUY', 'BUY (High Risk)'):
            idx = max(0, idx - 1)  # nudge down one tier
            sv  = _TIERS[idx]
            se  = _EMOJIS.get(sv, "🟡")
            sc  = "Low"
        # Strong technical bullish (tech_score > 70) AND fundamental verdict is REDUCE/SELL
        elif tech_score > 70 and sv in ('REDUCE', 'SELL'):
            idx = min(len(_TIERS) - 1, idx + 1)  # nudge up one tier
            sv  = _TIERS[idx]
            se  = _EMOJIS.get(sv, "🟡")
            sc  = "Low"

    # ── RULE 8A — FORCED FUNDAMENTAL BUY (Hard Rule, No Exceptions) ─────────
    # When Score ≥ 75 AND Upside ≥ 20%, Fundamental Verdict MUST be BUY.
    # Weak technicals (ADX, RSI) affect Entry Timing only — not this verdict.
    _tgt_r8 = float(data.get('target', 0) or 0)
    _prc_r8 = float(data.get('price', 0) or 0)
    if _tgt_r8 and _prc_r8:
        _upside_r8 = (_tgt_r8 - _prc_r8) / _prc_r8 * 100
        if score >= 75 and _upside_r8 >= 20 and sv not in ('BUY', 'STRONG BUY', 'BUY (High Risk)'):
            sv = 'BUY'
            se = '🟢'
            sc = 'High' if score >= 80 else 'Medium'
            data['_rule8a_override'] = True

    # ── Entry Timing (independent of Fundamental Verdict) ────────────────────
    # Reflects technical readiness for entry — NEVER changes Fundamental Verdict.
    _adx_t = float(data.get('adx', 0) or 0)
    _rsi_t = float(data.get('rsi', 50) or 50)
    if sv in ('BUY', 'STRONG BUY', 'BUY (High Risk)'):
        if _rsi_t > 70:
            data['entry_timing'] = 'WAIT — RSI overbought, await pullback'
        elif _adx_t < 20:
            data['entry_timing'] = 'WAIT — trend not confirmed (ADX < 20)'
        elif _adx_t < 25:
            data['entry_timing'] = 'ADD ON DIP — await ADX > 25'
        else:
            data['entry_timing'] = 'BUY NOW — trend confirmed'
    elif sv in ('REDUCE', 'SELL', 'AVOID'):
        data['entry_timing'] = 'REDUCE INTO STRENGTH'
    else:
        data['entry_timing'] = 'WAIT'

    # ── 6. Store blended score + tech score for display ──────────────────────
    data['blended_score']  = blended
    data['tech_score']     = tech_score

    return sv, se, sc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3.5: DECISION TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_decision_type(verdict: str, data: dict) -> dict:
    """
    Classify the decision type and return mandatory context fields.

    Rules:
      BUY + bearish trend  → contrarian_early
      BUY + neutral trend  → early_reversal
      BUY + bullish trend  → trend_confirmed
      HOLD + weak tech + strong fund → wait_for_confirmation
      SELL/REDUCE + breakdown → trend_failure
    """
    trend   = str(data.get('trend', '') or '').strip()
    bearish = data.get('is_bearish', False) or trend == 'Bearish'
    bullish = not bearish and (data.get('bullish_count', 0) >= 2 or trend == 'Bullish')
    neutral = not bearish and not bullish

    v = verdict.upper()
    is_buy  = any(x in v for x in ('BUY', 'ACCUMULATE', 'STRONG BUY'))
    is_sell = any(x in v for x in ('SELL', 'REDUCE'))
    is_hold = 'HOLD' in v

    tech_score = int(data.get('tech_score', 50) or 50)
    fund_score = int(data.get('blended_score', 50) or 50)
    weak_tech  = tech_score < 45
    strong_fund = fund_score >= 65

    if is_buy and bearish:
        dt = "contrarian_early"
        label = "Tactical BUY — Contrarian Early"
        fields = {
            "why_now": "Oversold technicals + fundamental value divergence from price action",
            "what_confirms": "Price reclaim above SMA50; RSI sustained above 50; volume expansion on up-days",
            "what_invalidates": "New 52-week low; breakdown below key support; bearish macro catalyst",
        }
    elif is_buy and neutral:
        dt = "early_reversal"
        label = "BUY — Early Reversal"
        fields = {
            "why_now": "Neutral trend with improving momentum — potential inflection point",
            "confirmation_triggers": "MACD bullish crossover; price above SMA50; volume > 20-day average",
            "failure_conditions": "Rejection at SMA200; RSI fails to hold above 50; earnings miss",
        }
    elif is_buy and bullish:
        dt = "trend_confirmed"
        label = "BUY — Trend Confirmed"
        fields = {
            "why_now": "Confirmed uptrend — price above rising SMA200 with positive momentum",
            "continuation_conditions": "Price holds above SMA50 on pullbacks; volume confirms direction; macro tailwind intact",
            "invalidation_level": f"Close below SMA200 ({data.get('sma200', 'N/A')}); or breakdown with high volume",
        }
    elif is_hold and weak_tech and strong_fund:
        dt = "wait_for_confirmation"
        label = "HOLD — Wait for Confirmation"
        fields = {
            "why_hold": "Strong fundamentals offset by weak/uncertain technicals — timing not yet optimal",
            "what_confirms": "Technical trend alignment (price above SMA50 + MACD crossover) before scaling in",
            "no_action_case": "If confirmation doesn't materialize within 2-3 weeks, reassess entry thesis",
        }
    elif is_sell:
        dt = "trend_failure"
        label = "SELL — Trend Failure"
        fields = {
            "why_now": "Breakdown in trend structure — technical deterioration outweighs any fundamental support",
            "what_confirms": "Exit or reduce on any relief rally; do not add to losing positions",
            "what_invalidates": "Strong earnings beat + guidance raise + reclaim of SMA200",
        }
    else:
        dt = "wait_for_confirmation"
        label = f"HOLD — Wait for Confirmation"
        fields = {
            "why_hold": "Insufficient directional conviction — mixed signals across fundamental and technical dimensions",
            "what_confirms": "Clear trend establishment with volume confirmation",
            "no_action_case": "If no clear signal within next quarter, revisit asset allocation",
        }

    return {"decision_type": dt, "label": label, **fields}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MARKDOWN RENDERER
# ══════════════════════════════════════════════════════════════════════════════

