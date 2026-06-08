"""scorecard_engine.py -- calculate_score and generate_scorecard_markdown."""
import re
from typing import Optional

from core.scorecard_parser import sanitize_field, render_field, parse_report, _NEAR_ZERO_BETA_ALLOWED

def calculate_score(data: dict) -> Optional[tuple]:
    """
    نظام تسجيل محايد — كل factor عنده منطق واضح.
    
    التوزيع (100 نقطة):
      Quality Score         25 نقطة
      Valuation              20 نقطة
      Price Upside           15 نقطة
      Risk Profile           15 نقطة  ← البيتا هنا بتخصم
      Market Position        10 نقطة
      Technical Momentum     10 نقطة
      Sentiment              5 نقطة
      ─────────────────────────────
      Total                  100
    """
    if not data or not data.get('price'):
        return None

    factors = {}
    _ticker = str(data.get('ticker', ''))
    _is_crypto = _ticker.endswith('-USD') and any(c in _ticker for c in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT'])

    # ──────────────────────────────────────────────
    # 1. FUNDAMENTAL QUALITY (25 نقطة)
    # ──────────────────────────────────────────────
    if _is_crypto:
        # Crypto: fundamentals لا تنطبق — نعطي score محايد (12/25 = ~50%)
        # مع مكافأة خفيفة للعملات الكبيرة حسب market cap
        mc = data.get('mc', 0) or 0
        if mc >= 500e9:   q_score = 15  # BTC/ETH class
        elif mc >= 50e9:  q_score = 13
        elif mc >= 5e9:   q_score = 11
        else:             q_score = 8
        factors["Quality Score"] = (min(q_score, 25), 25)
    else:
        # ── Deterministic Quality Score — fixed 28-point denominator always ──
        # Each field has a neutral fallback so q_avail is ALWAYS 28.
        # Missing fields use sector-neutral midpoints — never skipped.
        # Removed: 'quality' from LLM report (circular dependency → non-deterministic).
        q_earned = 0
        Q_TOTAL = 28  # fixed denominator: nm(8) + rg(7) + roe(5) + gm(5) + de(2) + roic(1)
        #              smaller optional weights so missing ROIC/DE have minimal impact

        # Gross Margin (weight 5) — replaces circular LLM 'quality' field
        gm = data.get('gross_margin')
        if gm is not None:
            if gm >= 65:   q_earned += 5   # SaaS / high-IP
            elif gm >= 40: q_earned += 4   # strong
            elif gm >= 25: q_earned += 3   # solid industrial/energy
            elif gm >= 10: q_earned += 2   # thin margin business
            elif gm >= 0:  q_earned += 1
            # negative gross margin → 0
        else:
            q_earned += 3  # neutral fallback (60%)

        # Net Margin (weight 8) — primary profitability signal
        nm = data.get('net_margin')
        if nm is not None:
            if nm >= 35:   q_earned += 8
            elif nm >= 20: q_earned += 7
            elif nm >= 12: q_earned += 5
            elif nm >= 5:  q_earned += 3
            elif nm >= 0:  q_earned += 1
            # negative → 0
        else:
            q_earned += 4  # neutral fallback (50%)

        # Revenue Growth (weight 7)
        rg = data.get('rev_growth')
        if rg is not None:
            if rg >= 50:    q_earned += 7
            elif rg >= 25:  q_earned += 5
            elif rg >= 10:  q_earned += 3
            elif rg >= 0:   q_earned += 2
            elif rg >= -10: q_earned += 1
        else:
            q_earned += 3  # neutral fallback (~43%)

        # ROE (weight 5)
        roe = data.get('roe')
        if roe is not None:
            if roe >= 20:   q_earned += 5
            elif roe >= 12: q_earned += 4
            elif roe >= 6:  q_earned += 2
            elif roe >= 2:  q_earned += 1
        else:
            q_earned += 2  # neutral fallback (40%)

        # D/E Ratio (weight 2) — reduced weight; often missing for non-US
        de = data.get('debt_equity')
        if de is not None:
            try:
                de_f = float(de)
                if de_f < 0:     q_earned += 2
                elif de_f < 0.7: q_earned += 2
                elif de_f < 1.5: q_earned += 1
                # D/E >= 1.5 → 0
            except (ValueError, TypeError):
                q_earned += 1  # parse error → neutral
        else:
            q_earned += 1  # neutral fallback (50%)

        # ROIC (weight 1) — reduced weight; inconsistently available across data sources
        roic = data.get('roic')
        if roic is not None:
            try:
                roic_f = float(roic)
                roic_pct = roic_f * 100 if abs(roic_f) <= 1.0 else roic_f
                if roic_pct >= 12: q_earned += 1
                # below 12% → 0
            except (ValueError, TypeError):
                pass
        # ROIC missing → 0 contribution (weight too small to skew result)

        # Scale to 25-point space — denominator is always Q_TOTAL=28
        q_score = round(q_earned / Q_TOTAL * 25)

        # Large-cap safety floor — mega-cap with good fundamentals scores well naturally
        _mc = data.get('mc', 0) or 0
        if _mc >= 200e9:  q_score = max(q_score, 10)
        elif _mc >= 50e9: q_score = max(q_score, 8)
        elif _mc >= 10e9: q_score = max(q_score, 6)
        factors["Quality Score"] = (min(q_score, 25), 25)

    # ──────────────────────────────────────────────
    # 2. VALUATION (20 نقطة) — P/E عالي بيخصم
    # ──────────────────────────────────────────────
    if _is_crypto:
        # Crypto: P/E لا ينطبق — نستخدم Price vs SMA200 كـ proxy
        sma200 = data.get('sma200') or 0
        price  = data.get('price') or 0
        if sma200 and price:
            pct_from_sma = (price - sma200) / sma200
            if pct_from_sma < -0.30:    v_score = 18  # deep value — شديد الرخص
            elif pct_from_sma < -0.10:  v_score = 15  # undervalued
            elif pct_from_sma < 0.10:   v_score = 12  # fair value
            elif pct_from_sma < 0.30:   v_score = 8   # stretched
            else:                       v_score = 4   # overextended
        else:
            v_score = 10  # neutral
        factors["Valuation"] = (v_score, 20)
    else:
        v_score = 10  # default neutral

        fpe = data.get('forward_pe')
        if fpe is not None and fpe > 0:
            if fpe <= 8:     v_score = 18   # very cheap — but could be value trap
            elif fpe <= 15:  v_score = 17   # cheap
            elif fpe <= 25:  v_score = 14   # reasonable
            elif fpe <= 35:  v_score = 10   # pricey
            elif fpe <= 50:  v_score = 7    # expensive — needs growth to justify
            elif fpe <= 70:  v_score = 4    # very expensive
            else:            v_score = 2    # bubble territory

        # لو في growth عالي (>30%) يعوّض الـ high PE — max +4
        rg = data.get('rev_growth', 0) or 0
        growth_compensation = min(4, int(rg / 15))
        v_score = min(20, v_score + growth_compensation)

        # Value trap penalty: cheap PE + bearish trend = not attractive
        _is_bear = data.get('is_bearish') or data.get('bearish_count', 0) >= 2
        if fpe and fpe <= 15 and _is_bear:
            v_score = max(v_score - 3, 8)  # cheap + bearish = possible value trap

        factors["Valuation"] = (v_score, 20)

    # ──────────────────────────────────────────────
    # 3. PRICE UPSIDE / ATH RECOVERY (15 نقطة)
    # ──────────────────────────────────────────────
    if _is_crypto:
        # Crypto: use distance from ATH (52-week high as proxy)
        ath = data.get('year_high') or data.get('ath') or 0
        price = data.get('price') or 0
        if ath and price and ath > 0:
            ath_drop = (price - ath) / ath  # negative = below ATH
            if ath_drop > -0.10:     u_score = 5   # near ATH — limited upside
            elif ath_drop > -0.25:   u_score = 9   # moderate recovery potential
            elif ath_drop > -0.40:   u_score = 12  # significant upside
            elif ath_drop > -0.60:   u_score = 14  # deep discount
            else:                    u_score = 8   # may be broken — caution
        else:
            u_score = 7
        factors["ATH Recovery Potential"] = (u_score, 15)
    else:
        if data.get('target') and data.get('price'):
            upside = (data['target'] - data['price']) / data['price']
            if upside >= 0.40:   u_score = 15
            elif upside >= 0.25: u_score = 12
            elif upside >= 0.15: u_score = 9
            elif upside >= 0.05: u_score = 6
            elif upside >= 0:    u_score = 3
            else:                u_score = 0  # downside
            # Discount SMA/technical targets — they're mean-reversion levels, not analyst conviction
            if data.get('target_is_sma'):
                u_score = min(u_score, 9)
            # Risk-adjust upside when stock is in a bearish trend (below SMA200):
            # The analyst target is still valid long-term, but near-term probability is lower.
            # A 100% upside score while in a downtrend creates a misleading BUY signal.
            _sc_sma = data.get('sma200', 0) or 0
            _sc_pr  = data.get('price', 0) or 0
            if _sc_sma and _sc_pr:
                _sma_gap = (_sc_pr - _sc_sma) / _sc_sma   # negative = below SMA200
                if _sma_gap < -0.20:
                    u_score = min(u_score, 9)   # >20% below SMA200 → cap at 60%
                elif _sma_gap < -0.10:
                    u_score = min(u_score, 12)  # 10-20% below → cap at 80%
        else:
            u_score = 5  # no target = neutral
        factors["Price Upside"] = (u_score, 15)

    # ──────────────────────────────────────────────
    # 4. RISK PROFILE (15 نقطة) — البيتا بتخصم هنا
    # ──────────────────────────────────────────────
    r_score = 10  # default

    beta = data.get('beta')
    if beta is not None:
        if beta <= 0.8:    r_score = 15  # low vol — defensive
        elif beta <= 1.2:  r_score = 13  # market-like
        elif beta <= 1.5:  r_score = 10  # moderate risk
        elif beta <= 2.0:  r_score = 7   # high risk
        elif beta <= 2.5:  r_score = 4   # very high risk
        else:              r_score = 2   # extreme risk

    # لو bearish signals — خصم إضافي حسب الشدة
    bearish_c = data.get('bearish_count', 0)
    if bearish_c >= 3:
        r_score = max(5, r_score - 3)   # heavy bearish = moderate penalty
    elif bearish_c >= 2:
        r_score = max(0, r_score - 4)
    elif data.get('is_bearish'):
        r_score = max(0, r_score - 3)

    # Price below SMA200 = structural risk
    _sma = data.get('sma200', 0)
    _pr  = data.get('price', 0)
    if _sma and _pr and _pr < _sma * 0.90:
        r_score = max(2, r_score - 1)   # >10% below SMA200

    # ── Quality offset: high-quality mega-caps مع beta عالي مش نفس الـ risky stocks ──
    _quality = data.get('quality') or 0
    _mc_r = data.get('mc', 0) or 0
    _above_sma = (_pr and _sma and _pr > _sma)
    if beta and beta > 1.5:
        # لو quality عالي + mega-cap + فوق SMA200 → يخفف الـ penalty
        if _quality >= 90 and _mc_r >= 500e9 and _above_sma:
            r_score = min(r_score + 4, 11)   # NVDA class — عالي الجودة
        elif _quality >= 80 and _mc_r >= 100e9:
            r_score = min(r_score + 2, 9)    # large-cap quality offset
        elif _quality >= 70:
            r_score = min(r_score + 1, 8)    # moderate quality offset

    # لو في كتير من الـ risks المذكورة — خصم
    risk_count = data.get('risk_count', 0)
    if risk_count >= 4:
        r_score = max(0, r_score - 2)

    r_score = max(r_score, 2)  # RULE: risk_score floor — minimum 10% (2/15 ≈ 13%)
    factors["Risk Profile"] = (r_score, 15)

    # ──────────────────────────────────────────────
    # 5. MARKET POSITION / NETWORK DOMINANCE (10 نقطة)
    # ──────────────────────────────────────────────
    if _is_crypto:
        mc = data.get('mc', 0) or 0
        if mc >= 1e12:    mp_score = 10  # BTC class — $1T+
        elif mc >= 200e9: mp_score = 8   # ETH class
        elif mc >= 50e9:  mp_score = 6   # top 5
        elif mc >= 10e9:  mp_score = 4   # top 20
        elif mc >= 1e9:   mp_score = 2   # mid cap
        else:             mp_score = 1   # small cap
        factors["Network Dominance"] = (mp_score, 10)
    else:
        mp_score = 0
        mc = data.get('mc', 0) or 0
        if mc >= 500e9:   mp_score += 5
        elif mc >= 100e9: mp_score += 4
        elif mc >= 10e9:  mp_score += 3
        elif mc >= 1e9:   mp_score += 2
        else:             mp_score += 1

        moat_signals = data.get('moat_signals', 0)
        if moat_signals >= 3:   mp_score += 5
        elif moat_signals >= 2: mp_score += 3
        elif moat_signals == 1: mp_score += 1
        # Mega-cap bonus: MC>500B with quality>80 = implied moat even if not in text
        _mc = data.get('mc', 0) or 0
        _q  = data.get('quality', 0) or 0
        if _mc >= 500e9 and _q >= 80 and moat_signals == 0:
            mp_score += 2  # implied moat for world-class mega caps
        factors["Market Position"] = (min(mp_score, 10), 10)

    # ──────────────────────────────────────────────
    # 6. TECHNICAL MOMENTUM (10 نقطة)
    # ──────────────────────────────────────────────
    t_score = 5  # default neutral

    rsi = data.get('rsi')
    adx = data.get('adx')
    bullish_c = data.get('bullish_count', 0)
    bearish_c = data.get('bearish_count', 0)
    _sma200 = data.get('sma200', 0)
    _price  = data.get('price', 0)

    # RSI — healthy range is bullish, extremes are warning
    if rsi is not None:
        if 45 <= rsi <= 65:  t_score += 2   # healthy momentum
        elif rsi < 30:       t_score += 1   # deeply oversold = contrarian signal
        elif rsi < 40:       t_score += 0   # mildly oversold = neutral
        elif rsi > 75:       t_score -= 2   # overbought

    # ADX — only reward strong trend if it's BULLISH (price above SMA200)
    if adx is not None:
        _above_sma = (_price and _sma200 and _price > _sma200)
        if adx >= 25 and _above_sma:   t_score += 2   # strong bullish trend
        elif adx >= 25 and not _above_sma: t_score -= 1  # strong BEARISH trend = bad
        elif adx >= 15: t_score += 0  # neutral
        else:           t_score -= 1  # weak/choppy

    # Price vs SMA200 — core directional signal
    if _price and _sma200:
        _pct_from_sma = (_price - _sma200) / _sma200
        if _pct_from_sma > 0.05:    t_score += 1   # above SMA200
        elif _pct_from_sma < -0.10: t_score -= 2   # well below SMA200

    # Net sentiment signals
    net_signals = bullish_c - bearish_c
    if net_signals >= 3:    t_score += 2
    elif net_signals >= 1:  t_score += 1
    elif net_signals <= -2: t_score -= 2

    factors["Technical Momentum"] = (max(0, min(t_score, 10)), 10)

    # ──────────────────────────────────────────────
    # 7. ANALYST SENTIMENT / MARKET SENTIMENT (5 نقطة)
    # ──────────────────────────────────────────────
    if _is_crypto:
        # Crypto: use Fear & Greed Index as contrarian indicator
        fg = data.get('fear_greed', 50)
        if fg <= 15:     s_score = 4   # Extreme Fear — contrarian buy signal
        elif fg <= 30:   s_score = 3   # Fear — opportunity forming
        elif fg <= 55:   s_score = 3   # Neutral
        elif fg <= 75:   s_score = 2   # Greed — caution
        else:            s_score = 1   # Extreme Greed — danger
        # Bonus if price is deeply oversold + extreme fear (contrarian)
        if fg <= 20 and data.get('rsi', 50) < 35:
            s_score = 5
        factors["Analyst Sentiment"] = (s_score, 5)
    else:
        # Primary signal: real Wall Street analyst consensus (from data dict).
        # Fallback to llm_verdict (EisaX technical) only when no analyst data exists.
        real_consensus = (data.get('analyst_consensus') or '').lower().strip()
        sector_bonus   = 1 if data.get('is_tech') else 0

        if real_consensus:
            # Use actual analyst consensus — this reflects professional coverage directly
            if 'strong buy' in real_consensus:          s_score = 5
            elif 'buy' in real_consensus:               s_score = 4
            elif 'hold' in real_consensus or 'neutral' in real_consensus:
                                                        s_score = 3
            elif 'underperform' in real_consensus or 'reduce' in real_consensus:
                                                        s_score = 2
            elif 'sell' in real_consensus:              s_score = 1
            else:                                       s_score = 3  # unknown label → neutral
            # Reduce confidence when analyst count is low (<5 analysts)
            _n_analysts = data.get('analyst_count') or 0
            if _n_analysts and int(_n_analysts) < 5:
                s_score = max(s_score - 1, 1)
        else:
            # No analyst coverage: fall back to EisaX technical verdict
            llm_v = data.get('llm_verdict', 'HOLD')
            if 'STRONG BUY' in llm_v:   s_score = 4   # one notch lower — it's model-generated
            elif 'BUY' in llm_v:         s_score = 3
            elif 'SELL' in llm_v:        s_score = 1
            else:                        s_score = 2   # HOLD / REDUCE / unknown
            # No analyst coverage — cap further: it's EisaX FV only
            if data.get('target_is_estimate'):
                s_score = min(s_score, 2)

        # N/A when there is no genuine analyst coverage. The "Analyst Sentiment /
        # Wall St Consensus" pillar must reflect real sell-side coverage — never an
        # EisaX technical proxy dressed up as consensus. `target_is_estimate` (the
        # report's own "EisaX FV Estimate — no analyst coverage" signal) is
        # authoritative: some names (e.g. ADNOCGAS) carry a stray consensus rating
        # string yet have no analyst target, so honour the estimate flag too.
        _no_coverage = (not real_consensus) or bool(data.get('target_is_estimate'))
        if _no_coverage:
            # Show as 0/5 — keeps max = 100 always, no confusing rescaling
            # Flag for display: pillar table shows "(no coverage)" label
            data['_analyst_na'] = True
            factors["Analyst Sentiment"] = (0, 5)
        else:
            factors["Analyst Sentiment"] = (min(s_score + sector_bonus, 5), 5)

    # ──────────────────────────────────────────────
    # FINAL SCORE
    # ──────────────────────────────────────────────
    raw_total  = sum(v[0] for v in factors.values())
    _total_max = sum(v[1] for v in factors.values())

    # Rescale to /100 when factors were excluded (e.g. Analyst Sentiment N/A)
    if _total_max > 0 and _total_max < 100:
        final = round(raw_total / _total_max * 100)
    else:
        final = raw_total

    # Hard cap: لا سهم يأخد 100 — دايماً في uncertainty
    final = min(final, 95)  # RULE: quality_score = min(raw, 95) — never 100%
    # OVERCONFIDENCE GUARD: never use "perfect", "flawless", "guaranteed" in output
    # This is enforced at the prompt/LLM level — scorecard labels are calibrated above

    # Bearish trend penalty: price below SMA200 = structural headwind — cap at 70
    _is_below_sma200 = (data.get('sma200') and data.get('price') and
                        data['price'] < data['sma200'] * 0.98)
    if _is_below_sma200:
        _raw_score = final
        final = min(final, 69)  # can't be high-conviction BUY while below SMA200
        data['_score_capped'] = True
        data['_raw_score'] = _raw_score
        data['_cap_reason'] = "Price below SMA200 — technical override applied"
    else:
        data['_score_capped'] = False

    # Crash/extreme move penalty: single-day drop > 30% = forensic alert — cap at 60
    _daily_change = data.get('daily_change_pct', 0) or 0
    if _daily_change <= -30:
        final = min(final, 60)  # extreme crash = max caution

    return factors, final, data


def generate_scorecard_markdown(report_text: str) -> str:
    """Generate the full scorecard markdown for chat display."""
    data = parse_report(report_text)
    if not data:
        return ""

    result = calculate_score(data)
    if not result:
        return ""

    factors, final, data = result
    verdict_label, verdict_emoji, conviction = get_verdict(final, data)

    # Compute decision type classification
    decision_info = compute_decision_type(verdict_label, data)
    decision_type = decision_info["decision_type"]
    decision_label = decision_info["label"]

    upside = 0
    if data.get('target') and data.get('price'):
        upside = (data['target'] - data['price']) / data['price'] * 100

    # ── Build conviction note ──
    # ── Build conviction note ──
    conviction_notes = []
    beta = data.get('beta')
    fpe = data.get('forward_pe')
    is_negative = verdict_label in ("REDUCE", "SELL", "AVOID")
    is_positive = verdict_label in ("BUY", "ACCUMULATE", "STRONG BUY")

    if beta and beta > 2.0:
        conviction_notes.append(f"high beta risk (β={beta:.2f})")
    if fpe and fpe > 40:
        conviction_notes.append(f"premium valuation (fwd P/E {fpe:.1f}x)")

    # ── Upside: only mention if verdict allows ──
    if upside > 25 and not is_negative:
        conviction_notes.append("strong upside potential (+{:.0f}%)".format(upside))
    elif upside > 25 and is_negative:
        conviction_notes.append("upside limited by bearish technicals")

    # ── Quality: only positive framing for positive verdicts ──
    quality = data.get('quality')
    if quality and quality >= 90 and is_positive:
        conviction_notes.append("exceptional quality score")
    elif quality and quality >= 90 and is_negative:
        conviction_notes.append("strong fundamentals offset by technicals")

    if data.get('has_moat') and is_positive:
        conviction_notes.append("dominant moat")

    # ── Verdict-specific notes ──
    if is_negative:
        if data.get('bearish_trend'):
            conviction_notes.append("bearish primary trend")
        if not data.get('analyst_coverage'):
            conviction_notes.append("no analyst coverage — EisaX FV estimate only")
    
    conviction_str = ", ".join(conviction_notes) if conviction_notes else "balanced risk/reward"
    conviction_str = ", ".join(conviction_notes) if conviction_notes else "balanced risk/reward"

    lines = []
    lines.append("---")
    lines.append("## 🎯 EisaX Proprietary Score Card")
    # Canonical evidence label (replaces Low/Medium/High conviction)
    try:
        from core.services.decision_policy import canonical_evidence as _se_ce
        _engine_evidence = _se_ce(conviction)
    except Exception:
        _engine_evidence = conviction
    lines.append(
        f"**{data['ticker']}** | **{decision_label} {verdict_emoji}** | "
        f"Evidence: **{_engine_evidence}** | EisaX Score: **{final}/100** | Blended: **{data.get('blended_score', final)}/100**"
    )

    # Decision Type Classification
    _dt_label_map = {
        "contrarian_early": "Contrarian Early",
        "early_reversal": "Early Reversal",
        "trend_confirmed": "Trend Confirmed",
        "wait_for_confirmation": "Wait for Confirmation",
        "trend_failure": "Trend Failure",
    }
    _dt_display = _dt_label_map.get(decision_type, decision_type.replace("_", " ").title())
    lines.append("")
    lines.append(f"**Decision Type: {_dt_display}**")
    # Add primary driver and confirmation from decision_info fields
    _driver_keys = ["why_now", "why_hold"]
    _confirm_keys = ["what_confirms", "confirmation_triggers", "continuation_conditions"]
    _invalid_keys = ["what_invalidates", "failure_conditions", "invalidation_level", "no_action_case"]
    _driver = next((decision_info[k] for k in _driver_keys if k in decision_info), "")
    _confirm = next((decision_info[k] for k in _confirm_keys if k in decision_info), "")
    _invalid = next((decision_info[k] for k in _invalid_keys if k in decision_info), "")
    if _driver:
        lines.append(f"*Primary Driver: {_driver}*")
    if _confirm:
        lines.append(f"*Confirmation Needed: {_confirm}*")
    elif _invalid:
        lines.append(f"*Invalidation: {_invalid}*")
    lines.append("")

    lines.append("| Score Breakdown | Value |")
    lines.append("|-----------------|-------|")
    lines.append(f"| Tech Signal Score | {data.get('tech_score', '—')}/100 |")
    lines.append(f"| Blended Score | {data.get('blended_score', '—')}/100 |")
    lines.append("")

    if conviction_notes:
        lines.append(f"*Conviction driven by: {conviction_str}*")
        lines.append("")

    # Key metrics
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Live Price | ${data['price']:,.2f} |")
    if data.get('target'):
        sign = "+" if upside >= 0 else ""
        lines.append(f"| Price Target | ${data['target']:,.2f} ({sign}{upside:.1f}%) |")
    _beta_clean = sanitize_field("beta", beta, data.get('ticker', ''))
    if _beta_clean is not None:
        lines.append(f"| Beta | {_beta_clean:.2f} {'⚠️ High Risk' if _beta_clean > 2 else ''} |")
    elif beta is not None:
        lines.append(f"| Beta | N/A (data unverified) |")
    _fpe_clean = sanitize_field("forward_pe", fpe, data.get('ticker', ''))
    if _fpe_clean is not None:
        pe_note = "🔴 Very High" if _fpe_clean > 60 else ("🟡 High" if _fpe_clean > 35 else "🟢 Reasonable")
        lines.append(f"| Forward P/E | {_fpe_clean:.1f}x  {pe_note} |")
    elif fpe is not None:
        lines.append(f"| Forward P/E | N/A (data unverified) |")
    lines.append("")

    # Factor table
    lines.append("**Factor Analysis:**")
    lines.append("")
    lines.append("| Factor | Score | Bar |")
    lines.append("|--------|-------|-----|")

    for name, (val, max_v) in factors.items():
        pct = int((val / max_v) * 100)
        filled = int((val / max_v) * 10)
        bar = "█" * filled + "░" * (10 - filled)
        emoji = "🟢" if pct >= 75 else ("🟡" if pct >= 50 else "🔴")
        lines.append(f"| {name} | {pct}% | {emoji} `{bar}` |")

    lines.append("")

    # Overall bar
    filled_big = int((final / 100) * 20)
    big_bar = "█" * filled_big + "░" * (20 - filled_big)
    lines.append(f"**Overall: `{big_bar}` {final}/100**")
    lines.append("")
    lines.append("> *EisaX Proprietary Score | Abu Dhabi*")
    lines.append("---")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # NVDA — يفترض يجيب ~78-82 مش 99
    nvda_report = """
EisaX Intelligence Report: NVDA
Live Price: $193.23 | Sector: Technology | Quality Score: 99/100
Market Cap: 4.61T | Beta: 2.31 | Forward P/E: 26.9x | P/E (TTM): 47.8x
Target: $255.82
Net margin: 53.0% | Gross margin: 70.0% | ROE: 107.4%
Revenue growth: 62.5%
RSI: 49.7 | ADX: 9.5
The primary trend is Bullish, with price above the rising SMA200.
Has dominant market position and unassailable moat in AI infrastructure.
VERDICT: BUY
Key Risks: valuation risk, high beta, multiple compression risk, concentration risk.
"""

    # سهم متوسط — يفترض يجيب ~55-65
    avg_report = """
EisaX Intelligence Report: XOM
Live Price: $110.00 | Sector: Energy | Quality Score: 65/100
Market Cap: 440B | Beta: 0.9 | Forward P/E: 14.0x
Target: $120.00
Net margin: 10.5%
RSI: 52.0
Momentum is neutral. Some pricing power in energy sector.
VERDICT: HOLD
Key Risks: oil price volatility, regulatory risk.
"""

    print("=" * 60)
    print("TEST 1: NVDA (expected: 75-85)")
    print("=" * 60)
    print(generate_scorecard_markdown(nvda_report))

    print("\n" + "=" * 60)
    print("TEST 2: XOM (expected: 55-70)")
    print("=" * 60)
    print(generate_scorecard_markdown(avg_report))
