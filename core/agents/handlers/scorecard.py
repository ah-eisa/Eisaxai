# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.finance_helpers import _safe_div_yield, _consensus_divergence
logger = logging.getLogger(__name__)


class ScorecardMixin:
    def _build_decision_framework_block(self,
                                        *,
                                        verdict: str,
                                        confidence: int,
                                        conviction: str,
                                        conviction_note: str = "",
                                        beta: float,
                                        current_price: float,
                                        entry_price: float,
                                        sma50: float,
                                        next_earnings: Optional[str],
                                        currency_sym: str,
                                        is_local_mkt: bool,
                                        is_arabic: bool,
                                        is_crypto: bool = False,
                                        is_etf: bool = False,
                                        is_commodity: bool = False,
                                        is_reit: bool = False) -> str:
        """Build a compact advisory layer: confidence, uncertainty, horizon, and no-action case."""
        _verdict = str(verdict or "HOLD").upper()
        _beta = float(beta or 1.0)
        _cp = float(current_price or 0.0)
        _ep = float(entry_price or 0.0)
        _sma50 = float(sma50 or 0.0)
        _conviction_note_line = f"{conviction_note}\n" if conviction_note else ""

        def _fmt_price(v: float) -> str:
            if not v:
                return "N/A"
            return f"{v:,.2f} {currency_sym}" if is_local_mkt else f"${v:,.2f}"

        if _verdict in ("SELL", "REDUCE", "AVOID"):
            no_action_en = (
                "If price remains below SMA50 and no catalyst quality improves, keep exposure unchanged and await trend repair confirmation."
            )
            no_action_ar = "لو السعر فضل تحت SMA50 ومفيش تحسن واضح في الكاتاليست، الأفضل تثبيت المراكز والانتظار لحد تأكيد إصلاح الاتجاه."
        elif _verdict == "HOLD":
            if is_crypto:
                no_action_en = (
                    "If price stays in a range with no directional catalyst or on-chain signal shift, keep allocation unchanged and await a higher-conviction setup."
                )
                no_action_ar = "لو السعر في نطاق جانبي بدون محفز اتجاهي أو تحول في إشارات الأونشين، يفضل الإبقاء على التوزيع الحالي وانتظار إشارة أوضح."
            else:
                no_action_en = (
                    "If price stays in a range with no catalyst surprise, keep allocation unchanged and monitor entry conditions before adding."
                )
                no_action_ar = "لو السعر في نطاق جانبي بدون مفاجآت محفزة، يفضل الإبقاء على التوزيع الحالي ومراقبة شروط الدخول قبل أي إضافة."
        elif _cp and _ep and _cp > (_ep * 1.02):
            no_action_en = f"If price stays above the preferred entry zone ({_fmt_price(_ep)}), avoid chasing and await pullback confirmation."
            no_action_ar = f"لو السعر استمر أعلى من منطقة الدخول المفضلة ({_fmt_price(_ep)})، الأفضل عدم المطاردة وانتظار تأكيد البولباك."
        else:
            no_action_en = (
                "If confirmation above SMA50 fails, avoid adding and keep allocation unchanged."
            )
            no_action_ar = "لو فشل تأكيد الإغلاق أعلى SMA50، الأفضل عدم زيادة المراكز والإبقاء على التوزيع الحالي."

        # ── Uncertainty Driver 1: asset-type-specific ────────────────────
        if is_crypto:
            earn_u_en = "On-chain signals (MVRV, exchange outflows) and macro liquidity cycle shifts can rapidly reprice the asset."
            earn_u_ar = "إشارات الأونشين (MVRV، تدفق البورصات) وتحولات دورة السيولة الكلية قد تعيد التسعير بسرعة."
        elif is_etf or is_commodity:
            earn_u_en = "Supply/demand balance shifts, OPEC+ decisions, or macro regime change may materially move this asset."
            earn_u_ar = "تحولات ميزان العرض والطلب أو قرارات أوبك+ أو تغيير النظام الكلي قد تؤثر جوهرياً على هذا الأصل."
        elif is_reit:
            earn_u_en = "Rate path and occupancy cycle data are the primary re-rating triggers for this REIT."
            earn_u_ar = "مسار أسعار الفائدة ودورة الإشغال هما المحركان الأساسيان لإعادة تقييم صندوق العقارات هذا."
        else:
            earn_u_en = f"Upcoming earnings on {next_earnings} may materially reset guidance." if next_earnings else "Upcoming earnings/guidance timing may reset the thesis."
            earn_u_ar = f"نتائج الأرباح القادمة في {next_earnings} قد تعيد ضبط الفرضية بالكامل." if next_earnings else "توقيت الأرباح/التوجيهات القادمة قد يعيد تشكيل الفرضية."

        # ── Uncertainty Driver 2: beta / volatility ──────────────────────
        if is_crypto:
            macro_u_en = "Crypto vol is structurally elevated (annualised ~60-90%); position sizing should reflect this — not standard equity beta."
            macro_u_ar = "تذبذب العملات الرقمية مرتفع هيكلياً (~60-90% سنوياً)؛ يجب أن يعكس حجم المركز هذا الواقع لا معامل بيتا التقليدي."
        elif is_reit:
            macro_u_en = "Rate sensitivity is the dominant risk factor; a 50bps move in the 10Y can shift NAV by 5-10%."
            macro_u_ar = "حساسية أسعار الفائدة هي عامل المخاطر الأساسي؛ حركة 50 نقطة أساس في العشر سنوات قد تحرك NAV بنسبة 5-10%."
        elif _beta >= 1.5:
            macro_u_en = f"High beta sensitivity ({_beta:.2f}x) can amplify market drawdowns."
            macro_u_ar = f"حساسية بيتا مرتفعة ({_beta:.2f}x) وقد تضخم أي هبوط سوقي."
        elif _beta >= 1.1:
            macro_u_en = f"Moderate beta/rate sensitivity ({_beta:.2f}x) can shift risk/reward quickly."
            macro_u_ar = f"حساسية بيتا/الفائدة متوسطة ({_beta:.2f}x) وقد تغير معادلة المخاطرة بسرعة."
        else:
            macro_u_en = "Macro beta sensitivity is limited; thesis risk is more company-specific."
            macro_u_ar = "حساسية البيتا الكلية محدودة، والمخاطر الأكبر مرتبطة بتنفيذ الشركة نفسها."

        if is_arabic:
            return (
                "\n\n---\n"
                "## إطار القرار (Advisory Layer)\n"
                f"- **ثقة القرار:** {confidence}%\n"
                f"{_conviction_note_line}"
                "- **الأفق الزمني:** تكتيكي 1-3 أشهر | استراتيجي 12-36 شهر\n"
                f"- **حالة عدم اتخاذ إجراء:** {no_action_ar}\n"
                f"- **عوامل عدم اليقين:** 1) {earn_u_ar} 2) {macro_u_ar}\n"
                "> الهدف من هذا الإطار هو دعم القرار وليس إصدار أوامر تنفيذية مباشرة."
            )

        return (
            "\n\n---\n"
            "## Decision Framework (Advisory Layer)\n"
            f"- **Verdict Confidence:** {confidence}% (Conviction: {conviction})\n"
            f"{_conviction_note_line}"
            "- **Time Horizon:** Tactical 1-3 months | Strategic 12-36 months\n"
            f"- **No-Action Case:** {no_action_en}\n"
            f"- **Primary Uncertainty Drivers:** 1) {earn_u_en} 2) {macro_u_en}\n"
            "> This layer is advisory and supports decision quality; it is not an execution command."
        )

    def _compute_rolling_beta(self, ticker: str) -> float:
        """Compute 90-day rolling beta vs SPY for any ticker. Returns float."""
        try:
            import numpy as _np
            from core.data import get_prices as _gp
            _prices = _gp([ticker, "SPY"], start="2025-01-01", end=None, force_refresh=False)
            if ticker in _prices.columns and "SPY" in _prices.columns:
                _cr = _prices[ticker].pct_change().dropna()
                _sr = _prices["SPY"].pct_change().dropna()
                _common = _cr.index.intersection(_sr.index)
                if len(_common) > 30:
                    _cv = _np.cov(_cr.loc[_common].values, _sr.loc[_common].values)
                    beta = round(float(_cv[0, 1] / _cv[1, 1]) if _cv[1, 1] != 0 else 1.0, 2)
                    return max(0.3, min(beta, 4.0))
        except Exception as _e:
            logger.warning(f"[RollingBeta] Failed for {ticker}: {_e}")
        return 1.5 if ticker.endswith('-USD') else 1.0

    def _build_scorecard_md(self, target, real_price, analyst_target, fund, summary, dc_data, forward_pe, fg_data=None, onchain=None, effective_beta=None, display_target=None, target_is_estimate=False, target_is_sma=False, analyst_consensus=None, change_pct=0):
        """Build the EisaX Proprietary Score Card markdown block."""
        try:
            from core.scorecard import calculate_score, get_verdict
            # Use display_target (analyst OR EisaX FV) for upside/score calculations
            _display_target = display_target if display_target is not None else analyst_target
            _target_is_estimate = target_is_estimate

            # ── Use pre-computed beta or sector-appropriate default (NOT 1.0) ──
            _is_crypto_t = target.endswith('-USD') and any(c in target for c in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'AVAX'])
            _beta_raw = effective_beta or float(fund.get('beta') or dc_data.get('beta') or summary.get('beta') or 0)
            if _beta_raw and _beta_raw > 0:
                _sc_beta = _beta_raw
            else:
                # Sector-appropriate default — NOT 1.0 which overstates risk for defensive stocks
                _s_for_beta = (fund.get('sector', '') or '').lower()
                _sc_beta = (1.5 if _is_crypto_t
                            else 0.4 if any(x in _s_for_beta for x in ('energy', 'oil', 'gas', 'utilities'))
                            else 0.7 if any(x in _s_for_beta for x in ('real estate', 'financials', 'banks'))
                            else 1.1)  # tech/general default

            # ── On-chain data for crypto ──
            _onchain = onchain or {}
            _real_ath = _onchain.get('ath') or float(fund.get('year_high') or 0)

            quality_score = (int(fund.get('fundamental_score')) if fund.get('fundamental_score') else None)
            if quality_score is not None:
                quality_score = min(quality_score, 95)

            sc_data = {
                'price': real_price or 0,
                'target': (_display_target or 0),          # analyst target OR EisaX FV estimate
                'target_is_estimate': _target_is_estimate, # flag: True = EisaX FV, not analyst
                'target_is_sma': target_is_sma,            # flag: True = SMA technical target
                'beta': _sc_beta,
                'mc': fund.get('market_cap') or 0,
                # Pass None for missing quality fields — scorecard normalises against available data
                'quality': quality_score,
                'forward_pe': float(dc_data.get('forward_pe') or forward_pe or 0),
                'ttm_pe': float(fund.get('pe_ratio') or 0),
                'net_margin': (float(fund.get('net_margin')) if fund.get('net_margin') else None),
                'gross_margin': (float(fund.get('gross_margin')) if fund.get('gross_margin') else None),
                'rev_growth': (float(fund.get('revenue_growth')) if fund.get('revenue_growth') else None),
                'roe': (float(fund.get('roe')) if fund.get('roe') else None),
                'debt_equity': (float(fund.get('debt_equity')) if fund.get('debt_equity') is not None and str(fund.get('debt_equity')) not in ('N/A', '', 'None') else None),
                'roic': (float(fund.get('roic')) if fund.get('roic') else None),
                'rsi': float(summary.get('rsi') or 50),
                'adx': float(summary.get('adx') or 20),
                'sma200': float(summary.get('sma_200') or fund.get('sma200') or summary.get('sma200') or 0),
                'year_high': _real_ath,
                'fear_greed': int((fg_data or {}).get('score', 50) or 50),
                'is_tech': fund.get('sector', '') in ['Technology', 'Semiconductors', 'Software', 'Communication Services'],
                'risk_count': sum(1 for r in ['concentration risk','valuation risk','multiple compression','cyclical','competition','regulatory','liquidity risk','high beta','interest rate'] if r in str(fund).lower() + str(summary).lower()),
                'is_bearish': summary.get('momentum') == 'Bearish',
                'ticker': target,
                'sector': fund.get('sector', 'Unknown'),
                # Real Wall Street analyst consensus — primary signal for Analyst Sentiment factor
                'analyst_consensus': analyst_consensus or '',
                'analyst_count': (int(fund.get('analyst_count') or 0) if fund.get('analyst_count') else 0),
                # Dividend yield (as decimal, e.g. 0.05 = 5%) — used in low-upside verdict override
                # yfinance may return '5.00%', '509.00%', 0.05, or 5.0 — normalise all to decimal
                'dividend_yield': _safe_div_yield(fund.get('dividend_yield') or dc_data.get('dividend_yield') or 0),
                'llm_verdict': (
                    'SELL' if (summary.get('trend') == 'Bearish' and summary.get('momentum') == 'Bearish')
                    else 'REDUCE' if (summary.get('trend') == 'Bearish' and summary.get('momentum') not in ('Bullish',) and not (_display_target and real_price and (_display_target - real_price) / real_price > 0.30))
                    # Override HOLD → BUY when upside > 30% AND analyst consensus = Strong Buy
                    else 'BUY' if (
                        summary.get('trend') == 'Bullish'
                        or (
                            _display_target and real_price
                            and ((_display_target - real_price) / real_price) > 0.30
                            and "strong buy" in (analyst_consensus or "").lower()
                        )
                    )
                    else 'HOLD'
                ),
                'is_crypto': bool(is_crypto if 'is_crypto' in dir() else False),
                'is_etf':    bool(_etf_meta_early is not None if '_etf_meta_early' in dir() else False),
                'avg_volume': float(fund.get('volume_avg90d') or fund.get('avg_volume') or 0),
                'annual_vol': float(summary.get('annual_vol', 0) or 0),
            }

            # Moat signals — guard against None values
            moat_count = 2 if (sc_data.get('quality') or 0) >= 90 else 1
            if (sc_data.get('net_margin') or 0) >= 30: moat_count += 1
            if (sc_data.get('rev_growth') or 0) >= 30: moat_count += 1
            # Regional market leader bonus — UAE/Saudi/Egypt dominant companies
            if any(str(target).upper().endswith(x) for x in ('.AE', '.DU', '.AD', '.SR', '.KW', '.CA')):
                moat_count += 1  # regional dominance premium
            sc_data['moat_signals'] = moat_count
            sc_data['has_moat'] = moat_count >= 2
            sc_data['daily_change_pct'] = change_pct or 0

            # ── Richer bullish/bearish signal counting ──
            _trend = summary.get('trend', '')
            _momentum = summary.get('momentum', '')
            _rsi_v = sc_data.get('rsi', 50)
            _sma200_v = sc_data.get('sma200', 0)
            _price_v = sc_data.get('price', 0)

            _bullish_c = 0
            if _trend == 'Bullish':         _bullish_c += 2
            if _momentum == 'Bullish':      _bullish_c += 1
            if _price_v and _sma200_v and _price_v > _sma200_v * 1.02:
                _bullish_c += 1                              # price above SMA200
            if 45 <= _rsi_v <= 65:          _bullish_c += 1  # healthy RSI range
            if summary.get('macd', 0) > summary.get('macd_signal', 0): _bullish_c += 1  # MACD bullish crossover

            _bearish_c = 0
            if _trend == 'Bearish':         _bearish_c += 2
            if _momentum == 'Bearish':      _bearish_c += 1
            if _price_v and _sma200_v and _price_v < _sma200_v * 0.95:
                _bearish_c += 1                              # price well below SMA200
            if _rsi_v < 30:                 _bearish_c += 1  # deeply oversold
            if _rsi_v > 80:                 _bearish_c += 1  # overbought → bearish risk

            sc_data['bullish_count'] = _bullish_c
            sc_data['bearish_count'] = _bearish_c
            result = calculate_score(sc_data)
            if not result:
                return ""

            factors, final, sc_data = result  # capture updated sc_data (has _score_capped, etc.)
            verdict_sc, emoji, conviction = get_verdict(final, sc_data)

            # ── Decision Engine binding layer (Week 4) ─────────────────────
            # Build interpretation labels from sc_data (ADX/RSI are deterministic).
            # Then apply hard constraints via build_decision — the result overrides
            # verdict_sc so the displayed verdict is NEVER contradicted by signals.
            try:
                from core.services.decision_engine import (
                    build_decision as _build_decision,
                    classify_decision_type as _classify_decision_type,
                )
                from core.services.interpretation_engine import (
                    build_interpretation_labels as _build_interp_labels_de,
                )
                _de_price   = float(sc_data.get('price') or 0)
                _de_sma200  = float(sc_data.get('sma200') or 0)
                _de_avgvol  = float(sc_data.get('avg_volume') or 0)
                _de_labels = _build_interp_labels_de(
                    adx=float(sc_data.get('adx') or 0),
                    rsi=float(sc_data.get('rsi') or 50),
                    price=_de_price,
                    entry_price=_de_sma200 or None,    # SMA200 as entry-zone reference
                    div_yield=float(sc_data.get('dividend_yield') or 0) or None,
                    volume_today=_de_avgvol or None,   # proxy; gives "normal" conviction
                    volume_avg=_de_avgvol or None,
                )
                _upside_for_de = (
                    (sc_data['target'] - sc_data['price']) / sc_data['price'] * 100
                    if sc_data.get('target') and sc_data.get('price')
                    else 0.0
                )
                _de_result = _build_decision(
                    interpretation_labels=_de_labels,
                    score_data={**sc_data, 'upside_pct': _upside_for_de,
                                'scorecard_verdict': verdict_sc,
                                'eisax_score': final},  # EisaX Score for Rule 8A
                )
                # Override verdict; keep emoji/conviction from scorecard
                verdict_sc    = _de_result['verdict']
                decision_type = _classify_decision_type(verdict_sc, _de_labels)

                # ── RULE 8A — Total-Return tiered enforcement ─────────────────
                # Use TOTAL RETURN (price upside + dividend yield), not price
                # upside alone, so mature dividend-rich stocks (Aramco, Gulf
                # banks, telcos) aren't permanently stuck at HOLD just because
                # their price-to-target gap is small. Threshold tightens as
                # quality drops, so weak-quality names still need a big gap.
                _upside_r8a = (
                    (sc_data['target'] - sc_data['price']) / sc_data['price'] * 100
                    if sc_data.get('target') and sc_data.get('price') else 0.0
                )
                # _safe_div_yield returns decimal (0.05 = 5%); convert to %.
                # Defensive cap at 10% — data sources occasionally hit the
                # _safe_div_yield 30% ceiling for corrupt rows (e.g. Aramco
                # has ~5% but yfinance can return 30.0). A real common-stock
                # sustainable yield above 10% is exceedingly rare; capping
                # here prevents the Rule 8A promotion from being gamed by
                # bad upstream data.
                _div_y_raw = float(sc_data.get('dividend_yield') or 0.0)
                _div_y_r8a = _div_y_raw * 100 if _div_y_raw <= 1.0 else _div_y_raw
                _div_y_r8a = min(_div_y_r8a, 10.0)
                _total_ret_r8a = _upside_r8a + _div_y_r8a
                _r8a_promote = False
                _r8a_tier    = None
                if final >= 80 and _total_ret_r8a >= 12.0:
                    _r8a_promote, _r8a_tier = True, "high-quality (≥80, TR≥12%)"
                elif final >= 70 and _total_ret_r8a >= 15.0:
                    _r8a_promote, _r8a_tier = True, "good-quality (≥70, TR≥15%)"
                elif final >= 60 and _total_ret_r8a >= 20.0:
                    _r8a_promote, _r8a_tier = True, "acceptable (≥60, TR≥20%)"
                if _r8a_promote and verdict_sc not in ('BUY', 'STRONG BUY'):
                    verdict_sc = 'BUY'
                    conviction = 'High' if final >= 80 else 'Medium'
                    emoji      = '🟢'
                    sc_data['_rule8a_applied'] = True
                    logger.info(
                        f"[Rule8A] {target}: Score={final}, Upside={_upside_r8a:.1f}%, "
                        f"DivYield={_div_y_r8a:.2f}%, TotalReturn={_total_ret_r8a:.1f}% "
                        f"→ Fundamental=BUY [{_r8a_tier}] (was {_de_result['verdict']})"
                    )
            except Exception as _de_err:
                import logging as _de_log
                _de_log.getLogger(__name__).warning(
                    "[DecisionEngine] binding failed for %s: %s", target, _de_err
                )
                # Fallback: ADX-aware classification (no LLM trend_state)
                _de_adx = float(sc_data.get('adx') or 0)
                if verdict_sc in ("BUY", "STRONG BUY"):
                    if _de_adx >= 25:
                        decision_type = "trend_confirmed"
                    elif _de_adx >= 20:
                        decision_type = "early_reversal"
                    else:
                        decision_type = "contrarian_early"
                elif verdict_sc == "HOLD":
                    decision_type = "wait_for_confirmation"
                else:
                    decision_type = "trend_failure"
            # ── End Decision Engine ────────────────────────────────────────

            _div_info = _consensus_divergence(
                verdict_sc, analyst_consensus or '',
                adx=float(sc_data.get('adx') or 20),
                beta=float(sc_data.get('beta') or 1.0),
            )
            from core.scorecard import compute_entry_quality as _ceq
            _eq_score, _eq_label, _eq_note = _ceq(sc_data)
            _entry_quality_block = (
                f'\n**Entry Quality: {_eq_score}/100 — {_eq_label}**\n'
                f'*{_eq_note}*\n'
            )
            upside = ((sc_data['target'] - sc_data['price']) / sc_data['price'] * 100) if sc_data['target'] else 0

            # Display mapping — internal codes → institutional-grade client-facing labels
            _VERDICT_DISPLAY = {
                "REDUCE": "Positioning: Underweight",
                "SELL":   "Risk Stance: Avoid",
                "BUY":    "BUY",
                "HOLD":   "HOLD",
                "AVOID":  "Risk Stance: Avoid",
            }
            _DECISION_TYPE_LABELS = {
                "contrarian_early": "Contrarian Early",
                "early_reversal": "Early Reversal",
                "trend_confirmed": "Trend Confirmed",
                "wait_for_confirmation": "Wait For Confirmation",
                "trend_failure": "Trend Failure",
            }
            _decision_type_label = _DECISION_TYPE_LABELS.get(
                decision_type, decision_type.replace('_', ' ').title()
            )
            verdict_display = _VERDICT_DISPLAY.get(verdict_sc, verdict_sc)
            if verdict_sc == "BUY":
                verdict_display = f"Tactical BUY — {_decision_type_label}"

            # ── Entry Timing (from scorecard Rule 8A or get_verdict) ─────────
            _entry_timing = sc_data.get('entry_timing', '')
            # If entry_timing not yet set (no Rule8A path), derive from technicals
            if not _entry_timing:
                _adx_et = float(sc_data.get('adx', 0) or 0)
                _rsi_et = float(sc_data.get('rsi', 50) or 50)
                if verdict_sc in ('BUY', 'STRONG BUY'):
                    if _rsi_et > 70:
                        _entry_timing = 'WAIT — RSI overbought, await pullback'
                    elif _adx_et < 20:
                        _entry_timing = 'WAIT — trend not confirmed (ADX < 20)'
                    elif _adx_et < 25:
                        _entry_timing = 'ADD ON DIP — await ADX > 25'
                    else:
                        _entry_timing = 'BUY NOW — trend confirmed'
                elif verdict_sc in ('REDUCE', 'SELL', 'AVOID'):
                    _entry_timing = 'REDUCE INTO STRENGTH'
                else:
                    _entry_timing = 'WAIT'

            # ── English timing preserved before Arabic translation ─────────────
            _entry_timing_en = _entry_timing  # always English; needed for decision logic

            # Arabic timing labels (user-facing Quick View only; English kept in prompt)
            # _is_arabic_request lives in _handle_analytics scope — guard against NameError here
            _is_ar_sc = False
            try:
                _is_ar_sc = bool(_is_arabic_request)
            except NameError:
                pass
            if _is_ar_sc:
                _TIMING_AR = {
                    'WAIT — RSI overbought, await pullback': 'انتظر — مؤشر RSI في منطقة التشبع، انتظر تراجعًا',
                    'WAIT — trend not confirmed (ADX < 20)': 'انتظر — الاتجاه غير مؤكد (ADX أقل من 20)',
                    'ADD ON DIP — await ADX > 25': 'شراء تدريجي عند التراجع — انتظر ADX فوق 25',
                    'BUY NOW — trend confirmed': 'شراء الآن — الاتجاه مؤكد',
                    'REDUCE INTO STRENGTH': 'خفّف مع الارتفاع',
                    'WAIT': 'انتظر تأكيدًا',
                }
                _entry_timing = _TIMING_AR.get(_entry_timing, _entry_timing)

            # ── Persist decision data for _handle_analytics (no regex needed) ──
            self._last_scorecard_decision = {
                'verdict':     verdict_sc,
                'timing_en':   _entry_timing_en,   # English; used for WAIT/BUY logic
                'timing':      _entry_timing,       # Display form (may be translated)
                'score':       final,
                'conviction':  conviction,
                'emoji':       emoji,
            }

            if _is_crypto_t:
                # ── Crypto-specific scorecard display ──
                # ATH: priority chain → CoinGecko real ATH → sc_data year_high → fund year_high
                _ath = (
                    float(_onchain.get('ath') or 0)
                    or float(sc_data.get('year_high') or 0)
                    or float(fund.get('year_high') or 0)
                )
                _ath_dist = ((sc_data['price'] - _ath) / _ath * 100) if _ath and _ath > 0 else 0
                _fg = sc_data.get('fear_greed', 50)
                _fg_label = "Extreme Fear" if _fg <= 20 else "Fear" if _fg <= 40 else "Neutral" if _fg <= 60 else "Greed" if _fg <= 80 else "Extreme Greed"
                _fg_emoji = "🔴" if _fg <= 25 else "🟠" if _fg <= 45 else "🟡" if _fg <= 55 else "🟢" if _fg <= 75 else "🔴"
                _sma200 = sc_data.get('sma200', 0)
                _sma_dist = ((sc_data['price'] - _sma200) / _sma200 * 100) if _sma200 else 0
                # ATH from CoinGecko (real ATH, not just 52w high)
                _ath_date = _onchain.get('ath_date', '')
                _circ = _onchain.get('circulating_supply', 0)
                _max_s = _onchain.get('max_supply', 0)
                _supply_pct = _onchain.get('supply_ratio', 0)
                _vol_24h = _onchain.get('total_volume_24h', 0)
                _hash_eh = _onchain.get('hash_rate_eh', 0)
                _active_addr = _onchain.get('active_addresses', 0)
                _n_tx = _onchain.get('n_tx_24h', 0)
                # Market Cap Rank: hardcoded fallback for known assets
                _mc_rank_raw = _onchain.get('mc_rank', 0) or 0
                _CRYPTO_RANK_FALLBACK = {'BTC-USD': 1, 'ETH-USD': 2, 'BNB-USD': 3, 'SOL-USD': 4, 'XRP-USD': 5, 'DOGE-USD': 8, 'ADA-USD': 9, 'AVAX-USD': 10}
                _mc_rank = _mc_rank_raw if _mc_rank_raw and _mc_rank_raw > 0 else _CRYPTO_RANK_FALLBACK.get(target.upper(), None)
                # Format ATH display
                _ath_display = (f"{self._format_local_price(_ath, target)} ({_ath_dist:+.1f}%)" + (f" 📅 {_ath_date}" if _ath_date else "")) if _ath and _ath > 0 else "N/A"
                _rank_display = f"#{_mc_rank}" if _mc_rank else "N/A"

                sc_md = f"""
---

## 🎯 EisaX Crypto Score Card
**{target}** | Fundamental: **{verdict_display} {emoji}** | Timing: **{_entry_timing}** | Conviction: **{conviction}** | EisaX Score: **{final}/100** | Blended: **{sc_data.get('blended_score', final)}/100**

*Crypto-specific scoring: Network Dominance, SMA200 Valuation, ATH Recovery, On-Chain Metrics*

| Metric | Value |
|--------|-------|
| Live Price | {self._format_local_price(sc_data['price'], target)} |
| Beta (90d vs SPY) | {sc_data['beta']:.2f} {'⚠️ High Vol' if sc_data['beta'] > 2 else '🟡 Moderate' if sc_data['beta'] > 1.3 else '🟢'} |
| All-Time High | {_ath_display} |
| Price vs SMA200 | {_sma_dist:+.1f}% {'🔴 Below' if _sma_dist < -10 else '🟡 Near' if _sma_dist < 10 else '🟢 Above'} |
| Fear & Greed Index | {_fg}/100 {_fg_emoji} {_fg_label} |
| Market Cap Rank | {_rank_display} |"""
                # ── On-Chain Metrics section ──
                if _circ or _hash_eh or _active_addr:
                    sc_md += f"""

**⛓️ On-Chain Metrics:**

| Metric | Value |
|--------|-------|"""
                    if _circ and _max_s:
                        sc_md += f"\n| Supply (Circulating / Max) | {_circ:,.0f} / {_max_s:,.0f} ({_supply_pct}%) |"
                    elif _circ:
                        sc_md += f"\n| Circulating Supply | {_circ:,.0f} |"
                    if _vol_24h:
                        sc_md += f"\n| 24h Volume | ${_vol_24h/1e9:.1f}B |"
                    if _hash_eh:
                        sc_md += f"\n| Hash Rate | {_hash_eh:.0f} EH/s |"
                    if _active_addr:
                        sc_md += f"\n| Active Addresses (24h) | {_active_addr:,} |"
                    if _n_tx:
                        sc_md += f"\n| Transactions (24h) | {_n_tx:,} |"
            else:
                sc_md = f"""
---

## 🎯 EisaX Proprietary Score Card
**{target}** | Fundamental: **{verdict_display} {emoji}** | Timing: **{_entry_timing}** | Conviction: **{conviction}** | EisaX Score: **{final}/100** | Blended: **{sc_data.get('blended_score', final)}/100**

*Conviction driven by: {", ".join(filter(None, [
    # Upside — only show as positive driver when conviction is Medium/High (final >= 60)
    (f"strong upside potential (+{upside:.0f}%)" if upside > 20 else f"moderate upside (+{upside:.0f}%)" if upside > 10 else f"modest upside (+{upside:.0f}%)")
        if (upside > 5 and verdict_sc not in ("REDUCE", "SELL", "AVOID") and final >= 60) else None,
    # For low-score stocks with upside, note the conflict
    f"upside (+{upside:.0f}%) constrained by weak fundamentals/data gaps" if (upside > 15 and final < 60) else None,
    "upside limited by bearish technicals" if (verdict_sc in ("REDUCE", "SELL", "AVOID") and upside > 10) else None,
    "attractive valuation" if (factors.get("Valuation", (0,1))[0] / factors.get("Valuation", (0,1))[1]) >= 0.75 and verdict_sc not in ("REDUCE", "SELL", "AVOID") and final >= 60 else None,
    "strong quality fundamentals" if (factors.get("Quality Score", (0,1))[0] / factors.get("Quality Score", (0,1))[1]) >= 0.65 else None,
    # Technical — oversold is a caution signal for low-conviction stocks, not a driver
    "oversold — potential bounce but trend bearish" if ((sc_data.get('rsi') or 50) < 30 and final < 60) else
    "bullish technical momentum" if (factors.get("Technical Momentum", (0,1))[0] / factors.get("Technical Momentum", (0,1))[1]) >= 0.75 else None,
    "low risk profile" if (factors.get("Risk Profile", (0,1))[0] / factors.get("Risk Profile", (0,1))[1]) >= 0.80 and verdict_sc not in ("REDUCE", "SELL", "AVOID") else None,
    "strong market position" if (factors.get("Market Position", (0,1))[0] / factors.get("Market Position", (0,1))[1]) >= 0.75 and verdict_sc not in ("REDUCE", "SELL", "AVOID") else None,
    "fundamental data gaps limit conviction" if (sc_data.get('quality') is None or (sc_data.get('net_margin') is None and sc_data.get('roe') is None)) and final < 60 else None,
    "limited upside vs risk" if upside <= 2 and final < 60 else None,
    "bearish primary trend" if summary.get('trend') == 'Bearish' else None,
    "price below SMA200" if (sc_data.get('sma200') and sc_data['price'] < sc_data['sma200'] * 0.95) else None,
    "no analyst coverage — EisaX FV estimate only" if sc_data.get('target_is_estimate') and final >= 60 else
    "no analyst coverage + data gaps — speculative" if sc_data.get('target_is_estimate') and final < 60 else None,
])) or "balanced risk-reward profile"}*

| Metric | Value |
|--------|-------|
| Live Price | {self._format_local_price(sc_data['price'], target)} |
| Price Target | {(self._format_local_price(sc_data['target'], target) + f" (+{upside:.1f}%)" + (" *[SMA Tech.]*" if sc_data.get('target_is_sma') else " *[EisaX FV Est.]*" if sc_data.get('target_is_estimate') else (f" *[Sell-side consensus — {sc_data['analyst_count']} analysts]*" if sc_data.get('analyst_count') and sc_data['analyst_count'] > 0 else " *[Analyst-derived]*"))) if sc_data['target'] else "N/A"} |
| Beta | {f"{sc_data['beta']:.2f}" if sc_data.get('beta') else "N/A"} {'⚠️ High Risk' if (sc_data.get('beta') or 0) > 2 else ''} |
| Forward P/E | {f"{sc_data['forward_pe']:.1f}x" if sc_data.get('forward_pe') else 'N/A'} {'🟢 Reasonable' if 0 < (sc_data.get('forward_pe') or 0) < 30 else '🟡 High' if (sc_data.get('forward_pe') or 0) >= 30 else ''} |"""

            sc_md += """

**Factor Analysis:**

| Factor | Score | Bar |
|--------|-------|-----|"""
            _tm_note = ""
            for fname, (val, max_v) in factors.items():
                if fname == "Risk Profile":
                    # ── Display as RISK LEVEL % (higher = riskier, more intuitive) ──
                    _risk_raw = 100 - int((val / max_v) * 100)
                    # Priority: method param → sc_data → fund beta → 1.0 fallback
                    _eff_beta_rp = float(
                        effective_beta if (effective_beta and float(effective_beta) > 0) else
                        sc_data.get('beta') if (sc_data.get('beta') and float(sc_data.get('beta')) > 0) else
                        fund.get('beta') if (fund and fund.get('beta') and float(fund.get('beta')) > 0) else
                        1.0
                    )
                    _beta_floor = (
                        65 if _eff_beta_rp > 2.0 else
                        50 if _eff_beta_rp > 1.5 else
                        35 if _eff_beta_rp > 1.0 else
                        20 if _eff_beta_rp > 0.5 else 0
                    )
                    # Crypto annual-vol floor (high vol even with low market beta)
                    _ann_vol_rp = float(sc_data.get('annual_vol', 0) or 0)
                    if _ann_vol_rp > 0.60:
                        _beta_floor = max(_beta_floor, 50)
                    pct = max(_risk_raw, _beta_floor)
                    filled = int(pct / 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    f_emoji = "🔴" if pct >= 65 else ("🟡" if pct >= 35 else "🟢")
                    sc_md += f"\n| {fname} | {pct}% Risk | {f_emoji} `{bar}` |"
                else:
                    pct = int((val / max_v) * 100)
                    filled = int((val / max_v) * 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    f_emoji = "🟢" if pct >= 75 else ("🟡" if pct >= 50 else "🔴")
                    sc_md += f"\n| {fname} | {pct}% | {f_emoji} `{bar}` |"
                    if (
                        fname == "Technical Momentum"
                        and pct <= 0
                        and str(verdict_sc or "").upper() not in ("SELL", "AVOID")
                    ):
                        _tm_note = (
                            "\n\n*0% reflects current bearish price trend — not a fundamental deficiency. "
                            "Reversion toward SMA50 would recover this component.*"
                        )
            if _tm_note:
                sc_md += _tm_note

            # ── Pillar Breakdown — scoring methodology transparency ────────────
            # Groups factors into 3 economic pillars so every point is traceable.
            if not _is_crypto_t:
                _PILLAR_MAP = {
                    "🏦 Fundamentals":      {
                        "keys": ["Quality Score", "Valuation", "Market Position"],
                        "desc": "Quality • Valuation • Market Position",
                    },
                    "📈 Technical & Risk":  {
                        "keys": ["Price Upside", "Risk Profile", "Technical Momentum"],
                        "desc": "Upside Potential • Risk Profile • Momentum",
                    },
                    "💬 Analyst Sentiment": {
                        "keys": ["Analyst Sentiment"],
                        "desc": "Wall St Consensus",
                    },
                }
            else:
                _PILLAR_MAP = {
                    "🌐 Network & Dominance": {
                        "keys": ["Quality Score", "Network Dominance"],
                        "desc": "Market Cap Tier • Network Dominance",
                    },
                    "📈 Price & Technical":   {
                        "keys": ["Valuation", "ATH Recovery Potential", "Technical Momentum"],
                        "desc": "Price vs SMA • ATH Recovery • Momentum",
                    },
                    "⚡ Risk & Sentiment":    {
                        "keys": ["Risk Profile", "Analyst Sentiment"],
                        "desc": "Volatility Risk • Fear & Greed",
                    },
                }

            _pillar_rows = ""
            _ptotal_max  = 0
            _ptotal_earn = 0
            _analyst_na  = result[2].get('_analyst_na', False) if result else False
            for _pname, _pinfo in _PILLAR_MAP.items():
                _p_earned = sum(factors[k][0] for k in _pinfo["keys"] if k in factors)
                _p_max    = sum(factors[k][1] for k in _pinfo["keys"] if k in factors)
                if _p_max == 0:
                    continue
                _ptotal_earn += _p_earned
                _ptotal_max  += _p_max
                # Special label for Analyst Sentiment when N/A
                _is_analyst_pillar = ("Analyst Sentiment" in _pinfo["keys"])
                if _is_analyst_pillar and _analyst_na:
                    _pillar_rows += (
                        f"\n| {_pname} | No analyst coverage | {_p_max} | 0 | "
                        f"⚪ `░░░░░░░░░░` *(N/A)* |"
                    )
                else:
                    _p_pct  = int(_p_earned / _p_max * 100)
                    _p_bar  = "█" * int(_p_pct / 10) + "░" * (10 - int(_p_pct / 10))
                    _p_icon = "🟢" if _p_pct >= 70 else ("🟡" if _p_pct >= 50 else "🔴")
                    _pillar_rows += (
                        f"\n| {_pname} | {_pinfo['desc']} | {_p_max} | {_p_earned} | "
                        f"{_p_icon} `{_p_bar}` |"
                    )

            # No rescaling note needed — max is always 100 now
            _rescale_note = ""
            sc_md += f"""

**📊 Score Breakdown:**

| Pillar | Factors | Max | Earned | Score |
|--------|---------|-----|--------|-------|{_pillar_rows}
| **TOTAL** | *(all pillars)* | **{_ptotal_max}** | **{_ptotal_earn}** | **{f"Raw: {sc_data['_raw_score']} → Capped: {final} (below SMA200)" if sc_data.get('_score_capped') and sc_data.get('_raw_score') and sc_data['_raw_score'] != final else f"{final}/100"}** |

> *Score is 100% deterministic — computed from live market data using explicit mathematical thresholds. No LLM estimation. Every point is traceable to a specific data input.*"""

            filled_big = int((final / 100) * 20)
            big_bar = "█" * filled_big + "░" * (20 - filled_big)
            # Show cap explanation if score was reduced
            _cap_note = ""
            if sc_data.get('_score_capped') and sc_data.get('_raw_score') and sc_data['_raw_score'] != final:
                _raw = sc_data['_raw_score']
                _cap_note = f"\n\n> ⚠️ **Technical Override:** Raw score was **{_raw}/100** → capped to **{final}/100** because price is below SMA200. Upgrade to BUY requires reclaiming SMA200 (${sc_data.get('sma200', 0):,.2f})."
            sc_md += f"""

**Overall: `{big_bar}` {final}/100**

> *EisaX Proprietary Score | Abu Dhabi*{_cap_note}
---"""
            return sc_md
        except Exception as e:
            logger.error(f"[Scorecard] failed: {e}")
            return ""

    def _build_factcheck_block(self, real_price, fund, summary, dc_data, forward_pe,
                               next_earnings=None, fg_data=None, ticker="", effective_beta=None):
        """Build the fact-check verification block with earnings urgency flag + Fear & Greed."""
        try:
            from datetime import datetime as _dt2
            _today   = _dt2.now().strftime("%b %d, %Y")
            _is_sar  = str(ticker).upper().endswith(".SR")
            _is_aed  = str(ticker).upper().endswith((".AE", ".DU"))
            _is_egp  = str(ticker).upper().endswith(".CA")
            _is_kwf  = str(ticker).upper().endswith(".KW")
            _is_qar  = str(ticker).upper().endswith(".QA")
            _sym     = ("﷼" if _is_sar else "د.إ" if _is_aed else "ج.م" if _is_egp else
                        "ف" if _is_kwf else "ر.ق" if _is_qar else "$")
            _is_local_price = _is_sar or _is_aed or _is_egp or _is_kwf or _is_qar
            _fp2 = real_price or summary.get('price', 0)
            _live_price = (f"{_fp2:,.2f} {_sym}" if _is_local_price and _fp2
                          else f"${_fp2:,.2f}" if _fp2 else "N/A")

            # ── Beta: single source of truth = effective_beta (pre-validated,
            #    same value as scorecard). Raw yfinance is not used directly
            #    because it can return garbage like -0.01 for GCC stocks.
            _beta_eff_fc = float(effective_beta) if effective_beta else 0.0
            if _beta_eff_fc < 0:
                _beta_live = "Not reliable"       # negative beta is garbage data
            elif _beta_eff_fc > 5:
                _beta_live = "Not reliable"       # absurdly high — don't display
            elif _beta_eff_fc > 0:
                _is_crypto_fc = str(ticker).upper().endswith('-USD')
                _beta_note = " *(rolling)*" if _is_crypto_fc else ""
                _beta_live = f"{_beta_eff_fc:.2f}{_beta_note}"
            else:
                # effective_beta is 0 (unavailable) — try dc_data only (StockAnalysis)
                _beta_dc_fc = float(dc_data.get('beta') or 0)
                if 0 < _beta_dc_fc <= 5:
                    _beta_live = f"{_beta_dc_fc:.2f}"
                elif _beta_dc_fc < 0 or _beta_dc_fc > 5:
                    _beta_live = "Not reliable"
                else:
                    _beta_live = 'N/A'
            # ── P/E sanity: values ≤ 0 or > 200 are not meaningful to display ──
            _pe_raw    = fund.get('pe_ratio') or (
                float(dc_data.get('pe_ratio', 0)) if dc_data.get('pe_ratio') else 0)
            _pe_float  = float(_pe_raw) if _pe_raw else 0.0
            _pe_live   = ("Not reliable" if _pe_float > 200
                          else f"{_pe_float:.1f}x" if _pe_float > 0
                          else 'N/A')
            _fpe_raw   = float(dc_data.get('forward_pe') or forward_pe or 0)
            _fpe_live  = ("Not reliable" if _fpe_raw > 200
                          else f"{_fpe_raw:.1f}x" if _fpe_raw > 0
                          else 'N/A')
            # Crypto: inject market cap from CoinGecko
            if not dc_data.get('market_cap') and str(ticker).upper().endswith(('-USD', '-USDT')):
                try:
                    import requests as _rq2
                    _cg_map = {'BTC-USD':'bitcoin','ETH-USD':'ethereum','SOL-USD':'solana',
                               'XRP-USD':'ripple','BNB-USD':'binancecoin','DOGE-USD':'dogecoin'}
                    _cg_id = _cg_map.get(ticker.upper())
                    if _cg_id:
                        _r = _rq2.get(f'https://api.coingecko.com/api/v3/simple/price'
                                      f'?ids={_cg_id}&vs_currencies=usd&include_market_cap=true',
                                      timeout=5)
                        if _r.status_code == 200:
                            _mc = _r.json().get(_cg_id, {}).get('usd_market_cap', 0)
                            if _mc > 0:
                                dc_data['market_cap'] = (f"${_mc/1e12:.2f}T" if _mc >= 1e12
                                                         else f"${_mc/1e9:.0f}B")
                except Exception:
                    pass
            # Stocks: جيب market cap و 52W من yfinance لو مش موجود
            if not dc_data.get('market_cap') or not dc_data.get('week_52_range'):
                try:
                    import yfinance as _yfc3
                    _fi = _yfc3.Ticker(ticker).fast_info
                    if not dc_data.get('market_cap'):
                        _mc = getattr(_fi, 'market_cap', None)
                        if _mc and _mc > 0:
                            # Local currency display for regional stocks
                            _is_local_mc = str(ticker).upper().endswith(('.AE', '.DU', '.AD', '.SR', '.KW', '.CA'))
                            if str(ticker).upper().endswith(('.AE', '.DU', '.AD')):
                                _mc_sym, _mc_sfx = 'AED ', ''
                            elif str(ticker).upper().endswith('.SR'):
                                _mc_sym, _mc_sfx = 'SAR ', ''
                            elif str(ticker).upper().endswith('.KW'):
                                _mc_sym, _mc_sfx = 'KWD ', ''
                            elif str(ticker).upper().endswith('.CA'):
                                _mc_sym, _mc_sfx = 'EGP ', ''
                            else:
                                _mc_sym, _mc_sfx = '$', ''
                            dc_data['market_cap'] = (f"{_mc_sym}{_mc/1e12:.2f}T" if _mc >= 1e12
                                                     else f"{_mc_sym}{_mc/1e9:.2f}B" if _mc >= 1e9
                                                     else f"{_mc_sym}{_mc/1e6:.0f}M")
                    if not dc_data.get('week_52_range'):
                        _lo = getattr(_fi, 'year_low', None)
                        _hi = getattr(_fi, 'year_high', None)
                        if _lo and _hi:
                            # Use local currency symbol for regional stocks, $ for US
                            _52w_sym = (_sym if (_is_sar or _is_aed or _is_egp) else '$')
                            dc_data['week_52_range'] = f"{_52w_sym}{_lo:,.2f} - {_52w_sym}{_hi:,.2f}"
                except Exception:
                    pass
            # ── DB fallback for Market Cap + 52W (covers UAE/Saudi/Egypt) ──────
            if not dc_data.get('market_cap') or not dc_data.get('week_52_range'):
                try:
                    import sqlite3 as _sq3
                    from core.config import CORE_DB as _cfg_core_db
                    _dbc = _sq3.connect(str(_cfg_core_db))
                    _dbr = _dbc.execute(
                        "SELECT market_cap, week_52_high, week_52_low FROM uae_fundamentals WHERE ticker=? LIMIT 1",
                        (str(ticker).upper(),)
                    ).fetchone()
                    _dbc.close()
                    if _dbr:
                        _db_mc, _db_hi, _db_lo = _dbr
                        if not dc_data.get('market_cap') and _db_mc and _db_mc > 0:
                            _sym_mc = ('AED ' if str(ticker).upper().endswith(('.AE','.DU'))
                                       else 'SAR ' if str(ticker).upper().endswith('.SR')
                                       else 'EGP ' if str(ticker).upper().endswith('.CA')
                                       else '$')
                            dc_data['market_cap'] = (f"{_sym_mc}{_db_mc/1e12:.2f}T" if _db_mc >= 1e12
                                                     else f"{_sym_mc}{_db_mc/1e9:.2f}B" if _db_mc >= 1e9
                                                     else f"{_sym_mc}{_db_mc/1e6:.0f}M")
                        if not dc_data.get('week_52_range') and _db_hi and _db_lo:
                            _52w_sym_db = (_sym if (_is_sar or _is_aed or _is_egp) else '$')
                            dc_data['week_52_range'] = f"{_52w_sym_db}{_db_lo:,.2f} – {_52w_sym_db}{_db_hi:,.2f}"
                except Exception:
                    pass
            _mc_live   = dc_data.get('market_cap') or 'N/A'
            _range_live = dc_data.get('week_52_range') or 'N/A'

            # LOCAL PRICE row (SAR / AED / EGP)
            _local_lbl = "SAR" if _is_sar else ("AED" if _is_aed else ("EGP" if _is_egp else ""))
            _local_row = (f"| Local Price ({_local_lbl}) | — | {real_price:,.2f} {_sym} | ➕ |\n"
                          if (_is_sar or _is_aed or _is_egp) and real_price else "")

            # ── Earnings date: skip past dates, label recent ones ──
            _earnings_raw  = next_earnings or dc_data.get('earnings_date') or fund.get('next_earnings_date') or 'N/A'
            _earnings_live = _earnings_raw
            _earnings_flag = ""
            try:
                from datetime import datetime as _dt3
                for _fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
                    try:
                        _earn_dt  = _dt3.strptime(str(_earnings_raw).split("T")[0].strip(), _fmt)
                        _days_to  = (_earn_dt - _dt3.now()).days
                        if _days_to < 0:
                            # Earnings already happened — label it as "recently reported"
                            _earnings_live = f"{_earnings_raw} *(recently reported — awaiting next date)*"
                        elif 0 <= _days_to <= 3:
                            _earnings_flag = f"\n\n> ⚠️ **URGENT CATALYST:** Earnings in **{_days_to} day(s)** ({_earnings_raw}). High volatility expected."
                        elif _days_to <= 14:
                            _earnings_flag = f"\n\n> 📅 **NEAR-TERM CATALYST:** Earnings in {_days_to} days ({_earnings_raw})."
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

            # Fear & Greed row
            _fg = fg_data or {}
            _fg_score  = _fg.get('score')
            _fg_rating = _fg.get('rating', '')
            _fg_label  = _fg.get('label_ar', '')
            _fg_emoji  = (
                "🔴" if _fg_score is not None and _fg_score < 25 else
                "🟠" if _fg_score is not None and _fg_score < 45 else
                "🟡" if _fg_score is not None and _fg_score < 55 else
                "🟢" if _fg_score is not None and _fg_score < 75 else
                "💹" if _fg_score is not None else "—"
            )
            _fg_row = (f"| Fear & Greed | — | {_fg_emoji} {int(_fg_score)} — {_fg_rating} | ➕ |\n"
                       if _fg_score is not None else "")

            return f"""\n\n---
🔍 **FACT-CHECK** *(Verified {_today})*

| Metric | Report | Live | Status |
|--------|--------|------|--------|
| Price | {_live_price} | {_live_price} | ✅ |
{_local_row}| Beta | — | {_beta_live} | ➕ |
| P/E (TTM) | — | {_pe_live} | ➕ |
| Forward P/E | — | {_fpe_live} | ➕ |
| Market Cap | — | {_mc_live} | ➕ |
| 52W Range | — | {_range_live} | ➕ |
{_fg_row}📅 **Next Earnings:** {_earnings_live}{_earnings_flag}

*Source: Yahoo Finance + StockAnalysis + CNN Fear&Greed — live at time of query*"""
        except Exception as e:
            logger.error(f"[FactCheck] build failed: {e}")
            return ""

    def _precompute_report_data(
        real_price, forward_pe, analyst_target, fund, summary, dc_data,
        currency_sym="$",
        is_crypto: bool = False,
        is_etf: bool = False,
    ) -> dict:
        """
        Pre-compute ALL numerical values for the report.
        LLM receives finished numbers — never computes anything itself.
        N/A only appears when source data is genuinely absent.
        """
        def _to_float(v) -> float:
            try:
                if v in (None, "", "N/A", "None"):
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        _ = currency_sym  # reserved for future currency-specific formatting
        p = _to_float(real_price)
        fpe = _to_float(forward_pe or (dc_data or {}).get("forward_pe"))
        eps_ttm = _to_float((fund or {}).get("eps") or (dc_data or {}).get("eps"))
        at = _to_float(analyst_target)
        sma50 = _to_float((summary or {}).get("sma_50"))
        sma200 = _to_float((summary or {}).get("sma_200"))
        beta = _to_float((dc_data or {}).get("beta") or (fund or {}).get("beta") or 1.0)

        out = {"beta": beta}

        # ── Forward EPS ──────────────────────────────────────────────────────
        if fpe > 0 and p > 0:
            out["forward_eps"] = round(p / fpe, 2)
            out["forward_eps_source"] = "price/fwd_pe"
        elif eps_ttm > 0:
            out["forward_eps"] = round(eps_ttm, 2)
            out["forward_eps_source"] = "ttm_eps_approx"
        else:
            out["forward_eps"] = None
            out["forward_eps_source"] = "unavailable"

        # ── Valuation Scenarios (Bear/Base/Bull) ─────────────────────────────
        fwd_eps = out["forward_eps"]
        if fwd_eps and fpe > 0:
            out["val_bear_pe"] = round(fpe * 0.70, 1)
            out["val_base_pe"] = round(fpe, 1)
            out["val_bull_pe"] = round(fpe * 1.40, 1)
            out["val_bear_price"] = round(fpe * 0.70 * fwd_eps, 0)
            out["val_base_price"] = round(fpe * fwd_eps, 0)
            out["val_bull_price"] = round(fpe * 1.40 * fwd_eps, 0)
            out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1) if p else None
            out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1) if p else None
            out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1) if p else None
        else:
            out["val_bear_price"] = out["val_base_price"] = out["val_bull_price"] = None
            out["val_bear_pe"] = out["val_base_pe"] = out["val_bull_pe"] = None
            out["val_bear_updown"] = out["val_base_updown"] = out["val_bull_updown"] = None

            # price-based fallback for assets with no P/E
            if is_crypto and p > 0:
                _sma200 = _to_float((summary or {}).get("sma_200"))
                _52wk_high = _to_float((dc_data or {}).get("52wk_high") or (dc_data or {}).get("fiftyTwoWeekHigh"))
                out["val_bear_price"] = round(p * 0.60, 0)           # -40% bear cycle
                out["val_base_price"] = round(_sma200, 0) if _sma200 > p * 0.50 else round(p * 1.20, 0)
                out["val_bull_price"] = round(_52wk_high, 0) if _52wk_high > p else round(p * 1.70, 0)
                out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1)
                out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1)
                out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1)
                out["val_bear_pe"] = "Bear cycle"
                out["val_base_pe"] = "SMA200 reversion"
                out["val_bull_pe"] = "ATH retest"
            elif is_etf and p > 0:
                _sma200 = _to_float((summary or {}).get("sma_200"))
                out["val_bear_price"] = round(p * 0.80, 0)
                out["val_base_price"] = round(_sma200, 0) if _sma200 > p * 0.70 else round(p * 1.08, 0)
                out["val_bull_price"] = round(p * 1.20, 0)
                out["val_bear_updown"] = round((out["val_bear_price"] - p) / p * 100, 1)
                out["val_base_updown"] = round((out["val_base_price"] - p) / p * 100, 1)
                out["val_bull_updown"] = round((out["val_bull_price"] - p) / p * 100, 1)
                out["val_bear_pe"] = "Stress scenario"
                out["val_base_pe"] = "SMA200 reversion"
                out["val_bull_pe"] = "Breakout scenario"

        # ── Upside to Target ─────────────────────────────────────────────────
        out["upside_to_target"] = round((at - p) / p * 100, 1) if at and p else None

        # ── Price vs SMAs ────────────────────────────────────────────────────
        out["pct_vs_sma50"] = round((p - sma50) / sma50 * 100, 1) if sma50 else None
        out["pct_vs_sma200"] = round((p - sma200) / sma200 * 100, 1) if sma200 else None

        # ── Entry Zone / Pullback Distance ───────────────────────────────────
        entry = sma50 if sma50 and p > sma50 * 1.02 else (sma200 if sma200 else None)
        if entry and p:
            out["entry_zone"] = round(entry, 2)
            out["pct_above_entry"] = round((p - entry) / p * 100, 1)
        else:
            out["entry_zone"] = None
            out["pct_above_entry"] = None

        # Technical S/R Ladder (S1/S2/S3 and R1/R2/R3)
        # Priority order: nearest SMA -> nearest Fibonacci -> recent swing -> 52W boundary.
        h52 = _to_float(
            (fund or {}).get("week52_high") or
            (fund or {}).get("year_high") or
            (dc_data or {}).get("fiftyTwoWeekHigh") or
            (dc_data or {}).get("52wk_high") or 0
        )
        l52 = _to_float(
            (fund or {}).get("week52_low") or
            (fund or {}).get("year_low") or
            (dc_data or {}).get("fiftyTwoWeekLow") or
            (dc_data or {}).get("52wk_low") or 0
        )

        _fib_levels = {}
        if h52 and l52 and h52 > l52:
            _rng = h52 - l52
            _fib_levels = {
                "23.6%": round(l52 + _rng * 0.236, 3),
                "38.2%": round(l52 + _rng * 0.382, 3),
                "50.0%": round((h52 + l52) / 2, 3),
                "61.8%": round(l52 + _rng * 0.618, 3),
                "78.6%": round(l52 + _rng * 0.786, 3),
            }

        _nearest_sma_label, _nearest_sma_val = None, None
        _sma_candidates = []
        if sma50:
            _sma_candidates.append(("SMA50", round(sma50, 3)))
        if sma200:
            _sma_candidates.append(("SMA200", round(sma200, 3)))
        if p > 0 and _sma_candidates:
            _nearest_sma_label, _nearest_sma_val = min(_sma_candidates, key=lambda x: abs(x[1] - p))

        _nearest_fib_label, _nearest_fib_val = None, None
        if p > 0 and _fib_levels:
            _fib_candidates = [(k, v) for k, v in _fib_levels.items() if abs(v - p) / max(p, 1.0) > 0.001]
            if _fib_candidates:
                _nearest_fib_label, _nearest_fib_val = min(_fib_candidates, key=lambda x: abs(x[1] - p))

        _swing_high = _to_float(
            (summary or {}).get("swing_high") or
            (summary or {}).get("recent_swing_high") or
            (summary or {}).get("high_20d") or
            (dc_data or {}).get("swing_high") or
            (dc_data or {}).get("recent_swing_high") or 0
        )
        _swing_low = _to_float(
            (summary or {}).get("swing_low") or
            (summary or {}).get("recent_swing_low") or
            (summary or {}).get("low_20d") or
            (dc_data or {}).get("swing_low") or
            (dc_data or {}).get("recent_swing_low") or 0
        )

        _levels = []

        def _push_level(price_v: float, level_type: str, basis: str, priority: int):
            if not (p > 0 and price_v):
                return
            _dist = abs(price_v - p) / p
            if _dist <= 0.0005:
                return
            _levels.append({
                "price": round(price_v, 3),
                "type": level_type,
                "basis": basis,
                "priority": priority,
                "distance": _dist,
            })

        if _nearest_sma_val:
            _push_level(_nearest_sma_val, "Resistance" if _nearest_sma_val > p else "Support", _nearest_sma_label, 1)

        if _nearest_fib_val:
            _push_level(_nearest_fib_val, "Resistance" if _nearest_fib_val > p else "Support", f"Fib {_nearest_fib_label}", 2)

        if _swing_high:
            _push_level(_swing_high, "Resistance" if _swing_high > p else "Support", "Recent Swing High", 3)
        if _swing_low:
            _push_level(_swing_low, "Resistance" if _swing_low > p else "Support", "Recent Swing Low", 3)

        if h52:
            _push_level(h52, "Resistance" if h52 > p else "Support", "52W High", 4)
        if l52:
            _push_level(l52, "Support" if l52 < p else "Resistance", "52W Low", 4)

        _dedup = {}
        for _lv in _levels:
            _k = (_lv["type"], round(_lv["price"], 3))
            if _k not in _dedup:
                _dedup[_k] = _lv
            else:
                _cur = _dedup[_k]
                if (_lv["priority"], _lv["distance"]) < (_cur["priority"], _cur["distance"]):
                    _dedup[_k] = _lv
        _levels = list(_dedup.values())

        _above = sorted([x for x in _levels if x["price"] > p], key=lambda x: (x["distance"], x["priority"], x["price"]))[:3]
        _below = sorted([x for x in _levels if x["price"] < p], key=lambda x: (x["distance"], x["priority"], -x["price"]))[:3]

        for i, _lv in enumerate(_above, 1):
            _lv["level"] = f"R{i}"
        for i, _lv in enumerate(_below, 1):
            _lv["level"] = f"S{i}"

        out["sr_levels_above"] = _above
        out["sr_levels_below"] = _below

        out["fib_resistance"] = round(_above[0]["price"], 3) if _above else None
        out["fib_resistance_pct"] = round((out["fib_resistance"] - p) / p * 100, 1) if out["fib_resistance"] else None
        out["fib_resistance_label"] = _above[0]["basis"] if _above else None
        out["fib_support"] = round(_below[0]["price"], 3) if _below else None
        out["fib_support_pct"] = round((out["fib_support"] - p) / p * 100, 1) if out["fib_support"] else None
        out["fib_key_support"] = round(l52, 3) if l52 else None
        out["fib_52w_high"] = round(h52, 3) if h52 else None
        out["fib_52w_low"] = round(l52, 3) if l52 else None
        out["fib_above_52w_high"] = bool(h52 and p > h52)

        _sr_rows = [
            "| Level | Price | Type | Basis |",
            "|-------|-------|------|-------|",
        ]
        for _lv in reversed(_above):
            _sr_rows.append(f"| {_lv['level']} | {currency_sym}{_lv['price']:,.2f} | {_lv['type']} | {_lv['basis']} |")
        _sr_rows.append(f"| Spot | {currency_sym}{p:,.2f} | Current | Live |" if p else "| Spot | N/A | Current | Live |")
        for _lv in _below:
            _sr_rows.append(f"| {_lv['level']} | {currency_sym}{_lv['price']:,.2f} | {_lv['type']} | {_lv['basis']} |")
        out["sr_levels_table"] = "\n".join(_sr_rows)

        return out


