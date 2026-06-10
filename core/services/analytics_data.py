"""
core/services/analytics_builder.py
────────────────────────────────────
Complex logic extracted from FinanceAgent._handle_analytics.

Public API
──────────
    enrich_after_fetch(target, fr) -> dict
        Derives analyst targets, beta, energy/crypto flags, fair value, etc.

    collect_news_waterfall(target, fr, dc_data, fund) -> tuple[list, str, float]
        Full news collection with 8+ fallback levels.

    build_data_block(target, fr, ctx, original_target=None) -> str
        Builds the structured text block for the LLM.

    build_analytics_prompt(target, data_block, ctx, scorecard_verdict_hint,
                           is_arabic, brain_ctx, local_injection,
                           research_summary, original_target=None,
                           macro_block="") -> str
        Builds the full DeepSeek prompt.

    assemble_report(target, fr, ctx, deepseek_reply, news_block, pos,
                    pre_scorecard_md, original_target=None) -> str
        Assembles the final markdown report.
"""

from __future__ import annotations

import logging
import math
import os
import re as _re

from core.services.data_fetcher import FetchResult

logger = logging.getLogger(__name__)


# ── A. enrich_after_fetch ─────────────────────────────────────────────────────

def build_data_block(
    target: str,
    fr: FetchResult,
    ctx: dict,
    original_target: str | None = None,
) -> str:
    """
    Build the structured text block passed to the LLM.
    Extracted from _handle_analytics lines 3199–3411.
    """
    fund          = fr.fund or {}
    dc_data       = fr.dc_data or {}
    summary       = fr.summary or {}
    var_95        = fr.var_95 or 0.02
    max_dd        = fr.max_dd or 0.20
    ev_out        = fr.ev_out or {}
    fg_data       = fr.fg_data or {}
    real_price    = fr.real_price
    change_pct    = fr.change_pct or 0.0
    next_earnings = fr.next_earnings

    analyst_target    = ctx.get("analyst_target")
    analyst_consensus = ctx.get("analyst_consensus")
    analyst_count     = ctx.get("analyst_count")
    forward_pe        = ctx.get("forward_pe")
    dividend_yield    = ctx.get("dividend_yield")
    effective_beta    = ctx.get("effective_beta", 1.0)
    is_energy         = ctx.get("is_energy", False)
    oil_data          = ctx.get("oil_data", {})
    onchain_data      = ctx.get("onchain_data", {})
    fv_label          = ctx.get("fv_label", "Analyst consensus")
    valuation_pe      = ctx.get("valuation_pe", 15)
    display_target    = ctx.get("display_target")
    target_is_estimate = ctx.get("target_is_estimate", True)
    etf_meta          = ctx.get("etf_meta")

    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")
    _is_local_currency = currency_lbl in ("SAR", "AED", "EGP", "KWF", "QAR")
    _is_local_mkt = currency_lbl != "USD"
    _t_upper = target.upper()

    news_sent  = ctx.get("news_sent", "N/A")
    news_score = ctx.get("news_score", 0.0)
    t10y       = getattr(fr, "t10y", "N/A")
    fed        = getattr(fr, "fed", "N/A")
    unemp      = getattr(fr, "unemp", "N/A")
    inflation  = getattr(fr, "inflation", "N/A")
    gdp        = getattr(fr, "gdp", "N/A")

    _fallback_price = real_price or summary.get("price", 0)
    price_str = (
        f"{_fallback_price:,.2f} {currency_sym} ({change_pct:+.2f}%)"
        if _fallback_price and _is_local_mkt and change_pct
        else f"{_fallback_price:,.2f} {currency_sym}"
        if _fallback_price and _is_local_mkt
        else f"${_fallback_price:,.2f} ({change_pct:+.2f}%)"
        if _fallback_price and change_pct
        else f"${_fallback_price:,.2f}"
        if _fallback_price else "N/A"
    )

    # ── Local format helpers ──────────────────────────────────────────────────
    def _B(n):
        try:
            if not n: return "N/A"
            v = float(n)
            if currency_lbl != "USD":
                if v >= 1e12: return f"{v/1e12:.2f}T {currency_sym}"
                if v >= 1e9:  return f"{v/1e9:.1f}B {currency_sym}"
                if v >= 1e6:  return f"{v/1e6:.0f}M {currency_sym}"
                return f"{v:,.0f} {currency_sym}"
            return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
        except Exception:
            logger.debug("[_B] format error for value %r", n, exc_info=True)
            return "N/A"

    def _P(n):
        return f"{n:.1f}%" if n else "N/A"

    def _X(n):
        return f"{n:.1f}x" if n else "N/A"

    # Engine news data for inline block
    _engine_news_data = fr.engine_news or {}

    # ADX classification — single source of truth
    from core.services.scorecard_engine import classify_adx as _classify_adx
    _adx_val = summary.get('adx', 0)
    _adx_short, _adx_desc = _classify_adx(_adx_val)

    # ── Component 5: Locked canonical values header ───────────────────────────
    # These six values are injected FIRST so the LLM sees them as the single
    # source of truth before any other section. The DATA LOCK preamble in the
    # prompt already forbids training-data recall; this block reinforces it by
    # providing the exact values the LLM must use — no recomputation allowed.
    _fwd_pe_str = (
        f"{float(forward_pe):.1f}x" if forward_pe else "not available"
    )
    _div_str = (
        f"{float(dividend_yield)*100:.2f}%" if dividend_yield else "not available"
    )
    _52h = fund.get("week52_high")
    _52l = fund.get("week52_low")
    _52w_str = (
        f"{_52l:,.2f} – {_52h:,.2f} {currency_sym}"
        if _52h and _52l else "not available"
    )
    _pe_str = f"{float(fund.get('pe') or 0):.1f}x" if fund.get('pe') else "not available"

    data_block = f"""
╔══════════════════════════════════════════════════════════════╗
║  [DATA BLOCK — LOCKED]  Use ONLY these values. No training. ║
║  Price      : {price_str:<47}║
║  Forward PE : {_fwd_pe_str:<47}║
║  Div Yield  : {_div_str:<47}║
║  52W Range  : {_52w_str:<47}║
║  Trailing PE: {_pe_str:<47}║
║  Market Cap : {_B(fund.get('market_cap')):<47}║
╚══════════════════════════════════════════════════════════════╝

TICKER: {original_target if original_target else target} (resolved: {target})
COMPANY: {fund.get('company_name') or (original_target if original_target else target)}
SECTOR: {fund.get('sector', 'N/A')} | INDUSTRY: {fund.get('industry', 'N/A')}
CURRENCY: {currency_lbl} (use {currency_sym} symbol in ALL price references){chr(10) + "IMPORTANT: This is an Egyptian stock (EGX). Market Cap, prices and all monetary values are in EGP (Egyptian Pound ج.م). Do NOT convert to USD or display in USD." if _t_upper.endswith(".CA") else ""}
LIVE PRICE: {price_str}
MARKET CAP: {_B(fund.get('market_cap'))}
QUALITY SCORE: {fund.get('fundamental_score', 'N/A')}/100

NEWS SENTIMENT: {news_sent} (score: {news_score})

MACRO: 10Y Treasury: {t10y}% | Fed Funds: {fed}% | Unemployment: {unemp}% | CPI YoY: {inflation}% | GDP Growth: {gdp}%

GROWTH:
- Revenue Growth YoY: {_P(fund.get('revenue_growth'))}
- EPS Growth YoY: {_P(fund.get('eps_growth'))}
- Revenue (TTM): {_B(fund.get('revenue'))}
- EPS (TTM): ${fund.get('eps', 'N/A')}

PROFITABILITY:
- Gross Margin: {_P(fund.get('gross_margin'))}
- Operating Margin: {_P(fund.get('operating_margin'))}
- Net Margin: {_P(fund.get('net_margin'))}
- ROE: {_P(fund.get('roe'))}
- ROIC: {_P(fund.get('roic'))}

VALUATION:
- P/E (TTM): {_X(fund.get('pe_ratio'))}
- Forward P/E: {_X(float(dc_data.get("forward_pe") or 0) or forward_pe)}
- P/S (TTM): {_X(fund.get('ps_ratio'))}
- EV/EBITDA: {_X(fund.get('ev_ebitda'))}
- Beta: {effective_beta}
- Gross Margin: {_P(fund.get('gross_margin'))}{" (Non-GAAP; GAAP may vary ~2-3%)" if fund.get('gross_margin') else ""}
- Dividend Yield: {f"{dividend_yield*100:.2f}%" if dividend_yield and dividend_yield > 0.001 else "Minimal (<0.1%)"}

ANALYST CONSENSUS:
- Recommendation: {analyst_consensus or 'N/A'} ({analyst_count or 'N/A'} analysts)
- Price Target (Mean): {((currency_sym if _is_local_currency else "$") + str(round(display_target, 2))) if display_target else 'N/A'}{" [" + fv_label + "]" if target_is_estimate else ""}
- Upside Potential: {f"{((display_target/real_price)-1)*100:.1f}%" if display_target and real_price else 'N/A'}
{"- NOTE: No analyst coverage found. Target shown is EisaX Fair Value Estimate (Forward EPS × " + str(valuation_pe) + "x sector P/E). Present as 'EisaX Fair Value Estimate' in section 5, NOT as analyst consensus. Do NOT use SMA200 as a price target." if target_is_estimate else ""}

BALANCE SHEET:
- Cash: {_B(fund.get('cash'))}
- Total Debt: {_B(fund.get('total_debt'))}
- Debt/Equity: {fund.get('debt_equity', 'N/A')}
- Current Ratio: {fund.get('current_ratio') or 'N/A'}

EARNINGS:
- Last Earnings Date: {fund.get('last_earnings_date', 'N/A')}
- NEXT EARNINGS DATE: {next_earnings or 'N/A'}
- EPS Actual vs Est (last): ${fund.get('last_eps_actual', 'N/A')} vs ${fund.get('last_eps_estimate', 'N/A')}
- Earnings Surprise: {fund.get('earnings_surprise_pct', 'N/A')}%
- Next Quarter EPS Estimate: ${ev_out.get('eps_est_avg', 'N/A')} (range: ${ev_out.get('eps_est_low','?')} – ${ev_out.get('eps_est_high','?')})
- Next Quarter Revenue Estimate: {f"${ev_out['rev_est_avg']/1e9:.1f}B" if ev_out.get('rev_est_avg') else 'N/A'} (range: {f"${ev_out['rev_est_low']/1e9:.1f}B" if ev_out.get('rev_est_low') else '?'} – {f"${ev_out['rev_est_high']/1e9:.1f}B" if ev_out.get('rev_est_high') else '?'})

MARKET SENTIMENT (Fear & Greed Index):
- Score: {fg_data.get('score', 'N/A')} / 100
- Rating: {fg_data.get('rating', 'N/A')} ({fg_data.get('label_ar', '')})
- Implication: {"Extreme fear — historically a contrarian buy signal; staged entries become more favorable" if (fg_data.get('score') or 50) < 25 else "Fear zone — market is risk-off; tighter stop losses advised" if (fg_data.get('score') or 50) < 45 else "Neutral sentiment" if (fg_data.get('score') or 50) < 55 else "Greed — market momentum favors bulls, but watch for complacency" if (fg_data.get('score') or 50) < 75 else "Extreme greed — elevated risk of correction; use caution on new entries"}

TECHNICALS:
- Trend: {summary['trend']} (Price vs SMA200)
- Momentum: {summary['momentum']} (MACD)
- RSI: {summary['rsi']:.1f} → {summary['condition']}
- MACD: {summary.get('macd', 0):.2f} | Signal: {summary.get('macd_signal', 0):.2f} | {"Bullish crossover" if summary.get('macd', 0) > summary.get('macd_signal', 0) else "Bearish crossover"}
- SMA50: {currency_sym}{summary['sma_50']:,.2f} | SMA200: {currency_sym}{summary['sma_200']:,.2f}
- Price vs SMA50: {f"{((real_price - summary['sma_50']) / summary['sma_50'] * 100):+.1f}%" if real_price and summary.get('sma_50') and float(summary.get('sma_50',0)) != 0 else "N/A"} | vs SMA200: {f"{((real_price - summary['sma_200']) / summary['sma_200'] * 100):+.1f}%" if real_price and summary.get('sma_200') and float(summary.get('sma_200',0)) != 0 else "N/A"}
- ADX: {_adx_val:.1f} ({_adx_desc}) | ATR: {summary.get('atr', 0):.2f}
{"- ⚠️ Technical Note: Momentum indicators (MACD/RSI) reflect price-driven buying pressure, while ADX measures trend strength independently of direction. A bullish momentum reading alongside a weak ADX (< 25) indicates early-stage or range-bound price action — not a confirmed trend. Treat momentum signals with reduced confidence until ADX sustains above 25." if (summary.get('adx', 0) < 25 and (summary.get('macd', 0) > 0 or summary.get('rsi', 0) > 55)) else ""}
{(lambda v_t, v_a: f"""
VOLUME:
- Today: {v_t/1e6:.1f}M vs 90-day avg {v_a/1e6:.1f}M → {"🔴 LOW volume ({:.0f}% of avg) — weak conviction in move".format(v_t/v_a*100) if v_a and v_t/v_a < 0.75 else "🟢 HIGH volume ({:.0f}% of avg) — strong conviction".format(v_t/v_a*100) if v_a and v_t/v_a > 1.25 else "⚪ Normal volume ({:.0f}% of avg)".format(v_t/v_a*100) if v_a else "N/A"}
""" if v_a else "")(
    fund.get('volume_today', 0) or 0,
    fund.get('volume_avg90d', 0) or 0
)}
{(lambda h52, l52, p: f"""
TECHNICAL LEVEL LADDER (S/R — ordered by proximity):
RULE: ALL 7 rows MUST appear. Use R3→R2→R1→SPOT→S1→S2→S3. Sources: SMA/EMA > Fibonacci > Swing > 52W. If data insufficient for a level, show "N/A — insufficient data".

| Level | Price | Type | Basis |
|-------|-------|------|-------|
| R3    | {f"{currency_sym}{h52:,.2f}" if h52 else "N/A — insufficient data"} | Resistance | {f"52W High" if h52 else "—"} |
| R2    | {f"{currency_sym}{l52 + (h52-l52)*0.618:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Resistance | {f"Fib 61.8%" if (h52 and l52) else "—"} |
| R1    | {f"{currency_sym}{l52 + (h52-l52)*0.50:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Resistance | {f"Fib 50% / Mid-Range" if (h52 and l52) else "—"} |
| SPOT  | {f"{currency_sym}{p:,.2f}" if p else "N/A"} | Current Price | Live |
| S1    | {f"{currency_sym}{l52 + (h52-l52)*0.382:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Support | {f"Fib 38.2%" if (h52 and l52) else "—"} |
| S2    | {f"{currency_sym}{l52 + (h52-l52)*0.236:,.2f}" if (h52 and l52) else "N/A — insufficient data"} | Support | {f"Fib 23.6%" if (h52 and l52) else "—"} |
| S3    | {f"{currency_sym}{l52:,.2f}" if l52 else "N/A — insufficient data"} | Support | {f"52W Low" if l52 else "—"} |
INSTRUCTION: In Section 3, you MUST include this FULL 7-row S/R table. Never skip or omit any row. Reference S1/R1 for entry/stop placement. Mention volume to confirm level breaks.
""")(
    fund.get('week52_high', 0) or 0,
    fund.get('week52_low', 0) or 0,
    real_price or 0
)}
RISK:
- VaR (95%, daily): {var_95*100:.2f}%
- Max Historical Drawdown: {max_dd*100:.2f}%
{"" if not onchain_data else f"""
ON-CHAIN METRICS (LIVE):
- All-Time High: ${(onchain_data.get('ath') or 0):,.0f} (ATH change: {(onchain_data.get('ath_change_pct') or 0):.1f}%, date: {onchain_data.get('ath_date', 'N/A')})
- Supply: {(onchain_data.get('circulating_supply') or 0):,.0f} / {(onchain_data.get('max_supply') or 0):,.0f} ({onchain_data.get('supply_ratio', 0)}% mined)
- 24h Volume: ${(onchain_data.get('total_volume_24h') or 0)/1e9:.1f}B
- Market Cap Rank: #{onchain_data.get('mc_rank', 'N/A')}
{f'- Hash Rate: {onchain_data["hash_rate_eh"]:.0f} EH/s' if onchain_data.get('hash_rate_eh') else ''}
{f'- Active Addresses (24h): {onchain_data["active_addresses"]:,}' if onchain_data.get('active_addresses') else ''}
{f'- Transactions (24h): {onchain_data["n_tx_24h"]:,}' if onchain_data.get('n_tx_24h') else ''}
IMPORTANT: Use these on-chain metrics in your analysis. Discuss supply scarcity, network activity, and hash rate health.
"""}
{"" if not oil_data.get('price') else f"""
OIL PRICE DATA (LIVE):
- Brent Crude: ${oil_data['price']:.2f}/bbl ({oil_data['change_pct']:+.1f}%)
IMPORTANT: This is an ENERGY SECTOR stock. Oil prices are the #1 driver of revenue and valuation.
Include an Oil Price Sensitivity Analysis table in your report showing impact at $50, $60, $70, $80, $90/bbl.
Discuss OPEC+ dynamics and energy transition risks.

OIL PRICE SENSITIVITY (pre-computed):
| Oil Price (Brent) | Change from Current | Est. Revenue Impact | Est. Stock Price |
|-------------------|--------------------|--------------------|-----------------|
| ${oil_data['price']:.0f}/bbl (current) | — | Base | {currency_sym}{real_price or 0:,.2f} |
| $90/bbl | {((90 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((90 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (90 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $80/bbl | {((80 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((80 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (80 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $70/bbl | {((70 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((70 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (70 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $60/bbl | {((60 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((60 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (60 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
| $50/bbl | {((50 - oil_data['price']) / oil_data['price'] * 100):+.0f}% | {((50 - oil_data['price']) / oil_data['price'] * 70):+.0f}% | {currency_sym}{(real_price or 0) * (1 + (50 - oil_data['price']) / oil_data['price'] * 0.55):,.2f} |
"""}
{(f"""SCENARIO ANALYSIS (Energy-Sector — Oil-Price-Adjusted):
Note: Impact already pre-calculated using 0.55x oil sensitivity. Copy EXACTLY — do NOT add extra columns.
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Oil Spike $150+/bbl | +{((((150 - oil_data.get('price',80)) / oil_data.get('price',80)) * 55)):.1f}% | {currency_sym}{(real_price or 0) * (1 + (((150 - oil_data.get('price',80)) / oil_data.get('price',80)) * 0.55)):,.2f} | Hold / partial profit |
| 🛢️ Oil Crash to $50/bbl | {(-((oil_data.get('price',80)-50)/oil_data.get('price',80))*55):.1f}% | {currency_sym}{((real_price or 0) * (1 + (-((oil_data.get('price',80)-50)/oil_data.get('price',80))*55))/100):,.2f} | Gold + Tech |
| 📉 OPEC+ Production Surge | {(-18 * 0.55):.1f}% | {currency_sym}{(real_price or 0) * (1 + (-18 * 0.55)/100):,.2f} | Diversified equities |
| 🌱 Energy Transition (long-term) | {(-30 * 0.55 * 0.75):.1f}% | {currency_sym}{(real_price or 0) * (1 + (-30 * 0.55 * 0.75)/100):,.2f} | Clean energy + Tech |
| 🏦 Fed Rate Shock +2% | {((-8 * max(float(effective_beta), 0.4)) + (-5 * 0.55)):.1f}% | {currency_sym}{(real_price or 0) * (1 + ((-8 * max(float(effective_beta), 0.4)) + (-5 * 0.55))/100):,.2f} | Treasuries + Cash |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if is_energy else (f"""SCENARIO ANALYSIS (UAE Real Estate — Geopolitical + Rate Sensitive):
Note: Dubai real estate reacts to regional geopolitics AND global rates, not just market beta ({effective_beta}).
Use -20% to -30% for geopolitical scenarios regardless of low beta — tourist/investor sentiment collapses in conflict.
| Scenario | Impact Driver | Est. Price Impact | Implied Price ({currency_sym}) | Suggested Hedge |
|----------|--------------|------------------|--------------------------|-----------------|
| 🚀 Dubai Tourism Boom | +35% tourism surge | +{(35 * 0.40):.1f}% | {currency_sym}{(real_price or 0) * (1 + (35 * 0.40)/100):,.2f} | Hold / add on dips |
| 🌍 Iran/Hormuz Conflict | Gulf security crisis | -{(28):.1f}% | {currency_sym}{(real_price or 0) * (1 - 28/100):,.2f} | Gold + global REITs |
| 📉 Dubai Bear Market | -30% DFM correction | -{(30 * 0.85):.1f}% | {currency_sym}{(real_price or 0) * (1 - 30 * 0.85/100):,.2f} | Cash + Bonds |
| 🏦 Fed Rate Shock +2% | Higher financing cost | -{(18 * max(float(effective_beta), 0.35)):.1f}% | {currency_sym}{(real_price or 0) * (1 - 18 * max(float(effective_beta), 0.35)/100):,.2f} | US Treasuries |
| 🌱 Expo/Infrastructure Catalyst | Mega-project boost | +{(20 * 0.50):.1f}% | {currency_sym}{(real_price or 0) * (1 + 20 * 0.50/100):,.2f} | Hold / add |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if (
    any(x in (fund.get('sector','') or '').lower() for x in ('real estate', 'property', 'reits'))
    and target.upper().endswith(('.DU', '.AE'))
) else (f"""SCENARIO ANALYSIS (Crash-Recovery — Post -39%+ Event):
⚠️ This stock experienced a severe single-day crash. Beta-adjusted scenarios are NOT meaningful here.
Use event-driven scenarios instead (corporate action, mean-reversion, or further collapse).
| Scenario | Trigger | Price Impact | Implied Price ({currency_sym}) | Suggested Action |
|----------|---------|-------------|--------------------------|-----------------|
| ✅ Corporate Action Clarified | Rights issue priced in — stock normalises | +{(45):.0f}% | {currency_sym}{(real_price or 0) * 1.45:,.2f} | BUY on confirmed clarity |
| 🔄 Partial Mean Reversion | Stock recovers 50% of crash | +{(25):.0f}% | {currency_sym}{(real_price or 0) * 1.25:,.2f} | Hold / add gradually |
| ⚠️ Fundamental Impairment | Crash = real earnings deterioration | -{(30):.0f}% | {currency_sym}{(real_price or 0) * 0.70:,.2f} | STOP LOSS immediately |
| 📉 Continued Selling / Forced Liquidation | No buyers for 1-2 weeks | -{(20):.0f}% | {currency_sym}{(real_price or 0) * 0.80:,.2f} | Volume confirmation pending |
| 🏦 EM Currency Devaluation | Local currency weakens -15% | -{(15):.0f}% | {currency_sym}{(real_price or 0) * 0.85:,.2f} | Hedge with USD exposure |
CRITICAL INSTRUCTION: In section 8, present THESE crash-recovery scenarios instead of generic beta-adjusted ones.
The #1 question investors need answered is: WHY did the stock crash -39%? Address this directly.
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""" if abs(change_pct or 0) >= 20 else f"""SCENARIO ANALYSIS (Beta-Adjusted — use these in section 9 of your report):
Note: Beta = {effective_beta}. Impact already pre-calculated (Market_Move × Beta). Copy EXACTLY — do NOT add extra columns.
REQUIREMENT: Show at least 2 BULLISH rows (🚀💡📈) and at least 2 BEARISH rows (📉🏦🤖⚠️).
| Scenario | Impact | Implied Price | Suggested Hedge |
|----------|--------|---------------|-----------------|
| 🚀 Bull Market Rally (+20%) | {(20 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (20 * float(effective_beta))/100):.2f} | Hold / add on dips |
| 💡 Fed Pivot / Rate Cut (+15%) | {(15 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (15 * float(effective_beta))/100):.2f} | Growth + Tech |
| 📉 AI/Tech Slowdown (-20%) | {(-20 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-20 * float(effective_beta))/100):.2f} | Healthcare + Staples |
| 🏦 Fed Rate Shock +2% (-18%) | {(-18 * float(effective_beta)):.1f}% | ${(real_price or 0) * (1 + (-18 * float(effective_beta))/100):.2f} | Value stocks + Cash |
INSTRUCTION FOR SECTION 9 (Scenario Analysis):
- You MUST include 3 core scenarios: Bear, Base, Bull — each with probability + expected price + expected return
- Core scenario probabilities MUST sum to 100%
- Any Macro Shock / Black Swan / Tail Risk scenario must be labeled as "💥 Tail Risk Overlay" and shown SEPARATELY
- Tail Risk Overlay must NOT be included in Expected Value calculation
- Expected Value = Σ(core_probability × core_return) across Bear/Base/Bull ONLY
- Show the EV calculation explicitly: "Expected Value: X.X%"
- After the core table, show tail risk separately:
  💥 **Tail Risk Overlay** | ~-25% | [trigger] | [hedge]
  ⚠️ *Not included in Expected Value calculation*
""")))}
{(lambda: (
    __import__('core.news_engine_client', fromlist=['build_news_prompt_block'])
    .build_news_prompt_block(_engine_news_data, target)
    if _engine_news_data and (_engine_news_data.get('direct') or _engine_news_data.get('sector') or _engine_news_data.get('country'))
    else (
        (chr(10) + "LATEST NEWS (LIVE — integrate into Section 4 Risks and Section 7 Why Now):" + chr(10)
         + chr(10).join(f"- {n['title']}" for n in (ctx.get('news_links') or [])[:5]) + chr(10)
         + "INSTRUCTION: Reference at least 1-2 of these headlines in Section 4 Key Risks and/or Section 7 Why Now.")
        if ctx.get('news_links') else ""
    )
)())}"""

    # ETF data_block override
    if etf_meta:
        try:
            from core.etf_intelligence import (
                build_etf_data_block as _build_etf_db,
                build_etf_scenarios as _build_etf_sc,
            )
            from core.macro_intelligence import get_live_macro as _etf_glm
            _etf_macro_live = {}
            try:
                _etf_macro_live = _etf_glm()
            except Exception as exc:
                logger.debug("[build_data_block] ETF macro fetch failed for %s: %s", target, exc)
            _etf_db = _build_etf_db(
                etf_meta, target, real_price or 0, change_pct or 0,
                summary, fg_data, macro=_etf_macro_live, var_95=var_95, max_dd=max_dd,
            )
            _etf_scenarios = _build_etf_sc(etf_meta["etf_type"], real_price or 100, _etf_macro_live)
            data_block = _etf_db + "\n\n" + _etf_scenarios
            logger.info("[build_data_block] ETF override: %s (%s)", target, etf_meta["etf_type"])
            if not fund.get("sector") or fund.get("sector") in ("Unknown", "N/A", ""):
                _is_futures_ticker = target.upper().endswith("=F") or target.upper() in (
                    "GC=F", "SI=F", "CL=F", "NG=F", "PL=F", "PA=F", "HG=F", "BZ=F"
                )
                _etf_sector_map = {
                    "commodity_gold": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_silver": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_platinum": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_palladium": "Commodities - Precious Metals" if _is_futures_ticker else "ETF - Precious Metals",
                    "commodity_copper": "Commodities - Industrial Metals" if _is_futures_ticker else "ETF - Industrial Metals",
                    "commodity_oil": "Commodities - Energy" if _is_futures_ticker else "ETF - Energy",
                    "commodity_other": "Commodities" if _is_futures_ticker else "ETF - Commodities",
                    "bond_treasury": "Fixed Income",
                    "bond_corporate": "Fixed Income",
                    "bond_tips": "Fixed Income",
                    "equity_index_us": "Equities - US Index",
                    "equity_index_intl": "Equities - International",
                    "equity_sector": "Equities - Sector",
                    "reit_etf": "Real Estate",
                    "leveraged": "Leveraged ETF",
                    "dividend": "Equities - Dividend",
                }
                fr.fund["sector"] = _etf_sector_map.get(etf_meta["etf_type"], "ETF")
        except Exception as _etf_db_e:
            logger.debug("[build_data_block] ETF override skipped: %s", _etf_db_e)

    # X sentiment block (appended to data_block)
    x_data = fr.x_data or {}
    if x_data and x_data.get("sentiment") and x_data.get("source") != "grok-unavailable":
        _xs   = x_data.get("sentiment", "")
        _xsc  = x_data.get("score", 0.0)
        _xsum = x_data.get("x_summary", "")
        _xbrk = x_data.get("breaking")
        _xthm = x_data.get("themes", [])
        _xpst = x_data.get("top_posts", [])

        _x_block = "\n\n--- X/Twitter Sentiment (Grok Live · last 48h) ---\n"
        _x_block += f"Overall: {_xs} (score: {_xsc:+.2f})\n"
        if _xsum:
            _x_block += f"Summary: {_xsum}\n"
        if _xbrk:
            _x_block += f"⚡ BREAKING: {_xbrk}\n"
        if _xthm:
            _x_block += f"Key Themes: {' · '.join(_xthm)}\n"
        if _xpst:
            _x_block += "Top Posts from X:\n"
            for _p in _xpst[:4]:
                _lk  = f" ({_p.get('likes',0):,} likes)" if _p.get("likes") else ""
                _src = _p.get("source", "")
                _txt = _p.get("text", "")[:160]
                _dt  = _p.get("date", "")
                _imp = _p.get("impact", "Neutral")
                _ico = "🟢" if _imp == "Positive" else "🔴" if _imp == "Negative" else "⚪"
                _x_block += f"  {_ico} {_src}{_lk} ({_dt}): \"{_txt}\"\n"
        _x_block += (
            "INSTRUCTION: Use this X sentiment data in Section 8 (Why Now?) under a "
            "'📱 X Sentiment' bullet. If there is BREAKING news, mention it in Section 4 "
            "(Key Risks). ONLY cite sources that appear in the Top Posts above."
        )
        data_block += _x_block
        logger.info("[build_data_block] X sentiment injected for %s: %s (%+.2f)", target, _xs, _xsc)

    return data_block

def build_analytics_prompt(
    target: str,
    data_block: str,
    ctx: dict,
    scorecard_verdict_hint: str,
    is_arabic: bool,
    brain_ctx: str,
    local_injection: str,
    research_summary: str,
    original_target: str | None = None,
    macro_block: str = "",
    pre_entry: str = "N/A",
    pre_stop: str = "N/A",
    pre_target: str = "N/A",
    user_ctx_block: str = "",
    research_context: str = "",
) -> str:
    """
    Builds the full DeepSeek investment memo prompt.
    Extracted from _handle_analytics lines 3684–3809.
    """
    from datetime import datetime as _dt

    fg_data      = ctx.get("fg_data", {}) or {}
    is_energy    = ctx.get("is_energy", False)
    oil_data     = ctx.get("oil_data", {}) or {}
    is_crash     = ctx.get("is_crash", False)
    crash_direction = ctx.get("crash_direction", "")
    change_pct   = ctx.get("change_pct", 0.0) or 0.0
    etf_meta     = ctx.get("etf_meta")
    x_data       = ctx.get("x_data", {}) or {}
    currency_sym = ctx.get("currency_sym", "$")
    currency_lbl = ctx.get("currency_lbl", "USD")

    _display_ticker = original_target if original_target else target

    # ── Interpretation block injection (Week 3) ────────────────────────────
    interpretation_block = ctx.get("interpretation_block", "")
    approved_phrase_block = ctx.get("approved_phrase_block", "")
    if interpretation_block:
        data_block = data_block + "\n\n" + interpretation_block
    if approved_phrase_block:
        data_block = data_block + "\n\n" + approved_phrase_block

    prompt = f"""You are EisaX, Chief Investment Officer - built by Eng. Ahmed Eisa.

🚨 CRITICAL: Today's date is {_dt.now().strftime("%B %d, %Y")}.{research_summary}
   - You MUST use this EXACT date in your memo header
   - Any historical data reference must be clearly labeled as "historical"
   - All analysis must reflect current 2026 market conditions
   - MEMO SUBJECT LINE: In the memo header, the "Re:" line MUST use the ticker exactly as the user typed it: **{_display_ticker}** — NOT the resolved symbol. E.g. if user typed "XAUUSD", write "Re: Analysis of XAUUSD" not "Re: Analysis of GC=F".

🔒 DATA INTEGRITY LOCK — NON-NEGOTIABLE COMPLIANCE RULE:
Every number, price, yield, ratio, metric, and date in this memo MUST come
EXCLUSIVELY from the DATA BLOCK provided below. You are STRICTLY FORBIDDEN
from using your training-data knowledge for ANY financial figure. This includes:
  - Stock prices (current, historical, or peer prices)
  - Dividend yields (for the subject company OR any peer)
  - P/E ratios — trailing, forward, or peer (use only what is in the data block)
  - Market cap, revenue, EPS, margins, ROIC, ROE
  - Earnings dates or fiscal quarter labels
  - Analyst price targets or consensus ratings
  - ANY percentage, dollar figure, or numeric metric not explicitly present in the data
⛔ If a value is NOT present in the DATA BLOCK → write "data not available at time of analysis."
⛔ NEVER estimate, approximate, recall, or infer a number from training knowledge.
⛔ NEVER use hedging phrases like "approximately", "historically around", or "typically" to
   disguise a training-data number. If the data doesn't have it, say it is unavailable.
This is a FINANCIAL COMPLIANCE REQUIREMENT. Using fabricated or recalled figures
exposes clients to materially misleading information. Violations are unacceptable.

Your advantage over general AI assistants:
- You are a SPECIALIZED financial analyst with 20+ years CIO experience
- You have access to LIVE market data provided in the DATA BLOCK below (not training data)
- You provide institutional-grade analysis using ONLY the supplied data

🎯 SCORECARD PRE-VERDICT (computed before this memo): **{scorecard_verdict_hint}**
⛔ TONE ALIGNMENT RULE — MANDATORY:
Your memo MUST be tonally consistent with the above verdict:
- If verdict = REDUCE or SELL → Executive Summary must reflect caution. No "compelling entry", "attractive opportunity", or buy language. Acknowledge the headwinds clearly.
- If verdict = HOLD → Balanced tone. Acknowledge both upside potential and risks equally.
- If verdict = BUY or ACCUMULATE → Constructive tone. State the opportunity while acknowledging risks.
- ⛔ NEVER write a bullish Executive Summary when the verdict is REDUCE/SELL. This creates a contradiction the client will immediately notice and destroys credibility.

🔴 LANGUAGE QUALITY RULES:
- ⛔ NEVER use boilerplate phrases like "according to recent analyst data", "market observers note", "analysts suggest", or "industry experts believe" — these are empty filler. Use the ACTUAL data provided or state explicitly that the data is unavailable.
- ⛔ NEVER cite a news source that is NOT in the LATEST NEWS section of the data below. Do NOT reference "The Times of India", "Hindustan Times", regional newspapers, blogs, or any outlet from your training knowledge. If you cite a source, it MUST appear verbatim in the LATEST NEWS section.
- ⛔ NEVER invent or paraphrase headlines not present in the LATEST NEWS data. If no relevant news exists, say "No relevant headlines at time of analysis."
- ⛔ BE CONSISTENT on valuation: if the Scorecard labels Forward P/E as "🟢 Reasonable", do NOT describe the same P/E as "elevated" in the memo body. Use the same label throughout.
- ⛔ EARNINGS DATE: Use ONLY the exact date from the data. NEVER combine a fiscal quarter label from one year with a date from another year (e.g. "Q1 2027 on April 29, 2026" is wrong). If unsure of the fiscal quarter label, just say "next earnings report on [date]".
- ✅ Peer comparisons in Section 6 MUST include actual numbers. E.g., "GOOGL trades at 22x forward P/E vs {target}'s Xx" — not just "GOOGL is a peer".
- ✅ If EPS growth estimate is available in the data, include the YoY % in Section 2.

Analyze the following data and write an institutional-grade investment memorandum.
{user_ctx_block}
{data_block}

{(f"""
⚠️ ETF ANALYSIS MODE — {etf_meta['etf_label'] if etf_meta else ''}
This is an ETF, NOT a stock. Follow ETF-specific rules:
- Section 2 = "{"Commodity Analysis" if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "Fund Analysis"}" (NOT Fundamental Analysis): Discuss what the fund/contract tracks, expense ratio cost drag, AUM liquidity, and how the underlying asset/index is valued. NO EPS, Revenue, ROE, ROIC, or corporate metrics.
- Section 5 = "Market Catalysts": No analyst consensus. Discuss macro catalysts that drive this fund (rate moves, commodity shifts, sector rotation, etc.).
- Section 6 = "⚔️ Peer Comparison": name 2 direct alternative funds (by ticker). Compare expense ratio, yield/return profile, and AUM in exactly 2 sentences. No corporate competitors — funds only.
- Section 7 = "EisaX Outlook": Compare to ALTERNATIVE investments (e.g., for GLD: compare to TLT, T-bills, TIPS; for TLT: compare to HYG, cash, SPY). Include one specific number and one risk/reward statement.
- Section 9 = Use the ETF-SPECIFIC scenario table provided in the data.
- Do NOT mention P/E ratio, EPS, Revenue, ROE, ROIC, analyst price targets, or earnings dates.
""") if etf_meta else ""}
Structure your response with these sections (ALL sections are MANDATORY — do NOT skip any):
1. **Executive Summary** (2-3 sentences with clear stance)
2. **{"Commodity Analysis" if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "Fund Analysis" if etf_meta else "Fundamental Analysis"}** ({"macro drivers, real yield sensitivity, USD relationship, central bank demand, and supply/demand dynamics for the underlying commodity. Do NOT use ETF/fund language — this is a commodity futures contract." if etf_meta and etf_meta.get("etf_type","").startswith("commodity") else "what the fund tracks, expense ratio drag, AUM size, macro drivers of the underlying asset" if etf_meta else "growth quality, profitability, valuation - mention Forward P/E and Gross Margin GAAP note"})
3. **Technical Outlook** (MANDATORY — you MUST include ALL of the following from the TECHNICALS data):
   - SMA50, SMA200, RSI, MACD, ADX values with trend direction and momentum condition
   - CRITICAL: use the exact RSI condition label from data — e.g. "RSI: 32.2 (Near Oversold)" not your own label
   - Volume vs average: state if volume is LOW/NORMAL/HIGH vs 90-day avg and what this means for conviction
   - Fibonacci levels: mention the nearest resistance ABOVE current price and the key support BELOW (from FIBONACCI LEVELS data)
   - ⚠️ Technical Note: Momentum indicators (MACD/RSI) reflect price-driven buying pressure, while ADX measures trend strength independently of direction. A bullish momentum reading alongside a weak ADX (< 25) indicates early-stage or range-bound price action — not a confirmed trend. Treat momentum signals with reduced confidence until ADX sustains above 25.
   - ⛔ Do NOT repeat these technical facts in Section 8 (Why Now) — Section 8 focuses on TIMING and CATALYSTS only
4. **Key Risks** (top 2-3 BUSINESS risks with severity rating):
   ⛔ DATA GAPS ARE NOT RISKS: If fundamental metrics (ROE, ROIC, Net Margin, etc.) are unavailable, note this ONCE in Section 2 as a data limitation. Do NOT list "Weak Fundamental Metrics" or "Data Unavailability" as a Key Risk in Section 4.
   ✅ Section 4 must contain only genuine business, macro, commodity, regulatory, or market risks (e.g., oil price volatility, competition, geopolitical risk, rate sensitivity, regulatory change).
   MANDATORY: If LATEST NEWS appears in the data, reference at least one relevant headline here as a named risk. E.g., "Geopolitical Risk (Severity: High): [headline about Hormuz/Iran/OPEC]..."
5. **Analyst Consensus & Catalysts** (mention price target, upside %, upcoming earnings)
6. **⚔️ Peer Comparison** (MANDATORY — do NOT skip — exactly 2 sentences, no more):
{"   ETF mode: name 2 direct alternative funds. Compare expense ratio, yield/return, and AUM. Format: \"vs [FUND]: [difference]. [why an investor would choose this one over it].\"" if etf_meta else
"   Stock mode — compare to the single closest DIRECT competitor in the same sub-industry:\n"
"   • Sentence 1 (Valuation): state both forward P/E (or EV/EBITDA, P/S for growth) values and the % premium or discount.\n"
"   • Sentence 2 (Edge): where does this company lead or lag vs the peer? (growth rate, margin, market share, moat, product pipeline)\n"
"   Format: \"vs [PEER_TICKER]: [valuation sentence]. [competitive position sentence].\"\n"
"   Example: \"vs NVDA: AMD trades at 24x fwd P/E vs NVDA's 35x — a 31% discount. AMD leads in CPU market share but lags NVDA's data center GPU dominance (NVDA holds ~80% market share vs AMD ~15%).\"\n"
"   ⛔ Do NOT write more than 2 sentences. ⛔ Do NOT include any rating or recommendation.\n"
"   ⛔ DATA LOCK — PEER TABLE: Every price, yield, P/E, and metric shown for ANY peer MUST come from the DATA BLOCK. Do NOT recall, estimate, or approximate peer figures from training knowledge.\n"
"   ⛔ If peer forward P/E is NOT in the data block → write 'forward P/E not available' (never estimate from training knowledge).\n"
"   ⛔ If the subject ticker appears in the peer table, its price MUST exactly match the LIVE PRICE shown at the top of the data block — no discrepancy allowed.\n"
"   ⛔ If you truly cannot compare numerically, name the peer and compare qualitatively (margins, growth, market share) using only facts stated in the data block.\n"
"   ⚡ PEER SELECTION: Choose the MOST RELEVANT competitor — for cloud/software companies this may be AMZN (AWS) or META, not necessarily GOOGL. For UAE/Saudi companies compare to the closest regional peer."}"
"\n⛔ BANNED PHRASES — never write these regardless of verdict:\n"
"- \"bullish trends are expected in 2026\" or any variation\n"
"- \"diversification is recommended, aligning with our balanced view\"\n"
"- Any generic forward-looking phrase not supported by the data above.\n"

7. **EisaX Outlook** — Write 2-3 sentences with:
   - One specific number (e.g. implied return, EV/EBITDA vs peers, FCF yield, or PEG ratio)
   - One clear risk/reward statement
   - ⛔ DO NOT include any verdict, buy/sell/hold rating, or recommendation
   - ⛔ DO NOT write any score or scorecard
   - The official verdict is auto-generated in the EisaX Scorecard below

8. **⏰ Why Now?** (MANDATORY — focus on TIMING and CATALYSTS, not technical analysis which belongs in Section 3):
   • Market Sentiment: Fear & Greed at {fg_data.get('score','N/A')} ({fg_data.get('rating','N/A')}) — what extreme reading means for entry timing RIGHT NOW
   • Upcoming Catalyst: next earnings date, product launch, regulatory event, or sector-specific driver — cite LATEST NEWS if relevant; explain WHY this catalyst matters NOW
   • Risk/Timing: one specific risk to the entry timing (NOT a repeat of Section 4 risks — frame it as timing risk, e.g. "Could fall further before earnings", "Momentum may not reverse until X")
   {"• Oil Price: Brent at $" + str(round(oil_data.get('price',0),2)) + "/bbl — impact on revenue and margins" if is_energy else ""}
   {("• 📱 X Sentiment: Copy EXACTLY — sentiment is **" + str(x_data.get('sentiment','')) + "** (score: " + f"{x_data.get('score',0):+.2f}" + "). Key themes: " + ", ".join(x_data.get('themes',[])[:2]) + ". Do NOT change the sentiment label or score — use them verbatim. Cite specific accounts from the Top Posts if available.") if x_data and x_data.get("sentiment") else ""}
   Format: "• [Factor]: [Implication]"

9. **🌍 Advanced Scenario Analysis**
   {"Include the Oil Price Sensitivity table AND the Energy-Sector scenario table from the data. Show how different oil prices ($50-$90/bbl) affect this stock." if is_energy else "Include a markdown table of 4 beta-adjusted scenarios from the SCENARIO ANALYSIS section in the data. REQUIREMENT: At least 2 scenarios must be BULLISH (upside cases) and at least 1 must be BEARISH. Do NOT generate all-bearish or all-downside scenarios — this is for institutional investors who need balanced upside and downside analysis."}
   Format:
   Emoji rule: 🚀📈💡 for BULLISH rows · 📉🏦🤖⚠️ for BEARISH rows. NEVER use 📉 on a positive-impact row.
   ⛔ The SCENARIO ANALYSIS data already has exactly 4 columns: Scenario | Impact | Implied Price | Suggested Hedge. Copy this table EXACTLY — do NOT add a Market Move column or split any cell. Use "Expected Price" as the header for the price column.
   | Scenario | Impact | Expected Price | Suggested Hedge |
   |----------|--------|----------------|-----------------|

{"10. **🛢️ Oil Price Sensitivity** (MANDATORY for energy stocks): Include the full Oil Price Sensitivity table from the data showing revenue impact at $50, $60, $70, $80, $90/bbl. Discuss the breakeven oil price and OPEC+ production outlook." if is_energy else ""}

Use actual numbers. Be specific. Institutional tone.
{"CRITICAL: This is an ENERGY sector stock. Oil prices are the PRIMARY driver. You MUST discuss oil price impact throughout the report, include the sensitivity table, and reference Brent crude at $" + str(round(oil_data.get('price',0),2)) + "/bbl." if is_energy else ""}
{"CURRENCY: Use " + currency_sym + " (" + currency_lbl + ") for ALL price references — NOT USD." if currency_lbl != "USD" else ""}
{"LANGUAGE: The user's request was in Arabic. Write the FULL report in Arabic. IMPORTANT: Use the SAME number of sections, SAME level of detail, and ALL 9 sections — do NOT simplify or shorten because it is in Arabic. Arabic and English reports must be identical in depth and structure. Section 6 (Peer Comparison) must still be exactly 2 sentences with competitor ticker and valuation numbers in Arabic." if is_arabic else "LANGUAGE: Write in English."}
{"🚨 EXTREME PRICE MOVE ALERT — " + crash_direction + " (" + f"{change_pct:+.2f}%" + " single-day move detected): This MUST be the FIRST thing addressed in Section 1 (Executive Summary). In Section 4 (Key Risks), you MUST investigate and explain the likely cause: check if this is an ex-dividend drop, rights issue (capital increase), trading halt lifted, forced selling, major news event, or circuit-breaker trigger. State the most probable cause based on available data. Do NOT treat this as a normal trading day — this is an exceptional event requiring forensic analysis." if is_crash else ""}
IMPORTANT RULES:
- Do NOT mention dividend yield unless above 0.5%
- Entry zone must ALWAYS be BELOW the current live price
- Stop loss: one consistent value only
- Analyst count: use the EXACT number from the data. Do NOT round or cap it.
- ⛔ EARNINGS DATE RULE: The NEXT EARNINGS DATE in the data is the ONLY date to use. Do NOT derive or guess fiscal quarter labels (Q1/Q2/Q3/Q4) from calendar dates — the fiscal year varies by company. Use the date as-is (e.g. "April 29, 2026") and say "next earnings" not "Q1 FY2027".
- ⛔ NEVER write "Score: XX/100" in sections 1-8. That appears ONLY in the Scorecard.
- ⛔ DO NOT create any scorecard table, score breakdown, scoring methodology, or positioning section in your response. NO "Growth: X/30", "Valuation: X/20", "Score: XX/100", "Confidence Score", "Entry Zone", "Stop Loss", "Target" sections. The EisaX Proprietary Scorecard AND Positioning Guide are automatically appended below your memo — ANY duplication causes critical display errors and will be rejected.
- ⛔ Your response MUST end after section 9. Do NOT add any additional sections, tables, or blocks after section 9.
- ALL 9 sections above are MANDATORY. Do NOT skip Technical Outlook, Why Now, or Advanced Scenario Analysis.
- ⛔ NEWS INTEGRATION RULE: If FRESH NEWS CONTEXT is provided in the data, you MUST cite at least 1 specific headline by name in Section 1 (Executive Summary) AND at least 1 in Section 4 (Key Risks). Do NOT generically mention "recent news" — quote or paraphrase the actual headline title. Failing to integrate news is a critical quality failure.
- ⛔ CONSISTENCY RULE: Section 8 (Why Now) must be CONSISTENT with the Scorecard verdict. If the verdict is REDUCE or SELL, do NOT frame the analysis as a "contrarian opportunity" or suggest it is a good entry point. Instead, explain what would need to change for the thesis to improve. If the verdict is HOLD/BUY, you may describe constructive entry timing.
- ⛔ UPSIDE LANGUAGE RULE: Only use "strong upside" when upside potential is genuinely >20%. For <10% upside use "modest upside" or "limited upside". For 10-20% upside use "moderate upside". Never call +3% to +5% returns "strong upside" — that misleads investors.
- ⛔ INTERPRETATION LOCK: If an [INTERPRETATION BLOCK — LOCKED] is provided in the data, you MUST use those exact labels verbatim. Do NOT substitute stronger or weaker claims. Do NOT reinterpret the labels. Narrate around them but never replace them.
    - Section 3 (Technical): use Trend Strength, RSI Zone, Support/Resistance Proximity, Volume Conviction
    - Section 8 (Why Now): use Entry Quality and Trend Strength for timing language
    - Section 2/7 (Fundamentals/Outlook): use Yield Quality for all yield references
    - Section 1 (Executive Summary): reflect Trend Strength and Entry Quality in stance language
Do NOT include a standalone Positioning section.{brain_ctx}
{macro_block}
"""

    prompt = prompt.replace("PLACEHOLDER_ENTRY", pre_entry)
    prompt = prompt.replace("PLACEHOLDER_TARGET", pre_target)
    prompt = prompt.replace("PLACEHOLDER_STOP", pre_stop)
    prompt += "\n\n🚨 MANDATORY: Entry=" + pre_entry + " | Stop=" + pre_stop + " | Target=" + pre_target + " — USE THESE EXACT LEVELS."
    if research_context:
        prompt += "\n\n" + research_context
    prompt += local_injection

    return prompt

