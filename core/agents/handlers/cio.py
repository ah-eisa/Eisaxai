# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import os
import state
from datetime import datetime
logger = logging.getLogger(__name__)


class CIOMixin:
    def _handle_cio_analysis(self, msg: str, sid: str = "default") -> Dict[str, Any]:
        """
        Direct CIO portfolio analysis — P&L, stress test, recommendation.
        Bypasses the full optimizer. Uses yfinance for live prices.
        """
        import yfinance as yf
        import re
        import os, requests as _req
        from datetime import datetime

        # ── Parse holdings from message ───────────────────────────────────────
        # Pattern: "TICKER: N shares @ average cost $PRICE" or "TICKER: N shares @ $PRICE"
        holdings = {}

        # Robust line-by-line parser — handles English, Arabic, Saudi tickers
        import re as _re2
        _ticker_pat = _re2.compile(r'([A-Z]{2,5}(?:\.[A-Z]{2,3})?|[0-9]{3,4}\.[A-Z]{2,3})')
        _number_pat = _re2.compile(r'[0-9]+(?:[,.][0-9]+)*')
        _skip_words = {'AT','OR','IN','OF','TO','BY','VS','AND','THE','FOR','SAR','USD','AED','EGP','SR'}

        for line in msg.replace(';','\n').split('\n'):
            line = line.strip().lstrip('-*\u2022 ')
            tks = [t for t in _ticker_pat.findall(line) if t not in _skip_words]
            if len(tks) == 1 and tks[0] not in holdings:
                ticker = tks[0]
                # Remove ticker from line before extracting numbers
                line_no_ticker = _ticker_pat.sub(' ', line)
                nums = [float(n.replace(',','')) for n in _number_pat.findall(line_no_ticker)]
                if len(nums) >= 2:
                    holdings[ticker] = {'shares': nums[0], 'avg_cost': nums[1]}
            elif len(tks) > 1:
                # Multiple tickers on one line — skip, handled by fallback
                pass

        # Fallback: whole-message scan for missed tickers
        if not holdings:
            _p = _re2.compile(
                r'([A-Z]{2,5}(?:\.[A-Z]{2,3})?|[0-9]{3,4}\.[A-Z]{2,3})'
                r'[^0-9]{0,20}([0-9]+(?:,[0-9]+)?)[^0-9]{0,30}([0-9]+(?:[.,][0-9]+)*)',
                _re2.IGNORECASE)
            _skip_words2 = {'AT','OR','IN','OF','TO','BY','VS','AND','THE','FOR','SAR','USD','AED','EGP','SR'}
            for m in _p.finditer(msg):
                t = m.group(1).upper()
                if t not in _skip_words2 and t not in holdings:
                    try:
                        holdings[t] = {'shares': float(m.group(2).replace(',','')),
                                       'avg_cost': float(m.group(3).replace(',',''))}
                    except Exception:
                        pass


        # ── Fetch live prices ─────────────────────────────────────────────────
        # Normalize crypto tickers — yfinance requires "BTC-USD" not "BTC"
        _CRYPTO_MAP = {
            'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
            'BNB': 'BNB-USD', 'ADA': 'ADA-USD', 'DOGE': 'DOGE-USD',
            'XRP': 'XRP-USD', 'DOT': 'DOT-USD', 'AVAX': 'AVAX-USD',
            'MATIC': 'MATIC-USD', 'LINK': 'LINK-USD', 'LTC': 'LTC-USD',
        }
        # UAE tickers need market_data_engine (yfinance returns 404 for .AE/.DU)
        _uae_tickers = {t for t in holdings if t.endswith('.DU') or t.endswith('.AE')}

        # Build original_ticker → yfinance_ticker map (None = skip yfinance)
        _yf_ticker_map = {}
        for t in holdings:
            if t in _CRYPTO_MAP:
                _yf_ticker_map[t] = _CRYPTO_MAP[t]
            elif t in _uae_tickers:
                _yf_ticker_map[t] = None   # handled by market_data_engine below
            else:
                _yf_ticker_map[t] = t

        # Batch-fetch all yfinance-eligible tickers
        _yf_tickers = [v for v in _yf_ticker_map.values() if v is not None]
        prices = {}
        try:
            if _yf_tickers:
                tickers_str = ' '.join(_yf_tickers)
                raw = yf.download(tickers_str, period='5d', auto_adjust=True,
                                  group_by='ticker', progress=False, threads=False)
                if isinstance(raw.columns, __import__('pandas').MultiIndex):
                    for t, yf_t in _yf_ticker_map.items():
                        if yf_t is None:
                            continue
                        try:
                            col = raw.xs(yf_t, axis=1, level=0)
                            prices[t] = float(col['Close'].dropna().iloc[-1])
                        except Exception:
                            pass
                else:
                    # Single-ticker download has flat columns
                    if 'Close' in raw.columns:
                        single_px = float(raw['Close'].dropna().iloc[-1])
                        for t, yf_t in _yf_ticker_map.items():
                            if yf_t is not None:
                                prices[t] = single_px
        except Exception as e:
            logger.error(f"[CIO] yfinance batch fetch failed: {e}")

        # Slow individual fallback for missed non-UAE tickers
        for t, yf_t in _yf_ticker_map.items():
            if yf_t is None or t in prices:
                continue
            try:
                info = yf.Ticker(yf_t).fast_info
                prices[t] = float(getattr(info, 'last_price', 0) or 0)
            except Exception:
                prices[t] = 0.0

        # UAE stocks: use market_data_engine (StockAnalysis / RapidAPI backed)
        for t in _uae_tickers:
            if prices.get(t, 0.0) > 0:
                continue
            try:
                from core.market_data_engine import get_latest_price as _get_uae_px
                _uae_res = _get_uae_px(t, 'AE')
                if _uae_res and _uae_res.get('close', 0) > 0:
                    prices[t] = float(_uae_res['close'])
                    logger.info(f"[CIO] UAE price for {t}: {prices[t]} AED")
            except Exception as _ue:
                logger.warning(f"[CIO] UAE price fetch failed for {t}: {_ue}")
            if t not in prices:
                prices[t] = 0.0

        # ── Currency detection & normalization ──────────────────────────────
        # Detect currency per ticker and normalize cost basis to match market price currency
        def _get_currency(ticker: str) -> tuple:
            """Returns (currency_code, symbol, usd_rate)
            usd_rate: multiply local price by this to get USD (1.0 for USD assets)
            """
            t = ticker.upper()
            if t.endswith('.SR'):
                return ('SAR', 'SAR', 1/3.75)   # 1 SAR = 0.2667 USD
            elif t.endswith('.AE') or t.endswith('.DU'):
                return ('AED', 'AED', 1/3.6725) # 1 AED = 0.2723 USD
            elif t.endswith('.CA'):
                return ('EGP', 'EGP', 1/50.0)   # approximate EGP rate
            elif t.endswith('.KW'):
                return ('KWF', 'KWF', 1/3070.0) # 1 KWD = 3.27 USD; 1000 fils = 1 KWD
            elif t.endswith('.QA'):
                return ('QAR', 'QAR', 1/3.64)   # 1 QAR = 0.2747 USD
            elif t.endswith('-USD') or t.endswith('BTC') or t.endswith('ETH'):
                return ('USD', '$', 1.0)
            else:
                return ('USD', '$', 1.0)         # US stocks default USD

        # ── Upgrade crypto prices via CoinGecko (real-time, no key needed) ─────
        _COINGECKO_IDS = {
            'BTC': 'bitcoin',    'ETH': 'ethereum',   'SOL': 'solana',
            'BNB': 'binancecoin','ADA': 'cardano',     'DOGE': 'dogecoin',
            'XRP': 'ripple',     'DOT': 'polkadot',    'AVAX': 'avalanche-2',
            'MATIC': 'matic-network', 'LINK': 'chainlink', 'LTC': 'litecoin',
        }
        _crypto_in_portfolio = [t for t in holdings if t in _COINGECKO_IDS]
        if _crypto_in_portfolio:
            try:
                _cg_ids = ','.join(_COINGECKO_IDS[t] for t in _crypto_in_portfolio)
                _cg_res = _req.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={_cg_ids}&vs_currencies=usd",
                    timeout=8
                )
                if _cg_res.status_code == 200:
                    _cg_data = _cg_res.json()
                    for t in _crypto_in_portfolio:
                        _cg_px = _cg_data.get(_COINGECKO_IDS[t], {}).get('usd', 0)
                        if _cg_px and float(_cg_px) > 0:
                            prices[t] = float(_cg_px)
                            logger.info(f"[CIO] CoinGecko live price {t}: ${_cg_px:,.2f}")
            except Exception as _cge:
                logger.warning(f"[CIO] CoinGecko fetch failed — keeping yfinance prices: {_cge}")

        # ── Compute P&L ───────────────────────────────────────────────────────
        today = datetime.now().strftime('%B %d, %Y')
        rows = []
        total_cost   = 0.0
        total_value  = 0.0
        table_lines  = ["| Ticker | Shares | Avg Cost | Current Price | Position Value (USD) | Unrealized P&L | Return |",
                        "|--------|--------|----------|---------------|----------------------|----------------|--------|"]

        for t, h in holdings.items():
            shares    = h['shares']
            avg_cost  = h['avg_cost']   # in local currency as user entered
            curr_px   = prices.get(t, 0.0)  # in local currency from yfinance
            currency, sym, usd_rate = _get_currency(t)

            # Normalize to USD for portfolio-level aggregation
            avg_cost_usd = avg_cost * usd_rate
            curr_px_usd  = curr_px  * usd_rate

            pos_cost  = shares * avg_cost_usd
            pos_val   = shares * curr_px_usd
            pnl       = pos_val - pos_cost
            ret_pct   = (pnl / pos_cost * 100) if pos_cost > 0 else 0.0
            total_cost  += pos_cost
            total_value += pos_val
            emoji = "📈" if pnl >= 0 else "📉"

            # Display in original currency for clarity
            if currency == 'USD':
                cost_display = f"${avg_cost:.2f}"
                px_display   = f"${curr_px:.2f}"
            else:
                cost_display = f"{avg_cost:.2f} {sym}"
                px_display   = f"{curr_px:.2f} {sym}"

            # Show fractional shares properly (e.g. 0.5 BTC, not "0")
            shares_display = f"{shares:g}" if shares != int(shares) else f"{int(shares):,}"
            table_lines.append(
                f"| **{t}** | {shares_display} | {cost_display} | {px_display} | "
                f"${pos_val:,.0f} | {emoji} ${pnl:+,.0f} | {ret_pct:+.1f}% |"
            )
            rows.append({'ticker': t, 'shares': shares, 'avg_cost': avg_cost_usd,
                        'curr_px': curr_px_usd, 'pos_cost': pos_cost,
                        'pos_val': pos_val, 'pnl': pnl, 'ret_pct': ret_pct,
                        'currency': currency})

        total_pnl     = total_value - total_cost
        total_ret_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        # ── Stress tests ──────────────────────────────────────────────────────
        scenarios = [(-0.15, "Mild Correction (-15%)"),
                     (-0.25, "Moderate Bear (-25%)"),
                     (-0.40, "Severe Crash (-40%)")]
        stress_lines = ["| Scenario | Portfolio Value | vs Cost Basis | P&L |",
                        "|----------|----------------|----------------|-----|"]
        for drop, label in scenarios:
            stressed_val = total_value * (1 + drop)
            vs_cost      = stressed_val - total_cost
            stress_lines.append(
                f"| {label} | ${stressed_val:,.0f} | "
                f"{'📉' if vs_cost < 0 else '📈'} ${vs_cost:+,.0f} | "
                f"{(vs_cost/total_cost*100):+.1f}% |"
            )

        # ── Build report body (profile card inserted later after region compute) ──
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        _report_pnl_block = f"""## 📊 Portfolio P&L Summary

{chr(10).join(table_lines)}

**Total Cost Basis:** ${total_cost:,.0f}
**Current Portfolio Value:** ${total_value:,.0f}
**Unrealized P&L: {pnl_emoji} ${total_pnl:+,.0f} ({total_ret_pct:+.1f}%)**

---

## 🧪 Stress Test Scenarios

{chr(10).join(stress_lines)}

---
"""

        # ── Fetch CURRENT dividend yields — parallel + cached ─────────────────
        # Lazy import — finance.py imports this mixin at class-body time,
        # so a module-top "from core.agents.finance import _div_yield_cache"
        # would be circular. Bound here gives the same shared TTLCache.
        from core.agents.finance import _div_yield_cache

        def _fetch_one_yield(ticker_str: str) -> float:
            """Fetch current market dividend yield (NOT yield-on-cost). Cached 1h."""
            cached = _div_yield_cache.get(f"dy_{ticker_str}")
            if cached is not None:
                return cached
            try:
                _t_info = yf.Ticker(ticker_str).info
                # trailingAnnualDividendYield is reliable decimal (0.004 = 0.4%)
                # dividendYield returns % value (0.4 for 0.4%) — avoid multiplying it
                _raw = float(_t_info.get("trailingAnnualDividendYield") or 0)
                result = min(max(_raw * 100, 0.0), 15.0)  # decimal→pct, clamp [0%,15%]
            except Exception:
                result = 0.0
            _div_yield_cache.set(f"dy_{ticker_str}", result)
            return result

        from concurrent.futures import ThreadPoolExecutor as _TPE
        _dy_tickers = [r['ticker'] for r in rows]
        with _TPE(max_workers=min(8, max(1, len(_dy_tickers)))) as _pool:
            _dy_vals = list(_pool.map(_fetch_one_yield, _dy_tickers))
        live_div_yields = dict(zip(_dy_tickers, _dy_vals))

        port_div_yield_current = sum(
            live_div_yields.get(r['ticker'], 0) * (r['pos_val'] / total_value)
            for r in rows
        ) if total_value > 0 else 0.0

        # ── Region classification ──────────────────────────────────────────────
        def _get_region(ticker: str) -> str:
            t = ticker.upper()
            if t in _CRYPTO_MAP or t.endswith('-USD'):
                return 'Crypto'
            elif t.endswith('.DU') or t.endswith('.AE'):
                return 'UAE'
            elif t.endswith('.SR'):
                return 'Saudi Arabia'
            elif t.endswith('.KW'):
                return 'Kuwait'
            elif t.endswith('.QA'):
                return 'Qatar'
            elif t.endswith('.CA'):
                return 'Egypt'
            elif t.endswith('.L'):
                return 'UK'
            elif t.endswith('.PA') or t.endswith('.DE') or t.endswith('.MI'):
                return 'Europe'
            elif t.endswith('.T') or t.endswith('.HK'):
                return 'Asia'
            else:
                return 'US'

        # Map tickers to regions and compute regional weights
        _ticker_regions = {r['ticker']: _get_region(r['ticker']) for r in rows}
        _region_value = {}
        for r in rows:
            _rg = _ticker_regions[r['ticker']]
            _region_value[_rg] = _region_value.get(_rg, 0.0) + r['pos_val']

        _region_weights_str = " | ".join(
            f"{rg} {(val/total_value*100):.1f}%"
            for rg, val in sorted(_region_value.items(), key=lambda x: -x[1])
        ) if total_value > 0 else ""

        # ── Investor profile: detect home region & MENA intentionality ────────
        _mena_regions = {'UAE', 'Saudi Arabia', 'Kuwait', 'Qatar', 'Egypt'}
        _gcc_regions  = {'UAE', 'Saudi Arabia', 'Kuwait', 'Qatar'}
        _mena_weight  = sum(_region_value.get(rg, 0) for rg in _mena_regions) / total_value * 100 if total_value > 0 else 0
        _gcc_weight   = sum(_region_value.get(rg, 0) for rg in _gcc_regions)  / total_value * 100 if total_value > 0 else 0
        _is_gcc_investor = _gcc_weight >= 20

        if _mena_weight >= 40:
            _investor_profile      = f"GCC/MENA-focused investor ({_mena_weight:.0f}% in MENA, {_gcc_weight:.0f}% in GCC). This regional exposure is intentional — respect it."
            _investor_profile_icon = "🌍 GCC/MENA-Focused"
        elif _mena_weight >= 20:
            _investor_profile      = f"Diversified investor with meaningful GCC/MENA exposure ({_mena_weight:.0f}% in MENA). Respect this intentional regional allocation."
            _investor_profile_icon = "🌍 Diversified + MENA"
        else:
            _investor_profile      = "Global investor with limited MENA exposure."
            _investor_profile_icon = "🌐 Global"

        # ── Compute specific calendar dates for execution timeline ─────────────
        from datetime import timedelta
        _today_dt     = datetime.now()
        # Skip weekends for trading days
        def _next_trading_date(dt, days_ahead):
            d = dt + timedelta(days=days_ahead)
            while d.weekday() >= 5:   # Saturday=5, Sunday=6
                d += timedelta(days=1)
            return d.strftime('%A, %b %d')

        _date_immediate  = _next_trading_date(_today_dt, 1)   # next trading day
        _date_this_week  = _next_trading_date(_today_dt, 4)   # ~end of week
        _date_next_review = (_today_dt + timedelta(days=30)).strftime('%B %d, %Y')

        # ── Replacement universe ───────────────────────────────────────────────
        # GCC-investor rule: within GCC markets (UAE/SA/KW/QA), cross-exchange
        # replacements are acceptable — they're all home-region for this investor.
        _GCC_COMBINED = (
            'EMAAR.DU, FAB.DU, ADNOCGAS.DU, ADNOCDIST.DU, DIB.DU, ENBD.DU, ALDAR.DU, DEWA.DU, TAQA.DU | '
            '2222.SR (Aramco), 1120.SR (Al-Rajhi), 2010.SR (SABIC), 2380.SR (Petrochem) | '
            'QNBK.QA, ORDS.QA | NBK.KW, KFH.KW'
        )
        _REGION_UNIVERSE = {
            'US':           'AAPL, MSFT, GOOGL, AMZN, NVDA, META, BRK-B, JPM, V, JNJ, XOM, UNH, LLY, HD, PG',
            'UAE':          _GCC_COMBINED if _is_gcc_investor else 'EMAAR.DU, FAB.DU, ADNOCGAS.DU, ADNOCDIST.DU, DIB.DU, ENBD.DU, ALDAR.DU, DEWA.DU, TAQA.DU',
            'Saudi Arabia': _GCC_COMBINED if _is_gcc_investor else '2222.SR (Aramco), 1120.SR (Al-Rajhi), 2010.SR (SABIC), 2380.SR (Petrochem)',
            'Kuwait':       _GCC_COMBINED if _is_gcc_investor else 'NBK.KW, ZAIN.KW, KFH.KW, HUMANSOFT.KW, AGILITY.KW',
            'Qatar':        _GCC_COMBINED if _is_gcc_investor else 'QNBK.QA, ORDS.QA, QIIB.QA, MARK.QA',
            'Egypt':        'COMI.CA, HRHO.CA, EAST.CA, SWDY.CA, EKHO.CA, ABUK.CA',
            'Crypto':       'BTC, ETH, SOL, BNB, AVAX, LINK',
            'UK':           'BP.L, SHEL.L, AZN.L, HSBA.L, ULVR.L',
            'Europe':       'ASML.AS, SAP.DE, LVMH.PA, NESN.SW, ROG.SW',
            'Asia':         '9984.T (SoftBank), 7203.T (Toyota), 0700.HK (Tencent), 9988.HK (Alibaba)',
        }

        # Per-ticker region + universe line for prompt
        _ticker_region_lines = "\n".join(
            f"  {r['ticker']} → {_ticker_regions[r['ticker']]} "
            f"(weight {r['pos_val']/total_value*100:.1f}%)"
            for r in sorted(rows, key=lambda x: -x['pos_val'])
        ) if total_value > 0 else ""

        _universe_lines = "\n".join(
            f"  {rg}: {names}"
            for rg, names in _REGION_UNIVERSE.items()
            if rg in _ticker_regions.values()
        )

        # ── Investor profile card (shown in the report itself) ────────────────
        _profile_card = f"""\n## 👤 Investor Profile\n\n| Field | Value |\n|-------|-------|\n| Profile | {_investor_profile_icon} |\n| MENA Exposure | {_mena_weight:.0f}% of portfolio |\n| GCC Exposure | {_gcc_weight:.0f}% of portfolio |\n| Cross-GCC replacements | {'✅ Allowed (same home region)' if _is_gcc_investor else '⛔ Not applicable'} |\n\n---\n"""

        # ── Send to DeepSeek for CIO recommendation ────────────────────────────
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        cio_section = ""
        if ds_key:
            # Build holdings summary WITHOUT cost basis in a way that prevents yield-on-cost calc
            def _fmt_shares(n):
                return f"{n:g}" if n != int(n) else f"{int(n):,}"
            holdings_summary = "\n".join(
                f"- {r['ticker']} [{_ticker_regions[r['ticker']]}]: {_fmt_shares(r['shares'])} shares | "
                f"current price ${r['curr_px']:.2f} | "
                f"return {r['ret_pct']:+.1f}% | "
                f"CURRENT dividend yield {live_div_yields.get(r['ticker'], 0):.2f}% (= annual div ÷ market price)"
                for r in rows
            )
            weights_summary = " | ".join(
                f"{r['ticker']} {r['pos_val']/total_value*100:.1f}%"
                for r in sorted(rows, key=lambda x: -x['pos_val'])
            ) if total_value > 0 else ""

            prompt = f"""You are EisaX, a CIO-level portfolio strategist. Today is {today}.

⛔ CRITICAL RULES — follow all exactly:
1. NEVER compute or reference "yield on cost". The ONLY valid dividend yield is CURRENT yield (annual dividend ÷ CURRENT market price), already provided below.
2. REGIONAL DISCIPLINE: When recommending to REDUCE, TRIM, or REPLACE any position, you MUST suggest an alternative from the approved universe for that region. Preserve the client's original regional allocation ratio.
3. Do NOT recommend moving money cross-region unless structurally critical — flag explicitly as "⚠️ Regional Shift" with a clear reason.
4. INVESTOR PROFILE: {_investor_profile} Honour this profile. For a GCC investor, UAE and Saudi/Kuwait/Qatar replacements are interchangeable within the same GCC home region.

CLIENT PORTFOLIO:
{holdings_summary}

PORTFOLIO METRICS:
- Total cost basis: ${total_cost:,.0f}
- Current value: ${total_value:,.0f}
- Unrealized P&L: ${total_pnl:+,.0f} ({total_ret_pct:+.1f}%)
- Current weights: {weights_summary}
- Regional allocation: {_region_weights_str}
- Current portfolio dividend yield: {port_div_yield_current:.2f}%

REGIONAL MAP:
{_ticker_region_lines}

APPROVED REPLACEMENT UNIVERSE (suggest ONLY from the matching region below):
{_universe_lines}

STRESS TEST:
- Mild correction (-15%): ${total_value*0.85:,.0f} (${total_value*0.85-total_cost:+,.0f} vs cost)
- Moderate bear (-25%): ${total_value*0.75:,.0f} (${total_value*0.75-total_cost:+,.0f} vs cost)
- Severe crash (-40%): ${total_value*0.60:,.0f} (${total_value*0.60-total_cost:+,.0f} vs cost)

Provide a CIO-grade analysis with EXACTLY these four sections:

## 4. 🎯 CIO Recommendation
Give a clear verdict: HOLD / PARTIAL SELL / BUY MORE / REBALANCE
State target weights for each position after rebalancing.
End this section with one line: **"Projected portfolio yield after rebalancing: X.XX%"** — compute this from the suggested new weights × the yields provided above.

## 5. 💡 Strategic Adjustments
For each change, use this exact format:
"Trim [X]% of [TICKER] → Rotate into [REPLACEMENT-TICKER] — [one-line rationale]"
• Include 2222.SR (Aramco) or other GCC names if the client has GCC exposure and they add diversification value.
• Never suggest cross-region rotation without ⚠️ Regional Shift flag.

## 6. 📅 Execution Plan
List trades in priority order (highest urgency first). For each trade specify:
- **Priority**: Immediate (by {_date_immediate}) / This Week (by {_date_this_week}) / Next Review (by {_date_next_review})
- **Order type**: Limit or Market — and why
- **Timing**: Which part of the trading session (e.g. "first 30 min after open", "avoid last 15 min")

## 7. ⚠️ Risk Flags
2–4 bullet points on concentration, liquidity, or correlation risks.

Be direct, numbers-first, institutional CIO tone. Max 750 words total."""

            try:
                r = _req.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1200, "temperature": 0},
                    timeout=120
                )
                data = r.json()
                cio_section = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if cio_section:
                    logger.info("[CIO] DeepSeek recommendation generated (%d chars)", len(cio_section))
                else:
                    cio_section = "*(CIO recommendation unavailable — DeepSeek returned empty response)*"
            except Exception as e:
                logger.error(f"[CIO] DeepSeek call failed: {e}")
                cio_section = f"*(CIO recommendation unavailable: {e})*"
        else:
            cio_section = "*(CIO recommendation unavailable — DEEPSEEK_API_KEY not set)*"

        _disclaimer = """
---

> ⚠️ **Disclaimer:** This analysis is based on provided cost basis data and live market prices fetched at the time of this request. All prices, returns, and recommendations are for informational purposes only and do not constitute financial advice. Verify all prices independently before execution. Past performance is not indicative of future results.
"""
        # Assemble full report: title → investor profile card → P&L → CIO → disclaimer
        _report_title = f"# 🎯 EisaX CIO Analysis — {today}\n"
        full_reply = _report_title + _profile_card + _report_pnl_block + cio_section + _disclaimer

        # Save as artifact
        state.set_artifact(sid, {
            "type": "cio_analysis",
            "content": full_reply,
            "source": "cio_direct",
            "exportable": True,
            "timestamp": datetime.now()
        })

        return {"type": "chat.reply", "reply": full_reply,
                "data": {"agent": "finance", "analysis_type": "cio_direct"}}


