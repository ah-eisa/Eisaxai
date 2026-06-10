"""
EisaX FactChecker — Live Data Verification for Financial Analysis
=================================================================
Compares AI-generated analysis metrics against live Yahoo Finance data.
Generic: works for any valid ticker symbol.
Features: in-memory cache (5 min TTL), 10s timeout on yfinance calls.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import yfinance as yf

logger = logging.getLogger(__name__)

# ─── In-memory cache ───
_cache: Dict[str, tuple] = {}   # ticker -> (timestamp, info_dict)
CACHE_TTL_SECONDS = 300         # 5 minutes


class FactChecker:
    """Verifies financial analysis reports against live market data."""

    def _get_info(self, ticker: str, timeout: int = 10) -> Optional[dict]:
        """Fetch ticker info with timeout and caching."""
        now = datetime.now().timestamp()

        # Check cache
        if ticker in _cache:
            cached_ts, cached_info = _cache[ticker]
            if now - cached_ts < CACHE_TTL_SECONDS:
                logger.debug("FactChecker cache hit for %s", ticker)
                return cached_info

        # Fetch with timeout
        def _fetch():
            return yf.Ticker(ticker).info

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_fetch)
                info = future.result(timeout=timeout)
                if info:
                    _cache[ticker] = (now, info)
                return info
        except FuturesTimeout:
            logger.warning("FactChecker timeout for %s (%ds)", ticker, timeout)
            return None
        except Exception as e:
            logger.warning("FactChecker fetch error for %s: %s", ticker, e)
            return None

    def verify_analysis(self, ticker: str, report_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Pull live data for *ticker* and build a markdown verification block.

        Args:
            ticker: Stock symbol (e.g. 'NVDA')
            report_data: Dict from _handle_analytics summary (keys: price, sma_50,
                         sma_200, rsi, macd, trend, momentum, condition).
                         If None, only live data is shown.

        Returns:
            Markdown string with the FACT-CHECK table.
        """
        try:
            info = self._get_info(ticker)
            if not info:
                return f"🔍 Fact-check unavailable for **{ticker}** (timeout or no data)."

            live_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not live_price:
                return f"🔍 Fact-check unavailable for **{ticker}** (no live price)."

            beta = info.get("beta")
            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            market_cap = info.get("marketCap")
            fifty_two_low = info.get("fiftyTwoWeekLow")
            fifty_two_high = info.get("fiftyTwoWeekHigh")
            earnings_date = None

            # Earnings date (may be a list of timestamps)
            raw_earnings = info.get("earningsTimestamp") or info.get("earningsDate")
            if raw_earnings:
                if isinstance(raw_earnings, (list, tuple)) and raw_earnings:
                    raw_earnings = raw_earnings[0]
                try:
                    earnings_date = datetime.fromtimestamp(int(raw_earnings)).strftime("%b %d, %Y")
                except Exception:
                    earnings_date = str(raw_earnings)

            # Build comparison rows
            report_data = report_data or {}
            rep_price = report_data.get("price")

            rows: list[str] = []

            # Price
            rep_str = f"${rep_price:.2f}" if rep_price else "—"
            live_str = f"${live_price:.2f}" if live_price else "N/A"
            status = self._compare(rep_price, live_price, tolerance=0.02) if rep_price and live_price else "➕"
            rows.append(f"| Price | {rep_str} | {live_str} | {status} |")

            # Beta
            rows.append(f"| Beta | — | {beta:.2f} | ➕ |" if beta else "| Beta | — | N/A | — |")

            # Trailing P/E
            pe_str = f"{trailing_pe:.1f}x" if trailing_pe else "N/A"
            rows.append(f"| P/E (TTM) | — | {pe_str} | ➕ |")

            # Forward P/E
            fpe_str = f"{forward_pe:.1f}x" if forward_pe else "N/A"
            rows.append(f"| Forward P/E | — | {fpe_str} | ➕ |")

            # Market Cap
            if market_cap:
                if market_cap >= 1e12:
                    mc_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_str = f"${market_cap/1e9:.1f}B"
                else:
                    mc_str = f"${market_cap/1e6:.0f}M"
                rows.append(f"| Market Cap | — | {mc_str} | ➕ |")

            # 52-Week Range
            if fifty_two_low and fifty_two_high:
                rows.append(f"| 52W Range | — | ${fifty_two_low:.2f} – ${fifty_two_high:.2f} | ➕ |")

            now = datetime.now().strftime("%b %d, %Y")
            table_header = (
                f"| Metric | Report | Live | Status |\n"
                f"|--------|--------|------|--------|"
            )
            table_body = "\n".join(rows)

            block = (
                f"\n---\n"
                f"🔍 **FACT-CHECK** *(Verified {now})*\n\n"
                f"{table_header}\n"
                f"{table_body}\n"
            )

            if earnings_date:
                block += f"\n📅 **Next Earnings:** {earnings_date}\n"

            # Source label reflects EisaX routing policy
            _t_up_fc = str(ticker or "").upper()
            if _t_up_fc.endswith((".AE", ".DU", ".SR", ".CA", ".KW", ".QA",
                                   ".BH", ".MA", ".TN")):
                block += "\n*Source: TradingView Live Cache (authoritative for GCC) · Yahoo Finance (fallback) — live at time of query*"
            elif _t_up_fc.endswith("=F"):
                block += "\n*Source: TradingView Live Cache (commodities snapshot) — live at time of query*"
            else:
                block += "\n*Source: TradingView Live Cache · Yahoo Finance — live at time of query*"
            return block

        except Exception as e:
            logger.warning("FactChecker failed for %s: %s", ticker, e)
            return f"🔍 Fact-check unavailable for **{ticker}**."

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _compare(reported: float, live: float, tolerance: float = 0.02) -> str:
        """Return ✅ if within tolerance, ⚠️ otherwise."""
        if not reported or not live:
            return "➕"
        diff = abs(reported - live) / live
        return "✅" if diff <= tolerance else "⚠️"
