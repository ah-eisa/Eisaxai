"""
finance_helpers.py
==================
Pure utility functions and static helpers extracted from core/agents/finance.py.

These have **no dependency on FinancialAgent instance state** — they are pure
functions or thin wrappers around standard libraries.  Keeping them here:

* reduces finance.py by ~360 lines
* makes them independently testable
* lets other modules import them without pulling in the full FinancialAgent class

Backward compatibility: finance.py re-exports everything from this module, and
the FinancialAgent class exposes the static methods as staticmethod aliases, so
all existing call-sites (self._method() / FinancialAgent._method()) continue to
work unchanged.
"""

from __future__ import annotations

import logging
import math
import re as _re
import time as _time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Verdict tier mapping ──────────────────────────────────────────────────────
_VERDICT_TIERS: dict[str, int] = {
    'SELL': 0,
    'AVOID': 0,
    'REDUCE': 1,
    'HOLD': 2,
    'ACCUMULATE': 3,
    'BUY': 3,
    'STRONG BUY': 4,
}


# ── Dividend yield parser ─────────────────────────────────────────────────────

def _safe_div_yield(v) -> float:
    """
    Parse dividend yield from any format yfinance/FMP might return:
      '5.00%'  → 0.05    (string with %)
      '509.00%'→ 0.0     (corrupt data — cap at 30%)
      0.05     → 0.05    (already decimal)
      5.0      → 0.05    (whole-number percentage)
    Returns decimal in [0.0, 0.30].
    """
    if not v:
        return 0.0
    try:
        if isinstance(v, str):
            cleaned = v.replace('%', '').replace(',', '').strip()
            val = float(cleaned)
            val_dec = val / 100.0
        else:
            val = float(v)
            val_dec = val / 100.0 if val > 1.0 else val
        if math.isnan(val_dec) or math.isinf(val_dec):
            return 0.0
        return max(0.0, min(val_dec, 0.30))
    except (ValueError, TypeError):
        return 0.0


# ── Analyst consensus divergence ─────────────────────────────────────────────

def _consensus_divergence(
    eisax_verdict: str,
    analyst_consensus: str,
    adx: Optional[float] = None,
    beta: Optional[float] = None,
) -> dict:
    """
    Returns divergence info between EisaX verdict and analyst consensus.
    analyst_consensus examples: 'Strong Buy', 'Buy', 'Hold', 'Underperform', 'Sell'
    adx / beta: optional — when provided, adds a "Driven by:" numeric reason line.
    Returns: {diverges: bool, gap: int, direction: str, message: str}
    """
    if not analyst_consensus or not eisax_verdict:
        return {'diverges': False, 'gap': 0, 'direction': 'none', 'message': ''}

    ac = analyst_consensus.lower().strip()
    if 'strong buy' in ac or 'outperform' in ac:
        cons_tier, cons_label = 4, 'Strong Buy'
    elif 'buy' in ac or 'overweight' in ac:
        cons_tier, cons_label = 3, 'Buy'
    elif 'hold' in ac or 'neutral' in ac or 'market perform' in ac:
        cons_tier, cons_label = 2, 'Hold'
    elif 'underperform' in ac or 'underweight' in ac or 'reduce' in ac:
        cons_tier, cons_label = 1, 'Reduce/Underperform'
    elif 'sell' in ac:
        cons_tier, cons_label = 0, 'Sell'
    else:
        return {'diverges': False, 'gap': 0, 'direction': 'none', 'message': ''}

    eisax_tier = _VERDICT_TIERS.get(eisax_verdict.upper().strip(), 2)
    gap = cons_tier - eisax_tier

    if abs(gap) < 2:
        return {'diverges': False, 'gap': gap, 'direction': 'none', 'message': ''}

    _driven = ""
    if adx is not None and beta is not None:
        _adx_label = (
            "weak trend" if adx < 20 else
            "moderate trend" if adx < 30 else
            "strong trend"
        )
        if gap >= 2:
            _driven = (
                f"\nDriven by: ADX={adx:.1f} ({_adx_label}) + Beta={beta:.2f} (amplified risk) "
                f"vs analyst 12-month horizon ignoring near-term technical damage."
            )
        else:
            _driven = (
                f"\nDriven by: ADX={adx:.1f} ({_adx_label}) + Beta={beta:.2f} "
                f"— EisaX weighting near-term technical recovery signals."
            )

    direction = 'EisaX more bearish' if gap > 0 else 'EisaX more bullish'
    if gap >= 2:
        msg = (
            f'⚠️ **Consensus Divergence:** EisaX rates **{eisax_verdict}** vs analysts **{cons_label}** '
            f'— EisaX is {abs(gap)} tiers more cautious.{_driven}'
        )
    else:
        msg = (
            f'⚠️ **Consensus Divergence:** EisaX rates **{eisax_verdict}** vs analysts **{cons_label}** '
            f'— EisaX is {abs(gap)} tiers more optimistic.{_driven}'
        )
    return {'diverges': True, 'gap': gap, 'direction': direction, 'message': msg}


# ── BTC ETF flow signal ───────────────────────────────────────────────────────

def _fetch_btc_etf_flows() -> str:
    """
    Fetch IBIT (iShares Bitcoin Trust) volume vs 90-day avg.
    Returns a formatted signal string, or '' on any failure (silent fail).
    """
    try:
        import yfinance as _yf_etf
        _ibit = _yf_etf.download("IBIT", period="95d", progress=False, auto_adjust=True)
        if _ibit is None or _ibit.empty or 'Volume' not in _ibit.columns:
            return ""
        _vol_s = _ibit['Volume'].dropna()
        if len(_vol_s) < 30:
            return ""
        _latest = float(_vol_s.iloc[-1])
        _hist   = _vol_s.iloc[:-1]
        _avg90  = float(_hist.tail(90).mean()) if len(_hist) >= 90 else float(_hist.mean())
        if _avg90 <= 0:
            return ""
        _pct = (_latest - _avg90) / _avg90 * 100
        if _pct >= 20:
            return (
                f"📊 ETF Signal: IBIT volume {_pct:+.0f}% above avg "
                f"— institutional accumulation signal"
            )
        elif _pct <= -20:
            return (
                f"📊 ETF Signal: IBIT volume {abs(_pct):.0f}% below avg "
                f"— institutional distribution signal"
            )
        else:
            return (
                f"📊 ETF Signal: IBIT volume normal ({_pct:+.0f}% vs avg) "
                f"— no directional institutional signal"
            )
    except Exception as _etf_e:
        logger.debug("[BTC ETF] IBIT fetch failed: %s", _etf_e)
        return ""


# ── Decision confidence scorer ────────────────────────────────────────────────

def _compute_decision_confidence(
    score: float,
    bullish_count: int,
    bearish_count: int,
    beta: float,
    verdict: str,
) -> int:
    """
    Deterministic confidence score for advisory framing.
    Keeps confidence bounded to avoid false certainty.
    """
    try:
        _score = float(score or 50.0)
        _bull  = int(bullish_count or 0)
        _bear  = int(bearish_count or 0)
        _beta  = float(beta or 1.0)
    except Exception:
        return 55

    signal_gap      = abs(_bull - _bear)
    beta_penalty    = max(0.0, _beta - 1.5) * 8.0
    verdict_penalty = 4.0 if str(verdict or "").upper() in ("SELL", "REDUCE", "AVOID") else 0.0
    raw_conf        = (_score * 0.72) + min(signal_gap * 6.0, 18.0) - beta_penalty - verdict_penalty
    confidence      = int(max(40, min(88, round(raw_conf))))

    if _score < 45:
        lo, hi = 45, 54
    elif _score < 60:
        lo, hi = 45, 60
    elif _score < 70:
        lo, hi = 55, 65
    elif _score < 80:
        lo, hi = 66, 80
    else:
        lo, hi = 81, 90

    confidence = int(max(lo, min(hi, confidence)))
    if _score >= 60 and _score < 70 and confidence < 55:
        confidence = 60
    return confidence


# ── Execution language softener ───────────────────────────────────────────────

def _soften_execution_language(text: str) -> str:
    """Safety net: tone down command-style execution phrasing to advisory phrasing."""
    if not text:
        return text
    softened = str(text)
    replacements = [
        (r'(?i)\bexit\s+100%\b', 'a close below the key level would weaken the technical thesis'),
        (r'(?i)\breduce\s+to\s+50%\b', 'elevated concentration increases risk/reward asymmetry'),
        (r'(?i)\binitiate\s+a\s+position\b', 'a confirmed close above the trigger level would strengthen the bull case'),
        (r'(?i)\badd\s+the\s+remaining\s+tranche\b', 'a confirmed close above the trigger level would strengthen the bull case'),
        (r'(?i)\bbuy\s+now\b', 'the current setup remains data-dependent and confirmation-sensitive'),
        (r'(?i)do\s+\*\*?not\*\*?\s+chase[^.]*\.?', 'current price is elevated relative to the identified entry zone'),
        (r'(?i)wait\s+for\s+a\s+pullback\s+before[^.]*\.?', 'a retracement toward the entry zone would improve the risk/reward profile'),
        (r'(?i)before\s+initiating\s+a\s+position', 'relative to the defined risk parameters'),
        (r'(?i)before\s+entering', 'relative to current risk parameters'),
    ]
    for pattern, repl in replacements:
        softened = _re.sub(pattern, repl, softened)
    return softened


# ── Scenario price rounder ────────────────────────────────────────────────────

def _round_scenario_prices(text: str, currency_sym: str = "$") -> str:
    """
    Round exact decimal prices in scenario/valuation TABLE CELLS ONLY to ranges.
    Targets only markdown table rows — never touches live price, SMA, RSI, etc.
    Example: ﷼24.96 → ~24.5–25.5 ﷼
    """
    if not text:
        return text

    def _to_range(m):
        raw = m.group(0)
        num_str = _re.search(r'[\d,]+\.?\d*', raw.replace(',', ''))
        if not num_str:
            return raw
        try:
            val = float(num_str.group().replace(',', ''))
        except ValueError:
            return raw
        if val < 1 or val > 100_000:
            return raw
        step = max(0.5, round(val * 0.025 * 2) / 2)
        lo = round((val - step) / step) * step
        hi = round((val + step) / step) * step
        if val >= 1000:
            return f"~{lo:,.0f}–{hi:,.0f} {currency_sym}".strip()
        if val >= 10:
            return f"~{lo:.1f}–{hi:.1f} {currency_sym}".strip()
        return f"~{lo:.2f}–{hi:.2f} {currency_sym}".strip()

    lines = text.split('\n')
    result = []
    in_scenario_section = False

    for line in lines:
        if any(kw in line for kw in ['Scenario', 'Bear', 'Bull', 'Base', 'Impact', 'Expected Price', 'Implied Price']):
            in_scenario_section = True

        if in_scenario_section and line.startswith('|'):
            line = _re.sub(
                r'(?<!~)(?:[\$﷼€£]\s?\d{1,6}(?:,\d{3})*\.\d{2}|SAR\s?\d{1,6}(?:,\d{3})*\.\d{2})(?!\d)',
                _to_range,
                line,
            )
        elif in_scenario_section and not line.startswith('|') and line.strip() and not line.startswith('>'):
            in_scenario_section = False

        result.append(line)

    return '\n'.join(result)


# ── On-chain data fetcher ─────────────────────────────────────────────────────

def _fetch_onchain(ticker: str) -> dict:
    """Fetch on-chain metrics for crypto assets. Free APIs only."""
    import requests as _rq
    out: dict = {}
    if not (ticker.endswith('-USD') and any(c in ticker for c in ['BTC', 'ETH', 'SOL', 'XRP'])):
        return out

    _cg_map = {
        'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana',
        'XRP-USD': 'ripple', 'BNB-USD': 'binancecoin', 'DOGE-USD': 'dogecoin',
        'ADA-USD': 'cardano', 'AVAX-USD': 'avalanche-2', 'LINK-USD': 'chainlink',
        'DOT-USD': 'polkadot',
    }
    cg_id = _cg_map.get(ticker)

    # ── 1. CoinGecko: ATH, supply, volume, rank ──────────────────────────────
    if cg_id:
        try:
            r = _rq.get(
                f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                params={"localization": "false", "tickers": "false",
                        "community_data": "false", "developer_data": "false"},
                timeout=8,
            )
            if r.status_code == 200:
                d = r.json()
                md = d.get('market_data', {})
                out['ath']               = md.get('ath', {}).get('usd') or 0
                out['ath_change_pct']    = md.get('ath_change_percentage', {}).get('usd') or 0
                out['ath_date']          = (md.get('ath_date', {}).get('usd', '') or '')[:10]
                out['circulating_supply'] = md.get('circulating_supply') or 0
                out['max_supply']        = md.get('max_supply') or 0
                out['total_volume_24h']  = md.get('total_volume', {}).get('usd') or 0
                out['mc_rank']           = d.get('market_cap_rank', 0)
                if out['max_supply'] and out['circulating_supply']:
                    out['supply_ratio'] = round(out['circulating_supply'] / out['max_supply'] * 100, 1)
        except Exception as _cg_e:
            logger.warning("[OnChain] CoinGecko failed for %s: %s", ticker, _cg_e)

    # ── 2. Blockchain.com: Hash Rate + Active Addresses (BTC only) ───────────
    if ticker == 'BTC-USD':
        try:
            r2 = _rq.get("https://api.blockchain.info/stats", timeout=8)
            if r2.status_code == 200:
                bs = r2.json()
                out['hash_rate_eh']           = round(bs.get('hash_rate', 0) / 1e9, 1)
                out['n_tx_24h']               = bs.get('n_tx', 0)
                out['minutes_between_blocks'] = round(bs.get('minutes_between_blocks', 0), 1)
        except Exception as _bc_e:
            logger.debug("[OnChain] blockchain.info/stats failed for %s: %s", ticker, _bc_e)
        try:
            r3 = _rq.get(
                "https://api.blockchain.info/charts/n-unique-addresses",
                params={"timespan": "1days", "format": "json"},
                timeout=8,
            )
            if r3.status_code == 200:
                vals = r3.json().get('values', [])
                if vals:
                    out['active_addresses'] = int(vals[-1].get('y', 0))
        except Exception as _ba_e:
            logger.debug("[OnChain] blockchain.info/addresses failed for %s: %s", ticker, _ba_e)

    return out
