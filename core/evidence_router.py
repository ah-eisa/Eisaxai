from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("eisax.evidence_router")

SECTION_FIELDS = [
    "peer_comparison",
    "valuation_scenarios",
    "analyst_consensus",
    "dcf_valuation",
    "technical_confirmation",
    "catalyst_section",
    "scenario_table",
    "dividend_analysis",
    "full_fundamental",
    "cross_market_context",
]

FUNDAMENTAL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "pe_ratio": ("pe_ratio", "pe_ttm", "trailing_pe"),
    "eps": ("eps", "eps_ttm", "trailing_eps"),
    "beta": ("beta",),
    "revenue": ("revenue", "total_revenue"),
    "net_margin": ("net_margin", "profit_margin"),
    "gross_margin": ("gross_margin",),
    "roe": ("roe", "return_on_equity"),
    "debt_equity": ("debt_equity", "debt_to_equity"),
    "ebitda": ("ebitda",),
    "free_cash_flow": ("free_cash_flow", "free_cf"),
}

EVENT_HINT_KEYS = (
    "scheduled_event",
    "upcoming_event",
    "next_event",
    "event_date",
    "next_earnings",
    "next_earnings_date",
    "earnings_date",
)

UNAVAILABLE_STRINGS = {"", "n/a", "na", "none", "null", "nan", "tbd", "unknown"}


@dataclass
class SectionAllowList:
    peer_comparison: bool = False
    valuation_scenarios: bool = False
    analyst_consensus: bool = False
    dcf_valuation: bool = False
    technical_confirmation: bool = False
    catalyst_section: bool = False
    scenario_table: bool = False
    dividend_analysis: bool = False
    full_fundamental: bool = False
    cross_market_context: bool = False

    # Diagnostics
    reasons: dict[str, str] = field(default_factory=dict)  # section → why it was disabled

    def enabled(self) -> list[str]:
        """Return list of section names that are enabled."""
        return [name for name in SECTION_FIELDS if getattr(self, name)]

    def disabled(self) -> list[str]:
        """Return list of section names that are disabled."""
        return [name for name in SECTION_FIELDS if not getattr(self, name)]

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in SECTION_FIELDS} | {
            "reasons": dict(self.reasons)
        }


def _as_dict(value: dict | None) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: list | None) -> list:
    return value if isinstance(value, list) else []


def _safe_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "")
        if cleaned.lower() in UNAVAILABLE_STRINGS:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _is_non_empty_string(value) -> bool:
    return isinstance(value, str) and value.strip().lower() not in UNAVAILABLE_STRINGS


def _has_value(value, *, allow_zero: bool = True) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in UNAVAILABLE_STRINGS:
            return False
        numeric = _safe_float(stripped)
        if numeric is not None:
            return allow_zero or numeric != 0
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    numeric = _safe_float(value)
    if numeric is not None:
        return allow_zero or numeric != 0
    return True


def _lookup(sources: list[dict], *keys: str):
    for source in sources:
        for key in keys:
            if key in source:
                value = source.get(key)
                if value is not None:
                    return value
    return None


def _has_key(sources: list[dict], *keys: str) -> bool:
    return any(key in source for source in sources for key in keys)


def _peer_has_pe(peer) -> bool:
    peer_dict = _as_dict(peer)
    return _has_value(
        _lookup([peer_dict], "pe_ratio", "pe", "pe_ttm", "trailing_pe"),
        allow_zero=False,
    )


def _set_section(
    allow: SectionAllowList,
    section: str,
    enabled: bool,
    ticker: str,
    reason: str = "",
) -> None:
    setattr(allow, section, enabled)
    if enabled:
        allow.reasons.pop(section, None)
        logger.info("[EvidenceRouter] %s: %s enabled", ticker, section)
        return
    allow.reasons[section] = reason
    logger.info("[EvidenceRouter] %s: %s disabled (%s)", ticker, section, reason)


def route_evidence(
    fund: dict,
    scorecard: dict,
    summary: dict | None = None,
    peers: list | None = None,
    analyst_data: dict | None = None,
    ticker: str = "",
) -> SectionAllowList:
    """
    Apply deterministic eligibility rules to decide which report sections may render.
    Logs each routing decision.
    """
    fund = _as_dict(fund)
    scorecard = _as_dict(scorecard)
    summary = _as_dict(summary)
    analyst_data = _as_dict(analyst_data)
    peers = _as_list(peers)

    fundamental_sources = [fund, scorecard]
    technical_sources = [summary, scorecard]
    analyst_sources = [analyst_data, fund, scorecard]
    allow = SectionAllowList()

    populated_fundamentals = 0
    for aliases in FUNDAMENTAL_FIELD_ALIASES.values():
        if _has_value(_lookup(fundamental_sources, *aliases), allow_zero=False):
            populated_fundamentals += 1

    if populated_fundamentals >= 6:
        _set_section(allow, "full_fundamental", True, ticker)
    else:
        _set_section(
            allow,
            "full_fundamental",
            False,
            ticker,
            f"only {populated_fundamentals}/10 core fundamental fields available",
        )

    peers_with_pe = sum(1 for peer in peers if _peer_has_pe(peer))
    if len(peers) >= 4 and peers_with_pe >= 3:
        _set_section(allow, "peer_comparison", True, ticker)
    else:
        _set_section(
            allow,
            "peer_comparison",
            False,
            ticker,
            f"need >=4 peers and >=3 with P/E; got {len(peers)} peers and {peers_with_pe} with P/E",
        )

    forward_eps = _lookup(fundamental_sources, "forward_eps", "eps_forward")
    analyst_target = _lookup(
        fundamental_sources,
        "analyst_target",
        "analyst_target_price",
    )
    analyst_target_value = _safe_float(analyst_target)

    has_forward_eps = _has_value(forward_eps)
    has_analyst_target = analyst_target_value is not None and analyst_target_value > 0
    if has_forward_eps and has_analyst_target:
        _set_section(allow, "valuation_scenarios", True, ticker)
    else:
        missing = []
        if not has_forward_eps:
            missing.append("forward_eps")
        if not has_analyst_target:
            missing.append("analyst_target>0")
        _set_section(
            allow,
            "valuation_scenarios",
            False,
            ticker,
            "missing " + ", ".join(missing),
        )

    growth_value = _lookup(fundamental_sources, "earnings_growth", "revenue_growth")
    if has_forward_eps and has_analyst_target and _has_value(growth_value):
        _set_section(allow, "scenario_table", True, ticker)
    else:
        missing = []
        if not has_forward_eps:
            missing.append("forward_eps")
        if not has_analyst_target:
            missing.append("analyst_target>0")
        if not _has_value(growth_value):
            missing.append("earnings_growth_or_revenue_growth")
        _set_section(
            allow,
            "scenario_table",
            False,
            ticker,
            "missing " + ", ".join(missing),
        )

    analyst_count = _safe_float(_lookup(analyst_sources, "analyst_count"))
    dc_consensus = _lookup(analyst_sources, "dc_consensus")
    if (analyst_count is not None and analyst_count >= 3) or _is_non_empty_string(dc_consensus):
        _set_section(allow, "analyst_consensus", True, ticker)
    else:
        _set_section(
            allow,
            "analyst_consensus",
            False,
            ticker,
            "need analyst_count >= 3 or non-empty dc_consensus",
        )

    free_cash_flow = _lookup(fundamental_sources, "free_cash_flow", "free_cf")
    revenue_growth = _lookup(fundamental_sources, "revenue_growth")
    history_hint = _has_key(fundamental_sources, "5y") or _has_value(revenue_growth)
    if _has_value(free_cash_flow) and _has_value(revenue_growth) and history_hint:
        _set_section(allow, "dcf_valuation", True, ticker)
    else:
        missing = []
        if not _has_value(free_cash_flow):
            missing.append("free_cash_flow")
        if not _has_value(revenue_growth):
            missing.append("revenue_growth")
        if not history_hint:
            missing.append("5y_history_hint")
        _set_section(
            allow,
            "dcf_valuation",
            False,
            ticker,
            "missing " + ", ".join(missing),
        )

    adx_value = _safe_float(_lookup(technical_sources, "adx"))
    if adx_value is not None and adx_value >= 20:
        _set_section(allow, "technical_confirmation", True, ticker)
    else:
        _set_section(
            allow,
            "technical_confirmation",
            False,
            ticker,
            "ADX below 20 or unavailable",
        )

    next_earnings = _lookup(analyst_sources, "next_earnings", "next_earnings_date", "earnings_date")
    has_event_hint = _is_non_empty_string(next_earnings) or any(
        _has_value(_lookup(fundamental_sources, key)) for key in EVENT_HINT_KEYS
    )
    if has_event_hint:
        _set_section(allow, "catalyst_section", True, ticker)
    else:
        _set_section(
            allow,
            "catalyst_section",
            False,
            ticker,
            "no next_earnings or scheduled-event hint",
        )

    div_yield = _safe_float(_lookup(fundamental_sources, "div_yield", "div_yield_pct", "dividend_yield"))
    payout_ratio = _lookup(fundamental_sources, "payout_ratio")
    dividend_rate = _lookup(fundamental_sources, "dividend_rate")
    if div_yield is not None and div_yield > 0 and (
        _has_value(payout_ratio) or _has_value(dividend_rate)
    ):
        _set_section(allow, "dividend_analysis", True, ticker)
    else:
        _set_section(
            allow,
            "dividend_analysis",
            False,
            ticker,
            "need div_yield > 0 and payout_ratio or dividend_rate",
        )

    if allow.full_fundamental:
        _set_section(allow, "cross_market_context", True, ticker)
    else:
        _set_section(
            allow,
            "cross_market_context",
            False,
            ticker,
            "requires full_fundamental eligibility",
        )

    logger.info(
        "[EvidenceRouter] %s: enabled=%s disabled=%s",
        ticker,
        allow.enabled(),
        allow.disabled(),
    )
    return allow


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rich_fund = {
        "pe_ratio": 24.8,
        "eps": 6.1,
        "beta": 1.12,
        "revenue": 124_000_000_000,
        "net_margin": 21.4,
        "gross_margin": 46.2,
        "roe": 18.9,
        "debt_equity": 42.0,
        "ebitda": 38_500_000_000,
        "free_cash_flow": 16_800_000_000,
        "forward_eps": 7.05,
        "analyst_target": 212.0,
        "revenue_growth": 11.5,
        "earnings_growth": 13.0,
        "div_yield": 1.2,
        "payout_ratio": 28.0,
        "5y": True,
        "scheduled_event": "Capital markets day",
    }
    rich_summary = {"adx": 28.4}
    rich_peers = [
        {"ticker": "MSFT", "pe_ratio": 34.0},
        {"ticker": "GOOGL", "pe_ratio": 26.0},
        {"ticker": "META", "pe_ratio": 29.0},
        {"ticker": "AMZN", "pe_ratio": 44.0},
    ]
    rich_analyst = {"analyst_count": 18, "dc_consensus": "Moderate Buy", "next_earnings": "2026-07-24"}

    adx_low_data_fund = {
        "pe_ratio": 9.8,
        "beta": 0.84,
        "eps": 1.43,
        "sector": "Utilities",
    }

    crypto_like_fund = {"price": 108_500.0, "symbol": "BTC-USD"}
    crypto_like_summary = {"sma50": 103_200.0, "sma200": 91_000.0}

    scenarios = [
        (
            "1. Rich-data US stock",
            route_evidence(
                fund=rich_fund,
                scorecard={},
                summary=rich_summary,
                peers=rich_peers,
                analyst_data=rich_analyst,
                ticker="NVDA",
            ),
        ),
        (
            "2. ADX low-data",
            route_evidence(
                fund=adx_low_data_fund,
                scorecard={},
                summary={},
                peers=[],
                analyst_data={},
                ticker="UTILITY.AE",
            ),
        ),
        (
            "3. Crypto-like",
            route_evidence(
                fund=crypto_like_fund,
                scorecard={},
                summary=crypto_like_summary,
                peers=[],
                analyst_data={},
                ticker="BTC-USD",
            ),
        ),
    ]

    for title, allow_list in scenarios:
        print(title)
        print(allow_list.to_dict())
