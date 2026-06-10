"""portfolio_manager.py -- thin re-export shim. Public API and import-as-pm pattern unchanged."""
from core.pm_helpers import (
    _kv, get_param, parse_float, parse_int, _fmt_pct, _fmt_float,
    render_weights, detect_risk_pref, recommend_etfs, method_from_risk,
)
from core.pm_tickers import (
    _normalize_tickers, has_placeholder_tickers, get_ticker_name,
    _tv_to_yfinance, get_top_regional_tickers, smart_expand_tickers,
)
from core.pm_reporting import (
    compute_risk_score, render_optimize_reply, render_report,
    build_portfolio_report_body, generate_executive_report_llm,
    _compute_extras, generate_strategy_guide_llm,
)
from core.pm_optimizer import optimize_and_get_data

__all__ = [
    'get_param', 'parse_float', 'parse_int', 'render_weights',
    'detect_risk_pref', 'recommend_etfs', 'method_from_risk',
    '_normalize_tickers', 'has_placeholder_tickers', 'get_ticker_name',
    '_tv_to_yfinance', 'get_top_regional_tickers', 'smart_expand_tickers',
    'compute_risk_score', 'render_optimize_reply', 'render_report',
    'build_portfolio_report_body', 'generate_executive_report_llm',
    '_compute_extras', 'generate_strategy_guide_llm', 'optimize_and_get_data',
]
