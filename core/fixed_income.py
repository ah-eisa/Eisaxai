"""fixed_income.py -- thin re-export shim. Public API unchanged."""
from core.fi_routing import (
    extract_isin,
    is_fixed_income_query,
    detect_sukuk_query_language,
    _validate_isin,
    _infer_country_code,
    ISIN_RE,
    VALID_ISIN_PREFIXES,
    _COUNTRY_RATINGS,
    _WGB_COUNTRY_DATA,
    _WGB_COUNTRY_SLUGS,
    _SUKUK_STRUCTURES,
    _HEADERS,
    _cache,
    _CACHE_TTL,
)
from core.fi_fetchers import (
    _fetch_openfigi,
    _serper_isin_lookup,
    _parse_name_components,
    _fetch_fmp_bond,
    _fetch_fred_yield,
    _fetch_benchmarks,
    _fetch_sovereign_cds,
    _fetch_rating_with_date,
    _fetch_market_price_and_ytm,
    _get_fx_rate,
    get_instrument_data,
)
from core.fi_scoring import (
    compute_fi_score,
    format_fi_for_prompt,
)

__all__ = [
    'extract_isin', 'is_fixed_income_query', 'detect_sukuk_query_language',
    'get_instrument_data', 'compute_fi_score', 'format_fi_for_prompt',
]
