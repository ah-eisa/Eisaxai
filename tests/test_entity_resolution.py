from __future__ import annotations

import pytest

import core.services.entity_resolution as er


@pytest.mark.parametrize(
    ("raw_query", "symbol", "market", "asset_type", "currency", "source"),
    [
        ("Analyze NVDA", "NVDA", "USA", "equity", "USD", "exact_ticker"),
        ("Analyze NVIDIA", "NVDA", "USA", "equity", "USD", "universe_exact"),
        ("Analyze Microsoft", "MSFT", "USA", "equity", "USD", "universe_exact"),
        ("Analyze Apple", "AAPL", "USA", "equity", "USD", "universe_exact"),
        ("Analyze 2222.SR", "2222.SR", "SAU", "equity", "SAR", "exact_ticker"),
        ("Analyze Saudi Aramco", "2222.SR", "SAU", "equity", "SAR", "universe_exact"),
        ("Analyze Aramco", "2222.SR", "SAU", "equity", "SAR", "universe_exact"),
        ("Analyze ADNOC Gas", "ADNOCGAS.AE", "UAE", "equity", "AED", "universe_exact"),
        ("Analyze Emaar", "EMAAR.DU", "UAE", "equity", "AED", "universe_exact"),
        ("Analyze ADIB", "ADIB.AE", "UAE", "equity", "AED", "universe_exact"),
        ("Analyze Commercial International Bank", "COMI.CA", "EGY", "equity", "EGP", "universe_exact"),
        ("Analyze Talaat Moustafa", "TMGH.CA", "EGY", "equity", "EGP", "universe_exact"),
        ("Analyze BTC", "BTC", "CRYPTO", "crypto", "USD", "exact_ticker"),
        ("Analyze Bitcoin", "BTC", "CRYPTO", "crypto", "USD", "universe_exact"),
        ("Analyze Ethereum", "ETH", "CRYPTO", "crypto", "USD", "universe_exact"),
    ],
)
def test_resolution_acceptance_cases(
    raw_query: str,
    symbol: str,
    market: str,
    asset_type: str,
    currency: str,
    source: str,
):
    result = er.resolve_asset_entity(raw_query)

    assert result.is_resolved
    assert result.symbol == symbol
    assert result.market == market
    assert result.asset_type == asset_type
    assert result.currency == currency
    assert result.resolution_source == source
    assert result.confidence == "high"


def test_unified_universe_loads_local_and_global_sources():
    universe = er.load_instrument_universe()
    symbols = {instrument.symbol for instrument in universe}

    assert "NVDA" in symbols
    assert "MSFT" in symbols
    assert "2222.SR" in symbols
    assert "COMI.CA" in symbols
    assert "BTC" in symbols


def test_universe_normalized_match_handles_company_suffixes():
    result = er.resolve_asset_entity("Analyze ADNOC Gas PJSC")

    assert result.is_resolved
    assert result.symbol == "ADNOCGAS.AE"
    assert result.resolution_source == "universe_normalized"


def test_universe_source_is_attached_to_local_market_matches():
    result = er.resolve_asset_entity("Analyze Saudi Aramco")

    assert result.is_resolved
    assert result.universe_source == "core.local_tickers.MARKET_DB"


def test_universe_source_is_attached_to_global_matches():
    result = er.resolve_asset_entity("Analyze NVIDIA")

    assert result.is_resolved
    assert result.universe_source == "core.tools.ticker_resolver"


def test_normalize_instrument_name_handles_punctuation_and_suffixes():
    assert er.normalize_instrument_name("ADNOC GAS PJSC") == "adnoc gas"
    assert er.normalize_instrument_name("NVIDIA Corp.") == "nvidia corp"


def test_normalize_lookup_key_handles_arabic_folding():
    assert er.normalize_lookup_key("أَدنــوك غاز") == "ادنوك غاز"
    assert er.normalize_lookup_key("إعمار") == "اعمار"


@pytest.mark.parametrize(
    ("raw_query", "symbol"),
    [
        ("تحليل أرامكو", "2222.SR"),
        ("تحليل أدنوك غاز", "ADNOCGAS.AE"),
        ("تحليل اعمار", "EMAAR.DU"),
        ("تحليل البنك التجاري الدولي", "COMI.CA"),
        ("تحليل بيتكوين", "BTC"),
    ],
)
def test_arabic_resolution_acceptance_cases(raw_query: str, symbol: str):
    result = er.resolve_asset_entity(raw_query)

    assert result.is_resolved
    assert result.symbol == symbol


def test_arabic_ambiguous_family_stays_blocked():
    result = er.resolve_asset_entity("تحليل أدنوك")

    assert not result.is_resolved
    assert result.resolution_status == "ambiguous"


def test_lookup_from_universe_marks_brand_family_as_ambiguous():
    result = er.lookup_from_universe("ADNOC")

    assert result is not None
    assert result.resolution_status == "ambiguous"
    assert {candidate["symbol"] for candidate in result.candidates} == {
        "ADNOCGAS.AE",
        "ADNOCDIST.AE",
        "ADNOCDRILL.AE",
    }


def test_ambiguous_candidates_include_local_name_for_arabic_ui():
    result = er.resolve_asset_entity("تحليل أدنوك")

    assert result.resolution_status == "ambiguous"
    local_names = {candidate.get("local_name") for candidate in result.candidates}
    assert "أدنوك للغاز" in local_names


def test_resolve_asset_entity_keeps_ambiguous_adnoc_blocked():
    result = er.resolve_asset_entity("Analyze ADNOC")

    assert not result.is_resolved
    assert result.resolution_status == "ambiguous"


def test_exact_ticker_with_suffix_is_authoritative():
    result = er.resolve_asset_entity("Analyze ADNOCGAS.AE")

    assert result.is_resolved
    assert result.symbol == "ADNOCGAS.AE"
    assert result.market == "UAE"
    assert result.resolution_source == "exact_ticker"


def test_crypto_pair_resolves_back_to_canonical_crypto_symbol():
    exact = er.is_exact_ticker("BTC-USD")

    assert exact is not None
    assert exact.symbol == "BTC"
    assert exact.market == "CRYPTO"


def test_wrong_market_prevention_for_gcc_name():
    result = er.resolve_asset_entity("Analyze EMAAR")

    assert result.is_resolved
    assert result.symbol == "EMAAR.DU"
    assert result.market == "UAE"
    assert result.resolution_source == "universe_exact"


def test_wrong_market_prevention_for_egypt_name():
    result = er.resolve_asset_entity("Analyze CIB")

    assert result.is_resolved
    assert result.symbol == "COMI.CA"
    assert result.market == "EGY"


def test_wrong_market_prevention_for_us_name():
    result = er.resolve_asset_entity("Analyze Microsoft")

    assert result.is_resolved
    assert result.symbol == "MSFT"
    assert result.market == "USA"


def test_unknown_company_name_stays_unresolved():
    result = er.resolve_asset_entity("Analyze Unknown Holdings")

    assert not result.is_resolved
    assert result.resolution_status == "unresolved"


def test_generic_us_ticker_fallback_remains_available():
    result = er.resolve_asset_entity("Analyze QQQ")

    assert result.is_resolved
    assert result.symbol == "QQQ"
    assert result.market == "USA"
    assert result.asset_type in {"equity", "etf"}


def test_partial_source_failure_keeps_local_universe_working(monkeypatch: pytest.MonkeyPatch):
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    monkeypatch.setattr(
        er,
        "load_global_universe_sources",
        lambda: (
            ("broken", _boom),
            ("local_only", er._load_local_market_universe),
        ),
    )

    result = er.resolve_asset_entity("Analyze Saudi Aramco")

    assert result.is_resolved
    assert result.symbol == "2222.SR"
    assert result.resolution_source == "universe_exact"


def test_partial_source_failure_keeps_global_universe_working(monkeypatch: pytest.MonkeyPatch):
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    monkeypatch.setattr(
        er,
        "load_global_universe_sources",
        lambda: (
            ("broken", _boom),
            ("global_only", er._load_tool_resolver_global_universe),
        ),
    )

    result = er.resolve_asset_entity("Analyze Microsoft")

    assert result.is_resolved
    assert result.symbol == "MSFT"
    assert result.resolution_source == "universe_exact"


def test_total_loader_failure_falls_back_to_phase2_alias_behavior(monkeypatch: pytest.MonkeyPatch):
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    monkeypatch.setattr(
        er,
        "load_global_universe_sources",
        lambda: (
            ("broken_local", _boom),
            ("broken_global", _boom),
        ),
    )

    result = er.resolve_asset_entity("Analyze Emaar")

    assert result.is_resolved
    assert result.symbol == "EMAAR.DU"
    assert result.resolution_source == "alias_map"


def test_total_loader_failure_keeps_ambiguity_safe(monkeypatch: pytest.MonkeyPatch):
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    monkeypatch.setattr(
        er,
        "load_global_universe_sources",
        lambda: (
            ("broken_local", _boom),
            ("broken_global", _boom),
        ),
    )

    result = er.resolve_asset_entity("Analyze ADNOC")

    assert not result.is_resolved
    assert result.resolution_status == "ambiguous"


def test_total_loader_failure_still_supports_us_and_crypto_aliases(monkeypatch: pytest.MonkeyPatch):
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    monkeypatch.setattr(
        er,
        "load_global_universe_sources",
        lambda: (
            ("broken_local", _boom),
            ("broken_global", _boom),
        ),
    )

    nvidia = er.resolve_asset_entity("Analyze NVIDIA")
    bitcoin = er.resolve_asset_entity("Analyze Bitcoin")

    assert nvidia.symbol == "NVDA"
    assert nvidia.resolution_source == "alias_map"
    assert bitcoin.symbol == "BTC"
    assert bitcoin.resolution_source == "alias_map"


@pytest.fixture(autouse=True)
def _reset_caches():
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()
    yield
    er.load_instrument_universe.cache_clear()
    er.build_instrument_index.cache_clear()


def _boom():
    raise RuntimeError("universe unavailable")
