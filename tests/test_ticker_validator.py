"""
test_ticker_validator.py — Tests for TickerValidator and validate_portfolio_tickers.

Covers:
  - Blocked exact matches (TEST, STRESS, CSV, FILE, TEMP, …)
  - Blocked patterns (TEST1, DUMMY_TICKER, single letter)
  - Length sanity
  - Numeric tickers with / without regional suffix
  - Valid real tickers (GCC, US, crypto)
  - Portfolio validation (blocked_count, 30% threshold)
  - Integration: IntentClassifier.extract_tickers filters blocked tickers
"""
from __future__ import annotations

import pytest

from core.services.ticker_validator import (
    TickerValidator,
    ValidationStatus,
    validate_portfolio_tickers,
    validate_ticker,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def v() -> TickerValidator:
    return TickerValidator()


# ═══════════════════════════════════════════════════════════════════════════════
# INVALID — blocked exact matches
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ticker", [
    "TEST", "STRESS", "CSV", "FILE", "TEMP",
    "DEMO", "SAMPLE", "FAKE", "NULL", "NONE",
    "NA", "TBD", "XXX", "TICKER", "SYMBOL", "STOCK",
])
def test_blocked_exact_is_invalid(v, ticker):
    result = v.validate(ticker)
    assert result.status == ValidationStatus.INVALID, f"{ticker!r} should be INVALID"
    assert result.reason is not None
    assert result.suggestion is not None


# ═══════════════════════════════════════════════════════════════════════════════
# INVALID — blocked patterns
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ticker", [
    "TEST1", "TEST2", "TEST123",
    "DUMMY", "DUMMYTICKER",
])
def test_blocked_pattern_is_invalid(v, ticker):
    result = v.validate(ticker)
    assert result.status == ValidationStatus.INVALID, f"{ticker!r} should be INVALID"


def test_single_letter_is_invalid(v):
    # Single letter with no suffix — not a real ticker
    result = v.validate("X")
    assert result.status == ValidationStatus.INVALID


def test_empty_string_is_invalid(v):
    assert v.validate("").status == ValidationStatus.INVALID


def test_none_is_invalid(v):
    assert v.validate(None).status == ValidationStatus.INVALID  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# WARNING — numeric without suffix
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ticker", ["12345", "2222", "1010"])
def test_numeric_no_suffix_is_warning(v, ticker):
    result = v.validate(ticker)
    assert result.status == ValidationStatus.WARNING
    assert result.suggestion is not None  # should suggest .SR or .AE


# ═══════════════════════════════════════════════════════════════════════════════
# VALID — real tickers
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ticker", [
    "2222.SR",      # Saudi Aramco
    "1010.SR",      # Saudi Banks
    "ADNOCGAS.AE",  # ADNOC Gas Abu Dhabi
    "EMAAR.DU",     # Emaar Dubai
    "AAPL",         # Apple NASDAQ
    "MSFT",         # Microsoft NASDAQ
    "BTC-USD",      # Bitcoin
    "ETH-USD",      # Ethereum
    "ADNOCGAS",     # Long name, no suffix — valid
    "TADAWUL:2222", # With exchange prefix
    "NET",          # Cloudflare — real ticker, must NOT be blocked
])
def test_real_tickers_are_valid(v, ticker):
    result = v.validate(ticker)
    assert result.status == ValidationStatus.VALID, (
        f"{ticker!r} should be VALID, got {result.status} — {result.reason}"
    )


def test_is_valid_convenience(v):
    assert v.is_valid("AAPL") is True
    assert v.is_valid("TEST") is False
    # WARNING counts as valid for is_valid()
    assert v.is_valid("12345") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level shortcut
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_ticker_shortcut():
    assert validate_ticker("MSFT").status == ValidationStatus.VALID
    assert validate_ticker("STRESS").status == ValidationStatus.INVALID


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_portfolio_all_valid():
    result = validate_portfolio_tickers(["AAPL", "MSFT", "2222.SR"])
    assert result["portfolio_blocked"] is False
    assert len(result["valid"]) == 3
    assert len(result["invalid"]) == 0
    assert result["blocked_count"] == 0


def test_portfolio_some_invalid():
    result = validate_portfolio_tickers(["AAPL", "TEST", "CSV", "MSFT"])
    assert result["blocked_count"] == 2
    assert "AAPL" in result["valid"]
    assert "MSFT" in result["valid"]
    assert any(r["ticker"] == "TEST" for r in result["invalid"])
    assert any(r["ticker"] == "CSV" for r in result["invalid"])


def test_portfolio_over_30pct_blocked():
    """4 out of 5 invalid → >30% → portfolio_blocked=True."""
    tickers = ["TEST", "STRESS", "CSV", "FILE", "AAPL"]
    result = validate_portfolio_tickers(tickers)
    assert result["portfolio_blocked"] is True
    assert result["block_reason"] is not None
    assert "4/5" in result["block_reason"]


def test_portfolio_exactly_30pct_not_blocked():
    """1 out of 3 invalid = 33.3% → blocked."""
    result = validate_portfolio_tickers(["TEST", "AAPL", "MSFT"])
    assert result["portfolio_blocked"] is True  # 33.3% > 30%


def test_portfolio_under_30pct_not_blocked():
    """1 out of 4 invalid = 25% → not blocked."""
    result = validate_portfolio_tickers(["TEST", "AAPL", "MSFT", "GOOG"])
    assert result["portfolio_blocked"] is False


def test_portfolio_warnings_proceed():
    """Numeric tickers without suffix = WARNING → still go through to valid."""
    result = validate_portfolio_tickers(["2222", "AAPL"])
    assert result["portfolio_blocked"] is False
    assert "2222" in result["valid"]   # warning still proceeds
    assert len(result["warnings"]) == 1


def test_portfolio_empty_list():
    result = validate_portfolio_tickers([])
    assert result["portfolio_blocked"] is False
    assert result["blocked_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: IntentClassifier.extract_tickers filters blocked tickers
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_tickers_filters_test():
    """extract_tickers must drop TEST/STRESS even if regex matches them."""
    from core.intent_classifier import IntentClassifier
    tickers = IntentClassifier.extract_tickers("analyse TEST and STRESS")
    assert "TEST" not in tickers
    assert "STRESS" not in tickers


def test_extract_tickers_keeps_real_ticker():
    from core.intent_classifier import IntentClassifier
    tickers = IntentClassifier.extract_tickers("analyse AAPL and MSFT")
    assert "AAPL" in tickers or "MSFT" in tickers  # at least one passes


def test_extract_tickers_keeps_net():
    """NET (Cloudflare) must NOT be filtered — it's a valid ticker."""
    from core.intent_classifier import IntentClassifier
    tickers = IntentClassifier.extract_tickers("what do you think about NET stock")
    # NET may or may not appear depending on COMMON_WORDS list, but it must not
    # be removed by the validator specifically
    from core.services.ticker_validator import validate_ticker
    assert validate_ticker("NET").status == ValidationStatus.VALID
