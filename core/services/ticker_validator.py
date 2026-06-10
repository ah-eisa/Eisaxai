"""
ticker_validator.py — Entry-point guard for invalid/test tickers.

Runs BEFORE any API call or analysis pipeline is invoked.
Adds < 1 ms overhead per request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ValidationStatus(Enum):
    VALID   = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class TickerValidationResult:
    status:     ValidationStatus
    ticker:     str
    reason:     str | None = None
    suggestion: str | None = None


class TickerValidator:
    """
    Deterministic, zero-network ticker validator.

    Design principle: reject clearly wrong input early;
    let the data layer handle ambiguous tickers naturally.
    """

    # ── Exact-match blocklist (base ticker, no suffix/prefix) ─────────────
    BLOCKED_EXACT: frozenset[str] = frozenset({
        "TEST", "STRESS", "CSV", "FILE",
        "TEMP", "DEMO", "SAMPLE", "FAKE",
        "NULL", "NONE", "NA", "N/A", "TBD", "XXX",
        "TICKER", "SYMBOL", "STOCK",
    })

    # NOTE: "NET" (Cloudflare) and "MOCK" are NOT blocked — they are real tickers.
    # Block only strings that can NEVER be a legitimate security.

    # ── Pattern blocklist (applied to the base ticker) ────────────────────
    BLOCKED_PATTERNS: list[str] = [
        r'^TEST\d+$',       # TEST1, TEST2, TEST123
        r'^DUMMY',          # DUMMY, DUMMY_TICKER
        r'^[A-Z]{1}$',      # single letter that isn't a real index prefix
    ]

    # ── Valid GCC/regional suffixes (numeric tickers must have one) ───────
    VALID_SUFFIXES: frozenset[str] = frozenset({
        '.SR', '.AE', '.DU', '.QA', '.KW', '.BH', '.OM',
        '.CA', '.EG',
    })

    # ── Valid exchange prefixes ────────────────────────────────────────────
    VALID_PREFIXES: frozenset[str] = frozenset({
        'TADAWUL:', 'ADX:', 'DFM:', 'QSE:', 'NASDAQ:', 'NYSE:', 'LSE:', 'TSX:',
    })

    def __init__(self) -> None:
        self._compiled = [re.compile(p) for p in self.BLOCKED_PATTERNS]

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, ticker: str) -> TickerValidationResult:
        """Return a TickerValidationResult; raises nothing."""
        if not ticker or not isinstance(ticker, str):
            return TickerValidationResult(
                status=ValidationStatus.INVALID,
                ticker=str(ticker or ""),
                reason="Ticker must be a non-empty string",
            )

        ticker_clean = ticker.strip().upper()

        # Strip known exchange prefix for base analysis; remember if one was present
        base = ticker_clean
        had_exchange_prefix = False
        for pfx in self.VALID_PREFIXES:
            if base.startswith(pfx):
                base = base[len(pfx):]
                had_exchange_prefix = True
                break

        # Strip known regional suffix for base analysis
        suffix = ""
        for sfx in self.VALID_SUFFIXES:
            if base.endswith(sfx):
                suffix = sfx
                base = base[: -len(sfx)]
                break

        # Also strip generic dot-suffix (e.g. ".L", ".PA") for base check
        if not suffix and '.' in base:
            parts = base.rsplit('.', 1)
            base = parts[0]

        # ── Check 1: blocked exact matches ────────────────────────────────
        if base in self.BLOCKED_EXACT:
            return TickerValidationResult(
                status=ValidationStatus.INVALID,
                ticker=ticker,
                reason=f"'{base}' is a reserved/test identifier, not a real security",
                suggestion="Use a real market ticker (e.g. 2222.SR, AAPL, MSFT)",
            )

        # ── Check 2: blocked patterns ─────────────────────────────────────
        for pat in self._compiled:
            if pat.match(base):
                return TickerValidationResult(
                    status=ValidationStatus.INVALID,
                    ticker=ticker,
                    reason=f"Ticker pattern '{base}' matches a test/invalid format",
                    suggestion="Use a real market ticker",
                )

        # ── Check 3: length sanity ─────────────────────────────────────────
        if len(base) < 1 or len(base) > 12:
            return TickerValidationResult(
                status=ValidationStatus.INVALID,
                ticker=ticker,
                reason=f"Base ticker length {len(base)} is outside valid range (1–12)",
            )

        # ── Check 4: pure-numeric base must have a recognised regional suffix
        #    OR a recognised exchange prefix (TADAWUL:2222 is unambiguous)
        if base.isdigit():
            if not suffix and not had_exchange_prefix:
                return TickerValidationResult(
                    status=ValidationStatus.WARNING,
                    ticker=ticker,
                    reason="Numeric ticker without a recognised market suffix",
                    suggestion=f"Did you mean {base}.SR or {base}.AE?",
                )

        return TickerValidationResult(status=ValidationStatus.VALID, ticker=ticker)

    def is_valid(self, ticker: str) -> bool:
        """Convenience boolean; WARNING counts as valid."""
        return self.validate(ticker).status != ValidationStatus.INVALID


# ── Portfolio helper ───────────────────────────────────────────────────────────

def validate_portfolio_tickers(tickers: list[str]) -> dict:
    """
    Validate a list of tickers for portfolio analysis.

    Returns
    -------
    {
      "valid":            list[str],
      "invalid":          list[{"ticker": str, "reason": str}],
      "warnings":         list[{"ticker": str, "reason": str}],
      "blocked_count":    int,
      "portfolio_blocked": bool,   # True when >30% invalid
      "block_reason":     str | None,
    }
    """
    validator = TickerValidator()
    valid:    List[str]  = []
    invalid:  List[dict] = []
    warnings: List[dict] = []

    for t in tickers:
        r = validator.validate(t)
        if r.status == ValidationStatus.VALID:
            valid.append(t)
        elif r.status == ValidationStatus.INVALID:
            invalid.append({"ticker": t, "reason": r.reason})
        else:  # WARNING
            warnings.append({"ticker": t, "reason": r.reason})
            valid.append(t)   # warnings still proceed

    total        = len(tickers)
    blocked      = len(invalid)
    invalid_pct  = blocked / total if total > 0 else 0
    portfolio_blocked = invalid_pct > 0.30

    return {
        "valid":             valid,
        "invalid":           invalid,
        "warnings":          warnings,
        "blocked_count":     blocked,
        "portfolio_blocked": portfolio_blocked,
        "block_reason": (
            f"{blocked}/{total} tickers failed validation"
            if portfolio_blocked else None
        ),
    }


# ── Module-level singleton (import-friendly) ──────────────────────────────────
_default_validator = TickerValidator()


def validate_ticker(ticker: str) -> TickerValidationResult:
    """Module-level shortcut using the shared singleton."""
    return _default_validator.validate(ticker)
