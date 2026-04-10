from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


DEFAULT_UNIVERSE: dict[str, list[str]] = {
    "us_large_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V",
        "JNJ", "UNH", "XOM", "PG", "MA", "HD", "CVX", "ABBV", "MRK", "KO",
        "AVGO", "COST", "PEP", "WMT", "BAC", "ORCL", "NFLX", "CRM", "AMD", "DIS",
    ],
    "uae": ["EMAAR.AE", "DIB.AE", "ENBD.AE", "ADNOCDIST.AE", "ALDAR.AE", "FAB.AE", "DIB.AE", "ADIB.AE", "DPW.AE"],
    "egypt": ["COMI.CA", "EKHW.CA", "HRHO.CA", "EKHO.CA", "TMGH.CA"],
    "saudi": ["2222.SR", "1180.SR", "2010.SR", "1120.SR", "2350.SR"],
}


@dataclass(slots=True)
class ScreenerFilter:
    pe_min: float | None = None
    pe_max: float | None = None
    roe_min: float | None = None
    roe_max: float | None = None
    market_cap_min: float | None = None
    market_cap_max: float | None = None
    volume_min: float | None = None
    rsi_min: float | None = None
    rsi_max: float | None = None
    price_above_sma200: bool | None = None
    dividend_yield_min: float | None = None
    revenue_growth_min: float | None = None
    sector: str | None = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _normalize_decimal(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    if number > 1 and number <= 100:
        return number / 100.0
    return number


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _compute_rsi(closes: pd.Series, period: int = 14) -> float | None:
    if closes is None or len(closes) <= period:
        return None
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    last_gain = _safe_float(avg_gain.iloc[-1])
    last_loss = _safe_float(avg_loss.iloc[-1])
    if last_gain is None or last_loss is None:
        return None
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


class StockScreener:
    def screen(
        self,
        tickers: list[str],
        filters: ScreenerFilter,
        max_workers: int = 8,
        include_sentiment: bool = False,
    ) -> list[dict]:
        universe = self._normalize_tickers(tickers)
        if not universe:
            return []
        workers = max(1, min(max_workers, len(universe)))
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._fetch_stock_data, ticker, filters): ticker
                for ticker in universe
            }
            for future in as_completed(future_map):
                ticker = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning("Screener fetch failed for %s: %s", ticker, exc)
                    results.append(self._empty_result(ticker))

        # G-9-A: enrich with sentiment (sequential, uses Redis cache)
        if include_sentiment and results:
            try:
                from core.sentiment import SentimentAnalyzer
                _sa = SentimentAnalyzer()
                for row in results:
                    try:
                        sent = _sa.analyze_ticker(row["ticker"], use_cache=True)
                        row["sentiment_score"]      = sent.get("score")
                        row["sentiment_label"]      = sent.get("label")
                        row["sentiment_confidence"] = sent.get("confidence")
                        row["sentiment_freshness"]  = sent.get("freshness")
                        row["sentiment_articles"]   = sent.get("article_count", 0)
                    except Exception as _se:
                        logger.debug("[screener] sentiment failed for %s: %s", row["ticker"], _se)
                        row["sentiment_score"] = None
                        row["sentiment_label"] = None
            except Exception as exc:
                logger.warning("[screener] sentiment enrichment failed: %s", exc)

        return results

    def _normalize_tickers(self, tickers: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for ticker in tickers or []:
            symbol = (ticker or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    def _fetch_stock_data(self, ticker: str, filters: ScreenerFilter) -> dict:
        stock = yf.Ticker(ticker)
        info = self._safe_info(stock)
        history = self._safe_history(stock)
        close_series = self._extract_series(history, "Close")
        volume_series = self._extract_series(history, "Volume")

        price = _round_or_none(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or (close_series.iloc[-1] if close_series is not None and not close_series.empty else None),
            4,
        )
        market_cap = _safe_float(info.get("marketCap"))
        pe_ratio = _round_or_none(info.get("trailingPE") or info.get("forwardPE"), 4)
        roe = _round_or_none(_normalize_decimal(info.get("returnOnEquity")), 6)
        rsi_14 = _compute_rsi(close_series) if close_series is not None else None
        sma_200 = (
            _round_or_none(close_series.tail(200).mean(), 4)
            if close_series is not None and len(close_series) >= 200
            else None
        )
        above_sma200 = None if price is None or sma_200 is None else price > sma_200
        dividend_yield = _round_or_none(
            info.get("trailingAnnualDividendYield") or info.get("dividendYield"),
            6,
        )
        if dividend_yield is not None and dividend_yield > 1:
            dividend_yield = round(dividend_yield / 100.0, 6)
        revenue_growth = _round_or_none(_normalize_decimal(info.get("revenueGrowth")), 6)
        sector = info.get("sector") or info.get("industry")
        exchange = info.get("fullExchangeName") or info.get("exchange") or info.get("quoteType")
        volume = _safe_float(
            info.get("averageVolume")
            or info.get("averageVolume10days")
            or (volume_series.tail(30).mean() if volume_series is not None and not volume_series.empty else None)
        )
        name = info.get("shortName") or info.get("longName") or info.get("displayName") or ticker

        result = {
            "ticker": ticker,
            "name": name,
            "price": price,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "roe": roe,
            "rsi_14": rsi_14,
            "sma_200": sma_200,
            "above_sma200": above_sma200,
            "dividend_yield": dividend_yield,
            "revenue_growth": revenue_growth,
            "sector": sector,
            "exchange": exchange,
            "volume": int(round(volume)) if volume is not None else None,
        }
        result["score"] = self._score_result(result, filters)
        return result

    def _safe_info(self, stock: yf.Ticker) -> dict:
        try:
            info = stock.info or {}
            return info if isinstance(info, dict) else {}
        except Exception as exc:
            logger.debug("Failed to load info for %s: %s", getattr(stock, "ticker", "?"), exc)
            return {}

    def _safe_history(self, stock: yf.Ticker) -> pd.DataFrame:
        try:
            history = stock.history(period="1y", interval="1d", auto_adjust=False)
            return history if isinstance(history, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            logger.debug("Failed to load history for %s: %s", getattr(stock, "ticker", "?"), exc)
            return pd.DataFrame()

    def _extract_series(self, history: pd.DataFrame, column: str) -> pd.Series | None:
        if history.empty or column not in history.columns:
            return None
        series = history[column].dropna()
        return series if not series.empty else None

    def _score_result(self, result: dict, filters: ScreenerFilter) -> int:
        score = 0

        if filters.pe_min is not None and self._gte(result.get("pe_ratio"), filters.pe_min):
            score += 1
        if filters.pe_max is not None and self._lte(result.get("pe_ratio"), filters.pe_max):
            score += 1
        if filters.roe_min is not None and self._gte(result.get("roe"), filters.roe_min):
            score += 1
        if filters.roe_max is not None and self._lte(result.get("roe"), filters.roe_max):
            score += 1
        if filters.market_cap_min is not None and self._gte(result.get("market_cap"), filters.market_cap_min):
            score += 1
        if filters.market_cap_max is not None and self._lte(result.get("market_cap"), filters.market_cap_max):
            score += 1
        if filters.volume_min is not None and self._gte(result.get("volume"), filters.volume_min):
            score += 1
        if filters.rsi_min is not None and self._gte(result.get("rsi_14"), filters.rsi_min):
            score += 1
        if filters.rsi_max is not None and self._lte(result.get("rsi_14"), filters.rsi_max):
            score += 1
        if filters.price_above_sma200 is not None and result.get("above_sma200") is filters.price_above_sma200:
            score += 1
        if filters.dividend_yield_min is not None and self._gte(result.get("dividend_yield"), filters.dividend_yield_min):
            score += 1
        if filters.revenue_growth_min is not None and self._gte(result.get("revenue_growth"), filters.revenue_growth_min):
            score += 1
        if filters.sector and self._sector_matches(result.get("sector"), filters.sector):
            score += 1

        return score

    def _empty_result(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "name": ticker,
            "price": None,
            "market_cap": None,
            "pe_ratio": None,
            "roe": None,
            "rsi_14": None,
            "sma_200": None,
            "above_sma200": None,
            "dividend_yield": None,
            "revenue_growth": None,
            "sector": None,
            "exchange": None,
            "volume": None,
            "score": 0,
        }

    def _gte(self, value: Any, threshold: float) -> bool:
        number = _safe_float(value)
        return number is not None and number >= threshold

    def _lte(self, value: Any, threshold: float) -> bool:
        number = _safe_float(value)
        return number is not None and number <= threshold

    def _sector_matches(self, value: Any, query: str) -> bool:
        if value is None:
            return False
        return str(query).strip().lower() in str(value).strip().lower()
