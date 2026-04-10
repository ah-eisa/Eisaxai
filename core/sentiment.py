"""
EisaX News Sentiment NLP — G-8
────────────────────────────────
VADER-based sentiment analysis on news headlines/summaries,
linked to tickers and cached in Redis.

Usage
─────
    from core.sentiment import SentimentAnalyzer
    result = SentimentAnalyzer().analyze_ticker("AAPL")
    # {'ticker': 'AAPL', 'score': 0.42, 'label': 'bullish', 'articles': [...]}
"""

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER

logger = logging.getLogger(__name__)

# ── Storage ───────────────────────────────────────────────────────────────────
_DB = Path("/home/ubuntu/investwise/data/sentiment.db")
_DB.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
SENTIMENT_CACHE_TTL = 3600          # 1 hour Redis TTL
SENTIMENT_DB_DAYS   = 30            # keep DB rows for 30 days
BULLISH_THRESHOLD   = 0.10          # compound score ≥ 0.10 → bullish
BEARISH_THRESHOLD   = -0.10         # compound score ≤ -0.10 → bearish

# Finance-domain word boosters (VADER compound adjustments)
_FINANCE_BOOST: dict[str, float] = {
    "beat":        0.15,  "beats":      0.15,
    "surge":       0.20,  "surges":     0.20,
    "record":      0.10,  "upgrade":    0.15,
    "buyback":     0.12,  "dividend":   0.10,
    "rally":       0.15,  "breakout":   0.15,
    "miss":       -0.15,  "misses":    -0.15,
    "downgrade":  -0.15,  "recall":    -0.20,
    "lawsuit":    -0.15,  "fraud":     -0.30,
    "bankruptcy": -0.35,  "default":   -0.30,
    "cut":        -0.10,  "layoff":    -0.15,
    "layoffs":    -0.15,  "crash":     -0.25,
}


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            headline    TEXT NOT NULL,
            source      TEXT,
            score       REAL NOT NULL,
            label       TEXT NOT NULL,
            analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_sent_ticker ON sentiment_cache(ticker);
        CREATE INDEX IF NOT EXISTS idx_sent_time   ON sentiment_cache(analyzed_at);
    """)
    conn.commit()
    return conn


# ── Scorer ────────────────────────────────────────────────────────────────────

class SentimentScorer:
    """Thread-safe VADER scorer with finance-domain boosts."""

    def __init__(self) -> None:
        self._vader = _VADER()

    def score(self, text: str) -> dict:
        """
        Score a single text string.

        Returns
        -------
        {'compound': float, 'pos': float, 'neu': float, 'neg': float, 'label': str}
        """
        if not text or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}

        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]

        # Apply finance-domain boosts
        lower = text.lower()
        boost = sum(v for k, v in _FINANCE_BOOST.items() if k in lower)
        compound = max(-1.0, min(1.0, compound + boost))

        label = (
            "bullish" if compound >= BULLISH_THRESHOLD
            else "bearish" if compound <= BEARISH_THRESHOLD
            else "neutral"
        )
        return {
            "compound": round(compound, 4),
            "pos":      round(scores["pos"], 4),
            "neu":      round(scores["neu"], 4),
            "neg":      round(scores["neg"], 4),
            "label":    label,
        }

    def score_many(self, texts: list[str]) -> list[dict]:
        return [self.score(t) for t in texts]


# ── News fetcher ──────────────────────────────────────────────────────────────

def _fetch_news_for_ticker(ticker: str, max_items: int = 20) -> list[dict]:
    """
    Fetch recent news headlines for a ticker.
    Tries yfinance first, then the internal news engine.
    Returns list of {'headline': str, 'source': str, 'published': str}
    """
    articles: list[dict] = []

    # ── yfinance news ──────────────────────────────────────────────────────
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        for item in raw_news[:max_items]:
            content = item.get("content", {})
            # yfinance ≥ 0.2.x returns nested content dict
            if isinstance(content, dict):
                title = content.get("title", "") or ""
                source = (content.get("provider") or {}).get("displayName", "") or ""
                pub = content.get("pubDate", "") or ""
            else:
                title  = item.get("title", "") or ""
                source = item.get("publisher", "") or ""
                pub    = item.get("providerPublishTime", "") or ""
                if isinstance(pub, (int, float)):
                    pub = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()

            if title:
                articles.append({
                    "headline":  title,
                    "source":    str(source),
                    "published": str(pub),
                })
    except Exception as exc:
        logger.debug("[sentiment] yfinance news fetch failed for %s: %s", ticker, exc)

    # ── Internal EisaX news engine fallback ───────────────────────────────
    if not articles:
        try:
            from db import get_news_for_ticker  # eisax-news internal
            rows = get_news_for_ticker(ticker, limit=max_items)
            for r in rows:
                articles.append({
                    "headline":  r.get("title", ""),
                    "source":    r.get("source", ""),
                    "published": r.get("published_at", ""),
                })
        except Exception:
            pass

    return articles[:max_items]


def _fetch_market_news(max_items: int = 30) -> list[dict]:
    """Fetch general market news (no specific ticker)."""
    articles: list[dict] = []
    try:
        import yfinance as yf
        # SPY / QQQ as market proxies for news
        for proxy in ("SPY", "QQQ", "DIA"):
            raw = yf.Ticker(proxy).news or []
            for item in raw[:10]:
                content = item.get("content", {})
                if isinstance(content, dict):
                    title  = content.get("title", "") or ""
                    source = (content.get("provider") or {}).get("displayName", "") or ""
                    pub    = content.get("pubDate", "") or ""
                else:
                    title  = item.get("title", "") or ""
                    source = item.get("publisher", "") or ""
                    pub    = item.get("providerPublishTime", "") or ""
                if title and title not in {a["headline"] for a in articles}:
                    articles.append({"headline": title, "source": str(source), "published": str(pub)})
            if len(articles) >= max_items:
                break
    except Exception as exc:
        logger.debug("[sentiment] market news fetch failed: %s", exc)
    return articles[:max_items]


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_to_db(ticker: str, articles: list[dict]) -> None:
    """Persist scored articles to SQLite (for history / trend tracking)."""
    try:
        conn = _get_conn()
        now  = datetime.now(timezone.utc).isoformat()
        with conn:
            for a in articles:
                conn.execute(
                    "INSERT OR IGNORE INTO sentiment_cache(ticker,headline,source,score,label,analyzed_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (ticker, a["headline"], a.get("source",""),
                     a["score"], a["label"], now),
                )
            # Prune old rows
            cutoff = (datetime.now(timezone.utc) - timedelta(days=SENTIMENT_DB_DAYS)).isoformat()
            conn.execute("DELETE FROM sentiment_cache WHERE analyzed_at < ?", (cutoff,))
        conn.close()
    except Exception as exc:
        logger.debug("[sentiment] db save failed: %s", exc)


def _load_history(ticker: str, hours: int = 24) -> list[dict]:
    """Load recent sentiment history from SQLite."""
    try:
        conn = _get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            "SELECT headline, source, score, label, analyzed_at"
            " FROM sentiment_cache WHERE ticker=? AND analyzed_at>=?"
            " ORDER BY analyzed_at DESC LIMIT 50",
            (ticker, cutoff),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def _aggregate(scored_articles: list[dict]) -> dict:
    """Compute aggregate stats from a list of scored articles."""
    if not scored_articles:
        return {"score": 0.0, "label": "neutral", "bullish": 0, "bearish": 0, "neutral": 0}

    scores = [a["score"] for a in scored_articles]
    avg    = round(sum(scores) / len(scores), 4)
    label  = (
        "bullish" if avg >= BULLISH_THRESHOLD
        else "bearish" if avg <= BEARISH_THRESHOLD
        else "neutral"
    )
    return {
        "score":   avg,
        "label":   label,
        "bullish": sum(1 for a in scored_articles if a["label"] == "bullish"),
        "bearish": sum(1 for a in scored_articles if a["label"] == "bearish"),
        "neutral": sum(1 for a in scored_articles if a["label"] == "neutral"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """Main entry point for sentiment analysis."""

    def __init__(self) -> None:
        self._scorer = SentimentScorer()

    # ── Redis cache helpers ────────────────────────────────────────────────

    def _cache_key(self, ticker: str) -> str:
        return f"sentiment:{ticker.upper()}"

    def _from_cache(self, ticker: str):
        try:
            from core.redis_store import cache_get
            import json
            raw = cache_get(self._cache_key(ticker))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _to_cache(self, ticker: str, data: dict) -> None:
        try:
            from core.redis_store import cache_set
            import json
            cache_set(self._cache_key(ticker), json.dumps(data), ttl_seconds=SENTIMENT_CACHE_TTL)
        except Exception:
            pass

    # ── Ticker analysis ────────────────────────────────────────────────────

    def analyze_ticker(self, ticker: str, use_cache: bool = True) -> dict:
        """
        Analyze news sentiment for a single ticker.

        Returns
        -------
        {
          ticker, score, label, bullish_count, bearish_count, neutral_count,
          article_count, cached, articles: [{headline, source, score, label}],
          timestamp
        }
        """
        ticker = ticker.upper().strip()

        # Try cache first
        if use_cache:
            cached = self._from_cache(ticker)
            if cached:
                cached["cached"] = True
                return cached

        # Fetch & score fresh
        articles = _fetch_news_for_ticker(ticker)
        scored   = []
        for a in articles:
            s = self._scorer.score(a["headline"])
            scored.append({
                "headline": a["headline"],
                "source":   a.get("source", ""),
                "score":    s["compound"],
                "label":    s["label"],
            })

        agg = _aggregate(scored)

        result = {
            "ticker":        ticker,
            "score":         agg["score"],
            "label":         agg["label"],
            "bullish_count": agg["bullish"],
            "bearish_count": agg["bearish"],
            "neutral_count": agg["neutral"],
            "article_count": len(scored),
            "cached":        False,
            "articles":      scored[:10],   # top 10 in response
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        # Persist & cache
        _save_to_db(ticker, scored)
        self._to_cache(ticker, result)
        return result

    def analyze_many(self, tickers: list[str], use_cache: bool = True) -> list[dict]:
        """Analyze multiple tickers (sequential to avoid rate limits)."""
        results = []
        for t in tickers:
            try:
                results.append(self.analyze_ticker(t, use_cache=use_cache))
            except Exception as exc:
                logger.warning("[sentiment] failed for %s: %s", t, exc)
                results.append({
                    "ticker": t.upper(), "score": 0.0, "label": "neutral",
                    "article_count": 0, "cached": False, "error": str(exc),
                })
        return results

    def market_sentiment(self, use_cache: bool = True) -> dict:
        """
        Aggregate market-wide sentiment from SPY/QQQ/DIA news.

        Returns
        -------
        {score, label, bullish_count, bearish_count, neutral_count,
         article_count, articles, timestamp}
        """
        cache_key = "sentiment:__market__"
        if use_cache:
            try:
                from core.redis_store import cache_get
                import json
                raw = cache_get(cache_key)
                if raw:
                    d = json.loads(raw)
                    d["cached"] = True
                    return d
            except Exception:
                pass

        articles = _fetch_market_news()
        scored   = []
        for a in articles:
            s = self._scorer.score(a["headline"])
            scored.append({
                "headline": a["headline"],
                "source":   a.get("source", ""),
                "score":    s["compound"],
                "label":    s["label"],
            })

        agg = _aggregate(scored)
        result = {
            "scope":         "market",
            "score":         agg["score"],
            "label":         agg["label"],
            "bullish_count": agg["bullish"],
            "bearish_count": agg["bearish"],
            "neutral_count": agg["neutral"],
            "article_count": len(scored),
            "cached":        False,
            "articles":      scored[:15],
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        try:
            from core.redis_store import cache_set
            import json
            cache_set(cache_key, json.dumps(result), ttl_seconds=SENTIMENT_CACHE_TTL)
        except Exception:
            pass

        return result

    def sentiment_trend(self, ticker: str, hours: int = 48) -> dict:
        """
        Return historical sentiment trend from local DB.

        Returns hourly buckets with avg score for the past `hours` hours.
        """
        ticker  = ticker.upper().strip()
        history = _load_history(ticker, hours=hours)

        # Bucket by hour
        buckets: dict[str, list[float]] = {}
        for row in history:
            ts  = row["analyzed_at"][:13]   # "YYYY-MM-DDTHH"
            buckets.setdefault(ts, []).append(row["score"])

        trend = [
            {"hour": h, "avg_score": round(sum(v)/len(v), 4), "count": len(v)}
            for h, v in sorted(buckets.items())
        ]

        return {
            "ticker":     ticker,
            "hours":      hours,
            "data_points": len(history),
            "trend":      trend,
        }
