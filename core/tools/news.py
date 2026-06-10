"""Tool: search_news — recent news for a company or topic."""
import logging
logger = logging.getLogger(__name__)

def search_news(query: str, limit: int = 5) -> dict:
    """
    Returns recent news headlines relevant to the query.
    Uses Serper → news_aggregator fallback.
    """
    import os
    limit = min(int(limit), 10)
    result = {"query": query, "articles": [], "error": None}

    try:
        # 1. Serper (best quality for MENA + global)
        _serper_key = os.getenv("SERPER_API_KEY", "")
        if _serper_key:
            import requests
            resp = requests.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": _serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": limit},
                timeout=8
            )
            if resp.status_code == 200:
                for item in resp.json().get("news", [])[:limit]:
                    result["articles"].append({
                        "title":  item.get("title", "")[:150],
                        "url":    item.get("link", ""),
                        "source": item.get("source", ""),
                        "date":   item.get("date", ""),
                    })
                if result["articles"]:
                    return result

        # 2. NewsAPI fallback
        _news_key = os.getenv("NEWS_API_KEY", "")
        if _news_key:
            import requests
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "pageSize": limit, "sortBy": "publishedAt",
                        "language": "en", "apiKey": _news_key},
                timeout=8
            )
            for a in resp.json().get("articles", [])[:limit]:
                result["articles"].append({
                    "title":  (a.get("title") or "")[:150],
                    "url":    a.get("url", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                    "date":   a.get("publishedAt", ""),
                })
            if result["articles"]:
                return result

        # 3. EisaX news aggregator last resort
        from core.news_aggregator import get_news as _agg
        for item in _agg(query=query, limit=limit):
            result["articles"].append({
                "title":  item.get("title", "")[:150],
                "url":    item.get("url", ""),
                "source": item.get("source", ""),
                "date":   item.get("publishedAt", ""),
            })

    except Exception as e:
        result["error"] = str(e)
        logger.warning("[Tool:search_news] %s: %s", query, e)

    return result
