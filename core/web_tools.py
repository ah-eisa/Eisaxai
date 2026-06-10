"""
EisaX Web Search Tools
"""
import logging
import os
import requests
logger = logging.getLogger(__name__)

def web_search(query: str, num_results: int = 5) -> dict:
    """
    Perform web search using available APIs
    Falls back to mock data if no API available
    """
    # Try Serper.dev (Google Search API)
    serper_key = os.getenv("SERPER_API_KEY", "")
    
    if serper_key:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key},
                json={"q": query, "num": num_results},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("organic", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                return {"success": True, "results": results}
        except Exception as e:
            logger.error(f"[web_search] Serper failed: {e}")
    # Fallback: Mock data for development
    return {
        "success": True, 
        "results": [
            {"title": "Market Outlook 2026", "link": "https://example.com", "snippet": "Bullish trends expected"},
            {"title": "Investment Strategies", "link": "https://example.com", "snippet": "Diversification recommended"}
        ],
        "note": "Mock data - set SERPER_API_KEY for real results"
    }

