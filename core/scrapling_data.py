"""
Scrapling-based data fetcher - backup for when APIs fail
"""
from scrapling import Fetcher
import logging

logger = logging.getLogger(__name__)
_fetcher = None

def _get_fetcher():
    global _fetcher
    if not _fetcher:
        _fetcher = Fetcher(auto_match=False)
    return _fetcher

def get_price_yahoo(ticker: str) -> dict:
    """جيب السعر من Yahoo Finance مباشرة"""
    try:
        f = _get_fetcher()
        page = f.get(f'https://finance.yahoo.com/quote/{ticker}/', timeout=15)
        
        prices = page.find_all(f'fin-streamer[data-symbol="{ticker}"]')
        result = {}
        for p in prices:
            field = p.attrib.get('data-field', '')
            val = p.attrib.get('data-value') or p.text
            if field and val:
                try:
                    result[field] = float(str(val).replace(',', ''))
                except Exception as _e:
                    result[field] = val
        
        if 'regularMarketPrice' in result:
            return {
                'price': result.get('regularMarketPrice'),
                'change': result.get('regularMarketChange'),
                'change_pct': result.get('regularMarketChangePercent'),
                'source': 'yahoo_scrapling'
            }
        return {}
    except Exception as e:
        logger.warning(f"Scrapling failed for {ticker}: {e}")
        return {}

def get_news_yahoo(ticker: str, max_items: int = 5) -> list:
    """جيب آخر أخبار السهم"""
    try:
        f = _get_fetcher()
        page = f.get(f'https://finance.yahoo.com/quote/{ticker}/news/', timeout=15)
        
        articles = page.find_all('a[data-ylk]')
        news = []
        seen = set()
        for a in articles:
            title = a.text.strip()
            href = a.attrib.get('href', '')
            if title and len(title) > 20 and title not in seen:
                seen.add(title)
                news.append({'title': title, 'url': href})
            if len(news) >= max_items:
                break
        return news
    except Exception as e:
        logger.warning(f"Scrapling news failed for {ticker}: {e}")
        return []

if __name__ == '__main__':
    for t in ['NVDA', 'AAPL', 'MSFT']:
        r = get_price_yahoo(t)
        print(f"{t}: ${r.get('price')} ({r.get('change_pct', 0):.2f}%)")
