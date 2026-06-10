# Task: Wire Live Market Data into EisaX Agent Responses

## Context
- Pipeline cache: 1037+ Arab stocks (UAE/KSA/EGX/KW/QA) updated every 15 min via `pipeline.py`
- Cache is accessible via: `from pipeline import cache; df, ts = cache.get_latest("uae")`
- Columns: ticker, name, close, change, volume, market_cap_basic, price_earnings_ttm, dividend_yield_recent, RSI, MACD.macd, sector, _market, _snapshot_ts

## Problem
The router routes "أفضل أسهم توزيعات في الإمارات" → FINANCIAL/PORTFOLIO_OPTIMIZE 
instead of a dedicated screener. There is no SCREENER route in the system.

## Two Changes Required

---

### Change 1: Add SCREENER route to ROUTER_PROMPT in core/prompt_manager.py

Read the file first: `cat core/prompt_manager.py`

Find the ROUTES + HANDLERS section in ROUTER_PROMPT and add a new route BEFORE the CRITICAL RULES section:

```
route=SCREENER:
  handler=SCREENER → user wants a ranked list/screening of stocks by a metric
                     TRIGGER SIGNALS: "أفضل أسهم", "best stocks", "top stocks", "أعلى dividend",
                     "highest yield", "أسهم توزيعات", "dividend stocks", "أسهم دفاعية",
                     "defensive stocks", "top performers", "أعلى عائد", "screening", "rank",
                     "قائمة أسهم", "recommend stocks" — WITHOUT specific ticker names
                     MUST NOT TRIGGER: if user provides specific tickers or asks for a portfolio build
```

Also add to the CRITICAL ROUTING RULES section:
```
- "أفضل أسهم"/"best stocks" + metric (dividend/yield/RSI) → SCREENER, never PORTFOLIO_OPTIMIZE
```

And add example routing:
```
"أفضل أسهم توزيعات في الإمارات" → SCREENER/SCREENER
"أعلى dividend yield في السعودية" → SCREENER/SCREENER  
"أسهم دفاعية في السوق المصري" → SCREENER/SCREENER
```

---

### Change 2: Wire SCREENER handler in core/orchestrator.py and core/services/market_route_handler.py

#### 2a. In core/orchestrator.py

Read the file first to understand the routing flow.

Find where routes are handled (look for the big if/elif chain that handles route types like STOCK_ANALYSIS, FINANCIAL, etc.).

Add SCREENER to the valid handlers set (find `_VALID_HANDLERS` dict).
Add SCREENER to `_ROUTE_HANDLER_MAP`.
Add handling: when route == "SCREENER" or handler == "SCREENER", call `handle_screening()` from market_route_handler.

#### 2b. In core/services/market_route_handler.py

The existing `_handle_dividend_screening()` function handles dividend/defensive screening.
Create a new public function `handle_screening()` that:
1. Detects what type of screening from the message (dividend yield, RSI, sector, etc.)
2. Detects market (UAE/KSA/EGX/etc.)
3. Calls the appropriate screener
4. Returns dict with `{"reply": ..., "session_id": ..., "agent_name": "EisaX Market Screener", "model": "SCREENER"}`

The function signature should match other handlers:
```python
async def handle_screening(
    message: str,
    session_id: str,
    user_id: str,
    orchestrator,
    instruction: str = "",
    **kwargs
) -> dict:
```

For screening types to support:
- dividend/توزيعات → sort by dividend_yield_recent DESC
- RSI oversold → sort by RSI ASC (RSI < 35)
- RSI overbought → sort by RSI DESC (RSI > 65)
- top gainers/أعلى ارتفاع → sort by change DESC
- top losers/أعلى انخفاض → sort by change ASC
- sector filter → filter by sector column
- default → sort by dividend_yield_recent DESC

For market detection from message:
- UAE/إمارات/ADX/DFM/دبي/أبوظبي → "uae"
- KSA/سعودية/تداول/ارامكو → "ksa"
- مصر/Egypt/EGX/بورصة → "egypt"
- كويت/Kuwait → "kuwait"
- قطر/Qatar → "qatar"
- default → "uae"

Output format:
```
🏆 {title} — {market_name}
*{count} سهم | آخر تحديث: {age:.0f} دقيقة*

| # | الشركة | الرمز | السعر | {metric} | P/E | RSI |
|---|--------|-------|-------|----------|-----|-----|
| 1 | ... | ... | ... | ... | ... | ... |
...

> المصدر: EisaX Live Cache ({timestamp})
```

---

### Validation
After making changes:
```bash
python3 -m py_compile core/prompt_manager.py
python3 -m py_compile core/orchestrator.py  
python3 -m py_compile core/services/market_route_handler.py
source venv/bin/activate && python3 -c "from core.agents.finance import FinancialAgent; print('IMPORT OK')"
```

Also test intent routing manually:
```bash
source venv/bin/activate && python3 -c "
from core.services.market_route_handler import _detect_market_intent, handle_screening
import asyncio

# Test intent
tests = ['أفضل أسهم توزيعات في الإمارات', 'أعلى dividend yield في السعودية', 'top gainers UAE today']
for t in tests:
    intent = _detect_market_intent(t)
    print(f'intent={intent}: {t[:50]}')
"
```

### Report
List every file and exact line numbers changed.
