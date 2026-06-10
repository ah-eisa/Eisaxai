# Task: Fix UAE Dividend Screening Intent Detection

## Bug
When user asks: "أفضل أسهم توزيعات في الإمارات" or "أعلى dividend yield في السعودية"
The system tries to run the portfolio optimizer and rejects the portfolio instead of returning screening results directly.

## Fix Required

### Step 1 — Explore first
Run these commands to understand the codebase:
```
grep -rn "dividend\|توزيعات\|screener\|screening" core/ --include="*.py" -l
grep -rn "portfolio\|optimizer\|allocat" core/ --include="*.py" -l
grep -rn "market_cache\|cache_timestamp\|live_data\|arab_market" core/ --include="*.py" -l | head -20
grep -rn "PORTFOLIO\|SCREENER\|intent" core/services/routing_service.py | head -30
grep -rn "handle_financial\|DIVIDEND\|SCREEN" core/services/market_route_handler.py | head -30
```

### Step 2 — Find where routing decision is made
Look at core/services/routing_service.py and core/services/market_route_handler.py to understand how intents are classified and routed.

### Step 3 — Find the market cache/library
Look for any in-memory cache or data structure that holds live UAE/KSA/EGX stock data. Check:
- Any global dict or list named cache, market_data, stocks, etc.
- Any function that returns stock lists with div_yield, price, rsi fields
- Check arab_dashboard_fixed.py, global_allocator.py for data structures

### Step 4 — Implement the fix

#### 4a. Add intent detection function
In the appropriate routing file, add a function to detect screening vs portfolio intent:

```python
_LIST_KEYWORDS = [
    "أفضل", "best", "top", "أعلى", "اقترح", "recommend",
    "أسهم توزيعات", "dividend stocks", "dividend yield",
    "أسهم دفاعية", "defensive stocks", "ranking", "قائمة",
    "أعلى عائد", "highest yield", "screen", "فلتر"
]
_PORTFOLIO_KEYWORDS = [
    "محفظة", "portfolio", "optimize", "وزّع", "allocate", 
    "ابني", "build", "construct", "توزيع الأصول"
]

def _detect_market_intent(message: str) -> str:
    """Returns 'screening' or 'portfolio' or 'unknown'"""
    msg_lower = message.lower()
    list_score = sum(1 for kw in _LIST_KEYWORDS if kw in msg_lower)
    port_score = sum(1 for kw in _PORTFOLIO_KEYWORDS if kw in msg_lower)
    if list_score > port_score:
        return "screening"
    elif port_score > 0:
        return "portfolio"
    return "unknown"
```

#### 4b. Add screening handler
Add a function that queries the market cache and returns formatted results:

```python
def _handle_dividend_screening(message: str, market: str = None) -> str:
    """Screen stocks from live cache by dividend yield"""
    # Detect market from message
    if not market:
        if any(w in message for w in ["UAE", "إمارات", "ADX", "DFM", "دبي", "أبوظبي"]):
            market = "UAE"
        elif any(w in message for w in ["KSA", "سعودية", "تداول", "TADAWUL"]):
            market = "KSA"
        elif any(w in message for w in ["مصر", "EGX", "بورصة"]):
            market = "EGX"
        else:
            market = "UAE"  # default
    
    # Find cache - look for the actual cache variable used in the project
    # Try to get live data from whatever cache exists
    stocks = _get_market_stocks_from_cache(market)
    
    if not stocks:
        return "⚠️ بيانات السوق غير متاحة حالياً — جاري التحديث\nحاول تاني خلال دقيقتين"
    
    # Filter and sort by dividend yield
    div_stocks = [s for s in stocks if s.get("div_yield") or s.get("dividend_yield") or s.get("dividendYield")]
    if not div_stocks:
        return "⚠️ لا تتوفر بيانات توزيعات كافية في الوقت الحالي"
    
    # Normalize field names and sort
    def get_yield(s):
        return float(s.get("div_yield") or s.get("dividend_yield") or s.get("dividendYield") or 0)
    
    div_stocks.sort(key=get_yield, reverse=True)
    top = div_stocks[:10]
    
    # Format output table
    market_name = {"UAE": "الإمارات (ADX/DFM)", "KSA": "السعودية (تداول)", "EGX": "مصر (EGX)"}.get(market, market)
    rows = ["| # | الشركة | السوق | السعر | Div Yield | P/E | RSI |",
            "|---|--------|-------|-------|-----------|-----|-----|"]
    for i, s in enumerate(top, 1):
        name = s.get("name") or s.get("ticker") or "N/A"
        exch = s.get("exchange") or s.get("market") or market
        price = s.get("price") or s.get("current_price") or 0
        dy = get_yield(s)
        pe = s.get("pe_ratio") or s.get("pe") or s.get("forwardPE") or "—"
        rsi = s.get("rsi") or s.get("rsi_14") or "—"
        currency = "د.إ" if market == "UAE" else ("ر.س" if market == "KSA" else "ج.م")
        rows.append(f"| {i} | {name} | {exch} | {currency}{float(price):,.2f} | {dy:.2f}% | {pe} | {rsi} |")
    
    import datetime
    now = datetime.datetime.now().strftime("%H:%M")
    table = "\n".join(rows)
    return f"🏆 أفضل أسهم توزيعات في {market_name}\n*بيانات محدّثة — {now}*\n\n{table}\n\n> المصدر: EisaX Market Cache"
```

#### 4c. Wire into routing
In the routing/handler where portfolio-type messages are classified, BEFORE calling the optimizer, check:

```python
intent = _detect_market_intent(message)
if intent == "screening" and any(w in message.lower() for w in ["dividend", "توزيعات", "yield", "defensive", "دفاعية"]):
    result = _handle_dividend_screening(message)
    return result, "SCREENING"
```

### Step 5 — Validate
```
python3 -m py_compile core/services/routing_service.py
python3 -m py_compile core/services/market_route_handler.py
python3 -m py_compile core/agents/finance.py  # if changed
source venv/bin/activate && python3 -c "from core.agents.finance import FinancialAgent; print('IMPORT OK')"
```

### Step 6 — Report
List every file and line number changed.
