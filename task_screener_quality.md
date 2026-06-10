# Task: Fix Screener Data Quality Filters

## Problem
`_build_screening_reply()` in `core/services/market_route_handler.py` returns junk results:
- ORIENT: 150% yield, volume=0, RSI=100 → obvious data anomaly
- Small illiquid stocks appear before major names like EMAAR, TALABAT

## Fix: Add quality pre-filter in `_screen_rows()` or `_build_screening_reply()`

Read the file first: `grep -n "_screen_rows\|_build_screening_reply\|def _screen" core/services/market_route_handler.py`
Then read the relevant section.

### Quality Filter Rules

Add a function `_apply_quality_filter(stocks: list[dict], screening_type: str) -> list[dict]` that filters BEFORE sorting:

```python
def _apply_quality_filter(stocks: list[dict], screening_type: str = "dividend") -> list[dict]:
    """Remove anomalous/illiquid stocks before ranking."""
    out = []
    for s in stocks:
        # 1. Skip zero-volume stocks (data anomaly or halted)
        vol = _to_float(s.get("volume") or 0)
        if vol < 50_000:
            continue
        
        # 2. Skip RSI anomalies (100 = data error, < 5 = data error)
        rsi = _to_float(s.get("RSI") or s.get("rsi") or 50)
        if rsi >= 99 or rsi <= 5:
            continue
        
        # 3. For dividend screening: cap yield at 25% (above = likely anomaly or special dividend)
        if screening_type == "dividend":
            dy = _get_row_yield_pct(s)
            if dy > 25:
                continue
        
        # 4. Minimum market cap: 200M (filter micro-caps with no liquidity)
        mc = _to_float(s.get("market_cap_basic") or 0)
        if mc > 0 and mc < 200_000_000:
            continue
        
        out.append(s)
    return out
```

Call `_apply_quality_filter(stocks, screening_type)` right BEFORE the sorting/ranking step in `_screen_rows()` or wherever the stocks list is sorted.

### Additional: For "stable dividend" queries add stability bonus
When screening_type == "dividend", after filtering, sort by a composite score instead of raw yield:

```python
# Composite score: balance yield vs stability signals
def _div_stability_score(s: dict) -> float:
    dy = _get_row_yield_pct(s)
    mc = _to_float(s.get("market_cap_basic") or 0) / 1e9  # in billions
    pe = _to_float(s.get("price_earnings_ttm") or 0)
    rsi = _to_float(s.get("RSI") or s.get("rsi") or 50)
    vol = _to_float(s.get("volume") or 0)
    
    score = dy  # base: yield %
    
    # Bonus: large market cap (more stable)
    if mc > 10:   score += 2
    elif mc > 2:  score += 1
    
    # Bonus: reasonable P/E (profitable company)
    if 3 < pe < 20: score += 1
    
    # Penalty: RSI extremes (distressed or overbought)
    if rsi < 30 or rsi > 75: score -= 1
    
    # Bonus: high volume (liquid)
    if vol > 1_000_000: score += 0.5
    
    return score
```

Use `_div_stability_score` as the sort key when `screening_type == "dividend"`.

### Validation
```bash
source venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, '.')
from core.services.market_route_handler import _build_screening_reply

result = _build_screening_reply('أفضل أسهم توزيعات في الإمارات')
print(result)
print()
# Should NOT contain ORIENT or yields > 25%
assert 'ORIENT' not in result, 'ORIENT should be filtered out'
assert '150' not in result, 'anomalous 150% yield should be gone'
print('QUALITY CHECKS PASSED')
"
python3 -m py_compile core/services/market_route_handler.py && echo "SYNTAX OK"
```

Also reload gunicorn after the fix:
```bash
kill -HUP $(cat /home/ubuntu/investwise/gunicorn.pid 2>/dev/null || ps aux | grep 'gunicorn.*master' | grep -v grep | awk '{print $2}' | head -1)
```

### Report
File and line numbers changed.
