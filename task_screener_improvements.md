# Task: Two screener improvements in core/services/market_route_handler.py

## Fix 1: Read requested count from message

In `_build_screening_reply()`, extract the number from the message before calling `_screen_rows()`.
Then pass `top_n` to `_screen_rows()` and use it instead of hardcoded `[:10]`.

Extract logic:
```python
import re as _re
_num_match = _re.search(r'\b(\d+)\b', message)
top_n = int(_num_match.group(1)) if _num_match and 3 <= int(_num_match.group(1)) <= 20 else 10
```

Change `_screen_rows(stocks, screening_type, message)` signature to accept `top_n=10` and use `rows[:top_n]` everywhere instead of `[:10]`.

Also update the freshness line to show the actual count:
```python
f"*{len(top)} سهم | آخر تحديث: {age_min:.0f} دقيقة*"
```
This already uses `len(top)` so it'll auto-update.

---

## Fix 2: Increase market cap weight in `_div_stability_score()`

Read the current function first: `grep -n "_div_stability_score" core/services/market_route_handler.py`

Update the market cap bonuses to be stronger so EMAAR (96B market cap) ranks above AWNIC (629M):

```python
def _div_stability_score(s: dict) -> float:
    dy = _get_row_yield_pct(s)
    mc = _to_float(s.get("market_cap_basic") or 0) / 1e9  # billions
    pe = _to_float(s.get("price_earnings_ttm") or 0)
    rsi = _to_float(s.get("RSI") or s.get("rsi") or 50)
    vol = _to_float(s.get("volume") or 0)

    score = dy  # base: yield %

    # Market cap bonus — much stronger weighting
    if mc > 50:    score += 6    # mega cap (EMAAR, FAB, ENBD)
    elif mc > 10:  score += 4    # large cap
    elif mc > 2:   score += 2    # mid cap
    elif mc > 0.5: score += 1    # small cap
    # micro cap: no bonus

    # P/E bonus: profitable and reasonably valued
    if 3 < pe < 15:  score += 1.5
    elif 15 <= pe < 25: score += 0.5

    # RSI: healthy range bonus
    if 35 <= rsi <= 65: score += 0.5

    # Volume bonus: liquid
    if vol > 5_000_000:   score += 1.5
    elif vol > 1_000_000: score += 1
    elif vol > 200_000:   score += 0.5

    return score
```

---

## Validation
```bash
source venv/bin/activate && python3 -c "
from core.services.market_route_handler import _build_screening_reply

# Test count extraction
r5 = _build_screening_reply('ايه أفضل 5 أسهم توزيعات مستقرة في الإمارات دلوقتي؟')
lines = [l for l in r5.split('\n') if l.startswith('|') and '|' in l[1:]]
data_rows = [l for l in lines if not l.startswith('|---') and 'الشركة' not in l]
print(f'Row count for top-5 query: {len(data_rows)} (expected 5)')
assert len(data_rows) == 5, f'Expected 5 rows, got {len(data_rows)}'

# Test EMAAR appears in top 5
assert 'EMAAR' in r5, 'EMAAR should appear with higher mc weight'
print(r5)
print('ALL CHECKS PASSED')
"
python3 -m py_compile core/services/market_route_handler.py && echo "SYNTAX OK"
```

## Report
List file and exact line numbers changed.
