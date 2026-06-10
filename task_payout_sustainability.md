# Task: Add payout ratio + sustainability indicator to dividend screener

## Context
We have `earnings_per_share_diluted_ttm`, `close`, and `dividend_yield_recent` in the pipeline cache.
Payout ratio = (dividend_yield_recent/100 * close) / earnings_per_share_diluted_ttm * 100

Current results are ranked by yield+market_cap but ignore payout ratio, so ICAP (134% payout) ranks above EMAARDEV (39% payout).

## Changes in core/services/market_route_handler.py

### 1. Add payout ratio helper function

```python
def _get_row_payout_ratio(s: dict) -> float | None:
    """Calculate payout ratio from available data. Returns None if not computable."""
    try:
        price = _to_float(s.get("close") or s.get("price") or s.get("current_price") or 0)
        dy_pct = _get_row_yield_pct(s)
        eps = _to_float(s.get("earnings_per_share_diluted_ttm") or s.get("eps") or 0)
        if price > 0 and dy_pct > 0 and eps > 0:
            dps = (dy_pct / 100) * price
            return round((dps / eps) * 100, 1)
    except Exception:
        pass
    return None


def _sustainability_flag(payout: float | None) -> str:
    """Return emoji sustainability indicator based on payout ratio."""
    if payout is None:
        return "—"
    if payout <= 50:
        return "🟢"   # excellent
    if payout <= 70:
        return "🟡"   # acceptable
    if payout <= 90:
        return "🟠"   # caution
    return "🔴"        # unsustainable (paying > earnings)
```

### 2. Update `_div_stability_score()` to penalize high payout ratio

Add to the existing function (after all the current bonuses):
```python
    # Payout ratio — most important for sustainability
    payout = _get_row_payout_ratio(s)
    if payout is not None:
        if payout <= 40:    score += 4    # excellent sustainability
        elif payout <= 60:  score += 2    # healthy
        elif payout <= 80:  score += 0    # neutral
        elif payout <= 100: score -= 3    # risky
        else:               score -= 6   # paying more than earnings
```

### 3. Add sustainability column to dividend screening output table

In `_build_screening_reply()`, for `screening_type == "dividend"`, update the table to include sustainability column:

Change header row from:
```
f"| # | الشركة | الرمز | السعر | {metric_label} | P/E | RSI |"
"|---|--------|-------|-------|----------|-----|-----|"
```

To (when screening_type == "dividend"):
```
f"| # | الشركة | الرمز | السعر | Div Yield | Payout | P/E | الاستدامة |"
"|---|--------|-------|-------|-----------|--------|-----|-----------|"
```

And update each row to include payout + sustainability flag:
```python
payout = _get_row_payout_ratio(s)
sustain = _sustainability_flag(payout)
payout_str = f"{payout:.0f}%" if payout is not None else "—"
rows.append(f"| {i} | {name} | {ticker or '—'} | {currency}{price:,.2f} | {dy_pct:.2f}% | {payout_str} | {pe} | {sustain} |")
```

The `metric_label` / `_fmt_screen_metric` column is replaced by the two new columns for dividend screening only. Other screening types (gainers, losers, etc.) keep the original format.

## Validation
```bash
source venv/bin/activate && python3 -c "
from core.services.market_route_handler import _build_screening_reply

result = _build_screening_reply('ايه أفضل 5 أسهم توزيعات مستقرة في الإمارات دلوقتي؟')
print(result)
print()

# EMAARDEV should rank #1 or #2 (lowest payout ratio 39%)
# EMAAR should be in top 3
# ICAP should rank low (134% payout)
lines = [l for l in result.split('\n') if l.startswith('| ') and '---' not in l and 'الشركة' not in l]
print('Rows:', len(lines))

# Check sustainability column exists
assert '🟢' in result or '🟡' in result or '🔴' in result, 'Missing sustainability flag'
print('Sustainability flags present ✓')
assert 'Payout' in result, 'Missing Payout column'
print('Payout column present ✓')
print('ALL CHECKS PASSED')
"
python3 -m py_compile core/services/market_route_handler.py && echo "SYNTAX OK"
```

## Report
List file and exact line numbers changed.
