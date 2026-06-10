# Phase D — Pre-grounding the LLM with FactSheet/SSOT

Status: **design only, no implementation**
Written: 2026-05-28
Author note: based on the existing prompt builder at `core/agents/handlers/analytics.py:2692` and the SSOT module added in commit `1f83c87`.

---

## 1. Why

Today's pipeline is **post-hoc**: the LLM writes a body that often disagrees with the SSOT FactSheet (wrong SMA200, wrong currency symbol, wrong verdict label, oil mentions inside a bank report, etc.). The `report_reconciler` then sweeps in and re-writes the body line-by-line.

Observed evidence from the workers=2 regression (4 fresh + 8 prior tickers, 24 corrections total):

| Correction type | Frequency | Cost |
|---|---|---|
| `live_price_factcheck_row` | 100% of reports | 1 swap each |
| `sma200_paren_swap` / `bold_swap` / `colon_swap` / `echo_swap` | ~60% | up to 4 swaps each |
| `verdict_swap_X_to_Y` | ~50% | 1–4 swaps |
| `sector_scrub_oil` (bank / RE) | 100% of banks + RE | 1–3 strips |
| `sma200_ladder_inject` | EMAAR-type cases | 1 table inject |

Every correction is a sign the LLM was given enough freedom to invent wrong data. **Pre-grounding** removes that freedom by handing the LLM the SSOT values in the system prompt **before** it generates, so the body matches the FactSheet without retro-active fixing.

### Expected outcome

| Metric | Today | After Phase D |
|---|---|---|
| Average reconciler corrections per report | 2–7 | **0–2** |
| `verdict_swap` rate | ≥ 50% | ≤ 10% |
| `sector_scrub_oil` on bank reports | 100% | ≤ 15% |
| Body internal consistency before reconciler | partial | strong |
| Tokens spent on LLM "creative" mistakes | non-trivial | ≈ zero |
| Reconciler still required? | yes (safety net) | yes (safety net, but mostly idle) |

---

## 2. Insertion Point — Confirmed

`core/agents/handlers/analytics.py:2692` — the `prompt = f"""You are EisaX, Chief Investment Officer..."""` block. After all the existing context blocks are appended (peer rows at 1976, scorecard at 2339, positioning at 2957, news at 2962, mode at 2996), insert the **FactSheet block right before** the DeepSeek call on line 3002.

```
… existing prompt assembly …
prompt += _local_data_injection          ← line 2962 today
prompt += _factsheet_block               ← NEW (Phase D)
…
response = requests.post(deepseek_url, json={"messages":[{"role":"user","content":prompt}]})
```

We do NOT touch the existing scorecard / peer / news / positioning blocks. The FactSheet block is **additive** and explicit — its job is to lock down a small set of facts the LLM frequently gets wrong.

---

## 3. Block Spec

### 3.1 Format

```
═══════════════════════════════════════════════════════════════════════════
GROUND TRUTH FOR THIS REPORT (Single Source of Truth — DO NOT CONTRADICT)
═══════════════════════════════════════════════════════════════════════════

Ticker          : ADNOCGAS.AE
Bare symbol     : ADNOCGAS
Market          : uae        Sector subtype: energy_producer (Energy Minerals)
Currency        : د.إ (AED)  — write all prices in this symbol; never use $
Snapshot age    : 12 minutes  (timestamp 2026-05-28T22:34:00Z)

LIVE TECHNICAL FACTS  (TV cache, do not invent alternative numbers)
- Price          : د.إ3.43
- SMA50          : د.إ3.27   (price vs SMA50 = +4.9%)
- SMA200         : د.إ3.41   (price vs SMA200 = +0.6%)
- RSI            : 56
- 52w range      : د.إ2.78 – د.إ3.78

VERDICT  (DecisionState authoritative — match this in the report header)
- Verdict        : Buy
- Action         : Scale In
- Risk           : High
- Confidence     : Medium
- Score          : 75/100  (Fundamental quality 71/100)

SECTOR GUARDRAILS  (Energy Producer)
- ALLOWED themes : crude / Brent / OPEC / refining / hydrocarbon margins
- BANNED themes  : oil-as-a-cost (not a producer concern), bank/credit themes,
                   real-estate, consumer-discretionary

WRITE RULES
1. The "Verdict" line in the report header MUST equal: **Buy**.
2. Every price you mention MUST be in د.إ. Never write $ except for Brent/USD oil.
3. SMA200 must equal د.إ3.41 — do not round to "approximately" or substitute
   a 200-day average from another data feed.
4. Do not invent SMA200 if it is null (commodities / fresh cryptos).
   Instead say "SMA200 not available — use SMA50 instead".
5. Stay in the energy-producer thesis. Do not include bank/RE language.

═══════════════════════════════════════════════════════════════════════════
END OF GROUND TRUTH BLOCK
═══════════════════════════════════════════════════════════════════════════
```

### 3.2 What goes into the block

Pulled directly from `FactSheet` fields (already authoritative):

| Block line | FactSheet field |
|---|---|
| Ticker, Bare symbol, Market | `ticker`, `bare_symbol`, `market` |
| Currency | `currency_symbol` + `currency_code` |
| Snapshot age | `snapshot_age_seconds` |
| Sector subtype | `sector_subtype.value`, `sector` |
| Price, SMA50, SMA200, RSI | `price`, `sma50`, `sma200`, `rsi` |
| `price_vs_sma200_pct` | already computed in FactSheet |
| 52w range | `low_52w`, `high_52w` (add to FactSheet if absent today) |
| Verdict, Action, Risk, Confidence, Score | `verdict`, `action`, `overall_risk_label`, `confidence`, `eisax_score`, `fundamental_quality_score` |
| Allowed / Banned themes | derived from `sector_subtype` via a static `_THEME_GUARDRAILS` dict |

### 3.3 Cases the block handles specially

- **`price = None`** → omit the LIVE TECHNICAL FACTS section entirely; do not lie. The reconciler safety net catches missing-price reports.
- **`sma200 = None`** (commodities, fresh crypto) → emit "SMA200: not available" — instruct the LLM not to invent.
- **`currency_symbol = None` for MENA** → blocking error, do not call LLM (already enforced in FactSheet validation).
- **`verdict = None`** → omit the VERDICT section; let LLM use scorecard pre-verdict.

---

## 4. New helper

In `core/services/fact_sheet.py`, add:

```python
def render_pregrounding_block(fs: FactSheet) -> str:
    """
    Render the SSOT FactSheet as a system-prompt block for the LLM.
    Output is a fenced 'GROUND TRUTH' section the LLM must respect.
    Idempotent and safe to call before every LLM request.
    """
```

**No side-effects, no I/O.** Returns a multi-line `str`. Pure function.

In `core/agents/handlers/analytics.py`, around line 2962:

```python
prompt += _local_data_injection
# Phase D: append the SSOT FactSheet block so the LLM grounds on it
try:
    from core.services.fact_sheet import build_fact_sheet, render_pregrounding_block
    _fs = build_fact_sheet(target, live_payload=live_payload)
    if not _fs.blocking_errors:
        prompt += "\n\n" + render_pregrounding_block(_fs)
        logger.info("[PreGround] %s: FactSheet block injected (%d chars)", target, len(_fs.price or ""))
except Exception as _e:
    logger.warning("[PreGround] %s: skipped — %s", target, _e)
```

Three-line addition wrapped in try/except so a FactSheet error never breaks report generation. The reconciler still runs after, so if the LLM ignores the block, corrections still land.

---

## 5. Theme guardrails table

Already implicit in `_NEWS_PROFILES` (used by reconciler's sector_scrub). Phase D needs it in a forward-facing form. Add to `fact_sheet.py`:

```python
_THEME_GUARDRAILS = {
    SectorSubtype.ENERGY_PRODUCER: {
        "allowed": ["crude", "Brent", "OPEC", "refining", "hydrocarbon margins"],
        "banned":  ["bank/credit themes", "real-estate", "consumer-discretionary"],
    },
    SectorSubtype.GAS_LNG: {
        "allowed": ["LNG spot", "gas demand", "pipeline capacity", "Hormuz transit"],
        "banned":  ["bank/credit themes", "real-estate", "consumer-discretionary"],
    },
    SectorSubtype.REAL_ESTATE_DEVELOPER: {
        "allowed": ["off-plan sales", "land acquisition", "construction backlog",
                    "mortgage rates", "rental yields"],
        "banned":  ["oil price", "crude/Brent", "OPEC", "refinery margins"],
    },
    SectorSubtype.BANK: {
        "allowed": ["NIM", "loan growth", "deposits", "central bank rate",
                    "capital adequacy / Basel ratios", "cost of risk"],
        "banned":  ["oil price (except as macro context for the country)",
                    "crude/Brent direct exposure", "OPEC quotas"],
    },
    SectorSubtype.INSURANCE: {"allowed": ["premiums", "claims", "underwriting"], "banned": []},
    SectorSubtype.TECHNOLOGY: {"allowed": [], "banned": []},
    SectorSubtype.CRYPTO: {"allowed": ["volatility", "halving cycle", "stablecoin flows"],
                            "banned":  ["P/E", "EPS", "dividends", "central bank policy specific to fiat issuers"]},
    SectorSubtype.COMMODITY: {"allowed": ["spot", "futures curve", "carry", "storage"],
                               "banned":  ["P/E", "EPS", "dividends"]},
    # ... other subtypes default to empty allowed/banned lists
}
```

---

## 6. Testing plan (when Phase D is implemented)

| Test | Method | Pass criteria |
|---|---|---|
| **T1 — block renders correctly per subtype** | Unit test on `render_pregrounding_block(fs)` with mock FactSheets for each `SectorSubtype` | Output matches expected template per type |
| **T2 — Buy report keeps Buy verdict in body** | Manual end-to-end on ADNOCGAS.AE with mock score=75 → verdict=Buy | Body header says "Verdict: Buy" with no reconciler `verdict_swap` correction |
| **T3 — currency stays in fs.currency_symbol** | Manual on EMAAR.AE (AED) | Body has only د.إ symbols for prices; no $ except Brent/oil refs |
| **T4 — bank report has zero oil mentions** | Manual on COMI.CA | `sector_scrub_oil` correction count = 0 |
| **T5 — SMA200 matches fs.sma200 exactly** | Manual on EMAAR.AE (SMA200=13.80) | Body has SMA200=13.80 verbatim, no `sma200_*_swap` correction |
| **T6 — Reconciler still acts as safety net** | Adversarial test: temporarily mangle the pre-grounding block so LLM gets wrong numbers | Reconciler still corrects to FactSheet values (proves safety net intact) |
| **T7 — Overall corrections reduced** | Re-run the 8-ticker regression pack | Average corrections drops from ~3 to ≤1 |
| **T8 — Token impact** | Compare prompt sizes pre/post | Block adds ~500 tokens; acceptable on DeepSeek's 128k context |

---

## 7. Rollout plan (when implemented)

1. **Land helper** `render_pregrounding_block` in `fact_sheet.py` — pure function, unit-tested, mergeable solo.
2. **Land integration** in `analytics.py` line 2962, behind feature flag `EISAX_PREGROUNDING=1` (default `0`).
3. **Staging A/B**: 10 tickers with flag on, 10 with flag off. Compare:
   - Reconciler correction count
   - Verdict-mismatch count
   - LLM completion latency (block adds ~5–10% tokens; should not impact much)
   - Subjective body readability
4. **Set default to `1`** on staging if A/B is favorable.
5. **Production promotion** as Phase 3 (was deferred in the Phase 1+2 plan).

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| **D-R1** | LLM "argues with" the block ("the SSOT says X but I think Y…") | Reword block as imperative + use ALL-CAPS + reconciler still runs |
| **D-R2** | Block bloats prompt → cold-cache calls hit `timeout=120s` more often | Keep block ≤ 600 tokens; LLM caches prompt prefix per session anyway |
| **D-R3** | A wrong FactSheet block leads the LLM into a wrong report | Reconciler enforces consistency post-hoc; block must come from validated FactSheet (no blocking_errors) |
| **D-R4** | Hard-coded `_THEME_GUARDRAILS` falls behind sector evolution | Reuse the existing `_NEWS_PROFILES` regex as basis; both stay in sync in `fact_sheet.py` |
| **D-R5** | Per-ticker prompt grows from ~6k → ~7k chars → DeepSeek pricing pressure | Measure cost in A/B; expected single-digit % cost bump |
| **D-R6** | Reconciler becomes dead code if pre-grounding works perfectly | Keep reconciler as safety net — it costs nothing if no corrections fire and protects against LLM regression |

---

## 9. Out of scope for Phase D

- **Auto-fix reconciler corrections at the LLM layer** (would require a feedback loop with retry; not designed here)
- **Removing the reconciler** (we keep it as safety net)
- **Changing the LLM provider** (DeepSeek stays)
- **Multi-turn LLM correction** (single-turn body still; reconciler patches output)

---

## 10. Effort estimate

| Task | Hours |
|---|---|
| Implement `render_pregrounding_block` + `_THEME_GUARDRAILS` | 2 |
| Wire into analytics.py behind feature flag | 1 |
| Unit tests T1, T2 | 2 |
| A/B harness + run 10 vs 10 | 3 |
| Read results + decide default | 1 |
| **Total to staging flag-on** | **9 hours** |

Phase D end-to-end through production: ~2 working days including soak.
