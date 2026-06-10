"""
eisax_integration_patch.py
──────────────────────────
HOW TO INTEGRATE eisax_news_intelligence.py into your existing market_updates.py
WITHOUT breaking anything.

Just 4 targeted changes. Each change is marked with a ── PATCH N ── comment.
Copy-paste each block into market_updates.py at the indicated location.

Changes summary:
  PATCH 1: Import the news module at top of file
  PATCH 2: Replace the daily AI prompt call with news-enriched version
  PATCH 3: Replace the weekly AI prompt call with news-enriched version  
  PATCH 4: Enrich the final update object before saving (adds GCC + signals)
"""

# ════════════════════════════════════════════════════════════════════════════
# ── PATCH 1 ── Add at the top of market_updates.py, with other imports
# ════════════════════════════════════════════════════════════════════════════

# Add this block right after the existing imports section:
"""
# News Intelligence Layer
try:
    from eisax_news_intelligence import (
        get_news_context,
        inject_news_into_daily_prompt,
        inject_news_into_weekly_prompt,
        enrich_daily_update,
        NewsContext,
    )
    _NEWS_LAYER_AVAILABLE = True
    logger.info("[market_updates] News intelligence layer loaded ✓")
except ImportError:
    _NEWS_LAYER_AVAILABLE = False
    logger.warning("[market_updates] News intelligence layer not available — running without live news")
"""


# ════════════════════════════════════════════════════════════════════════════
# ── PATCH 2 ── In generate_daily_update(), replace the prompt + AI call
# ════════════════════════════════════════════════════════════════════════════

# FIND this block in generate_daily_update() (around line 1345):
"""
    prompt = f\"\"\"You are EisaX — institutional AI investment intelligence used by portfolio managers.
Generate a Daily Market Pulse as valid JSON only. Institutional tone. Direct. No generic phrases.
...
\"\"\"
    ai_result = _generate_insight(prompt, max_tokens=1000)
"""

# REPLACE WITH:
"""
    # ── PATCH 2: News-enriched prompt ──────────────────────────────────────
    if _NEWS_LAYER_AVAILABLE:
        try:
            news_ctx = get_news_context()
            prompt = inject_news_into_daily_prompt(
                base_prompt="",           # not used — full prompt built internally
                news=news_ctx,
                moves_summary=moves_summary,
                regime=regime,
                conf=conf,
                fg=fg,
                today=today,
            )
            logger.info("[market_updates] Daily prompt enriched with %d news items", news_ctx.total_items)
        except Exception as exc:
            logger.warning("[market_updates] News enrichment failed, using base prompt: %s", exc)
            news_ctx = None
            prompt = _build_base_daily_prompt(today, moves_summary, fg, regime, conf, stance, invali)
    else:
        news_ctx = None
        prompt = _build_base_daily_prompt(today, moves_summary, fg, regime, conf, stance, invali)

    ai_result = _generate_insight(prompt, max_tokens=1200)  # +200 tokens for news grounding
"""


# ════════════════════════════════════════════════════════════════════════════
# ── PATCH 3 ── In generate_weekly_update(), replace the prompt + AI call
# ════════════════════════════════════════════════════════════════════════════

# FIND this block in generate_weekly_update() (around line 1455):
"""
    prompt = f\"\"\"You are EisaX — institutional AI investment intelligence. Style: Goldman/Bridgewater strategy note.
...
\"\"\"
    ai_result = _generate_insight(prompt, max_tokens=1400)
"""

# REPLACE WITH:
"""
    # ── PATCH 3: News-enriched weekly prompt ───────────────────────────────
    if _NEWS_LAYER_AVAILABLE:
        try:
            news_ctx = get_news_context()  # Uses cached context if < 20 min old
            prompt = inject_news_into_weekly_prompt(
                news=news_ctx,
                moves_summary=moves_summary,
                regime=regime,
                conf=conf,
                fg=fg,
                week_range=week_range,
                stance=stance,
                invali=invali,
            )
            logger.info("[market_updates] Weekly prompt enriched with %d news items", news_ctx.total_items)
        except Exception as exc:
            logger.warning("[market_updates] Weekly news enrichment failed: %s", exc)
            news_ctx = None
            prompt = _build_base_weekly_prompt(week_range, moves_summary, fg, regime, conf, stance, invali)
    else:
        news_ctx = None
        prompt = _build_base_weekly_prompt(week_range, moves_summary, fg, regime, conf, stance, invali)

    ai_result = _generate_insight(prompt, max_tokens=1600)  # Extra tokens for GCC section
"""


# ════════════════════════════════════════════════════════════════════════════
# ── PATCH 4 ── In _finalize_daily_update(), add enrichment before return
# ════════════════════════════════════════════════════════════════════════════

# FIND _finalize_daily_update() (around line 1306):
"""
def _finalize_daily_update(update: dict, moves_summary: dict, fg: dict) -> dict:
    update["data_timestamp"] = _get_market_data_timestamp()
    update["web_version"] = _build_web_version(update)
    full_report = _generate_full_report_text(update, moves_summary, fg) or _build_full_report_fallback(update)
    update["full_report"] = full_report
    update["linkedin_text"] = _build_linkedin_text_v2(update)
    return update
"""

# REPLACE WITH:
"""
def _finalize_daily_update(update: dict, moves_summary: dict, fg: dict, news_ctx=None) -> dict:
    update["data_timestamp"] = _get_market_data_timestamp()

    # ── PATCH 4: Enrich with GCC intelligence + cross-asset signals ─────────
    if _NEWS_LAYER_AVAILABLE and news_ctx is not None:
        try:
            update = enrich_daily_update(update, news_ctx, moves_summary)
        except Exception as exc:
            logger.warning("[market_updates] Enrichment failed: %s", exc)

    update["web_version"] = _build_web_version(update)
    full_report = _generate_full_report_text(update, moves_summary, fg) or _build_full_report_fallback(update)
    update["full_report"] = full_report
    update["linkedin_text"] = _build_linkedin_text_v2(update)
    return update
"""

# NOTE: Also update the call to _finalize_daily_update in generate_daily_update:
# OLD: update = _finalize_daily_update(update, moves_summary, fg)
# NEW: update = _finalize_daily_update(update, moves_summary, fg, news_ctx=news_ctx)


# ════════════════════════════════════════════════════════════════════════════
# ── HELPER FUNCTIONS ── Add these to market_updates.py for clean separation
# ════════════════════════════════════════════════════════════════════════════

# These are the existing prompt bodies extracted into functions so the
# news-enriched path and fallback path share the same base logic.
# Add these right before generate_daily_update():

"""
def _build_base_daily_prompt(today, moves_summary, fg, regime, conf, stance, invali):
    \"\"\"Original daily prompt — used as fallback when news layer unavailable.\"\"\"
    return f\"\"\"You are EisaX — institutional AI investment intelligence used by portfolio managers.
Generate a Daily Market Pulse as valid JSON only. Institutional tone. Direct. No generic phrases.

Market data ({today}):
{json.dumps(moves_summary, indent=2)}

Fear & Greed: {fg.get('score', 50)} ({fg.get('rating', 'Neutral')})
Regime: {regime} (confidence: {conf})
Pre-computed stance: {json.dumps(stance)}
Pre-computed invalidation: {json.dumps(invali)}

Return ONLY this JSON (no markdown fences):
{{
  "date": "{today}",
  "market_regime": "{regime}",
  "regime_confidence": "{conf}",
  "what_matters_now": ["<insight>","<insight>","<insight>"],
  "key_moves": [{{"asset":"<n>","move":"<±X.X%>","reason":"<cause>"}}],
  "eisax_view": {{"stance":"<>","overweight_assets":[],"underweight_assets":[],"neutral_assets":[],"focus":"<>","horizon":"<>"}},
  "why_now": "<2 sentences>",
  "what_invalidates": ["<trigger>","<trigger>","<trigger>"],
  "tactical_positioning": "<action>",
  "next_triggers": ["<event>","<event>","<event>"],
  "fear_greed_index": {fg.get('score', 50)}
}}
Hard rules: NEVER write: "markets showed resilience" / "investor confidence increased". 120-180 words total.
\"\"\"


def _build_base_weekly_prompt(week_range, moves_summary, fg, regime, conf, stance, invali):
    \"\"\"Original weekly prompt — used as fallback when news layer unavailable.\"\"\"
    return f\"\"\"You are EisaX — institutional AI investment intelligence. Style: Goldman/Bridgewater strategy note. Decisive.
Generate a Weekly Strategy Brief as valid JSON only.

Market data (week: {week_range}):
{json.dumps(moves_summary, indent=2)}
Fear & Greed: {fg.get('score', 50)} ({fg.get('rating', 'Neutral')})
Regime: {regime} (confidence: {conf})
Pre-computed stance: {json.dumps(stance)}
Pre-computed invalidation: {json.dumps(invali)}

Return ONLY this JSON:
{{
  "week_range": "{week_range}",
  "market_summary": "<3 sentences>",
  "positioning": "<allocation stance>",
  "asset_allocation_view": {{"equities":"<>","crypto":"<>","metals":"<>","commodities":"<>","cash":"<>"}},
  "regional_view": {{"US":"<>","GCC":"<>","Egypt":"<>"}},
  "winners_losers": {{"winners":["<>"],"losers":["<>"]}},
  "highest_conviction_opportunity": "<specific trade>",
  "key_risks": ["<risk>","<risk>","<risk>"],
  "what_changes_this_view": ["<trigger>","<trigger>"],
  "portfolio_angle": "<2-3 sentences>",
  "eisax_verdict": "<1 sentence action>"
}}
Rules: 250-350 words. NEVER use "markets showed resilience". eisax_verdict: action verb first.
\"\"\"
"""


# ════════════════════════════════════════════════════════════════════════════
# ── INSTALLATION CHECKLIST ──
# ════════════════════════════════════════════════════════════════════════════
"""
CHECKLIST:
□ 1. Copy eisax_news_intelligence.py to the same directory as market_updates.py
     → /home/ubuntu/investwise/core/services/eisax_news_intelligence.py

□ 2. Install feedparser (optional but recommended for cleaner RSS parsing):
     pip install feedparser

□ 3. Apply PATCH 1: Add import block at top of market_updates.py

□ 4. Add the two helper functions (_build_base_daily_prompt, _build_base_weekly_prompt)
     just before generate_daily_update()

□ 5. Apply PATCH 2 in generate_daily_update()

□ 6. Apply PATCH 3 in generate_weekly_update()

□ 7. Apply PATCH 4 in _finalize_daily_update()
     Also update the call site: pass news_ctx=news_ctx

□ 8. Test with: python -c "from eisax_news_intelligence import get_news_context; ctx = get_news_context(); print(ctx.format_for_prompt())"

WHAT YOU GET AFTER INTEGRATION:
✓ Every daily report grounded in today's actual headlines
✓ key_moves explain WHY using real news (not just "risk sentiment")  
✓ Dedicated gcc_note from live MENA sources (Zawya, Gulf News, Arab News, The National)
✓ Cross-asset correlation signals (divergences Grok never detects)
✓ gcc_intelligence block: TASI view, DFM view, oil-GCC link, sovereign wealth context
✓ news_sources[] attribution — you can show where intelligence came from
✓ 20-minute news cache — won't hammer feeds on repeated calls
✓ Full graceful fallback — if news fails, original behavior is preserved exactly
"""
