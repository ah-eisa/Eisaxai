"""
EisaX Long-term Memory Manager
Stores and retrieves user facts across sessions
"""
import sqlite3
import json
import logging
import time as _time
from datetime import datetime
from typing import Dict, List, Optional

from core.db import db, brain_db

logger = logging.getLogger(__name__)

# ── In-process TTL cache for get_rich_user_context ───────────────────────────
_ctx_cache: Dict[str, tuple] = {}  # {user_id: (fetched_at, ctx)}
_CTX_TTL = 300  # 5 minutes

from core.config import APP_DB as _cfg_app_db
DB_PATH = str(_cfg_app_db)

def _init_memory_table():
    with db.get_cursor() as (conn, c):
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category, key)
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS stock_memory (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL UNIQUE,
                last_verdict TEXT,
                last_price REAL,
                analysis_count INTEGER DEFAULT 1,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT
            )
        ''')

_init_memory_table()


# ── Memory Pruning ────────────────────────────────────────────────────────────

def prune_old_memory(
    stock_memory_days: int = 90,
    user_memory_days: int = 180,
    max_stock_rows: int = 500,
) -> dict:
    """
    Prevent unbounded memory growth by expiring stale rows.

    stock_memory:
      - Delete analyses older than `stock_memory_days` (default 90 days).
      - If more than `max_stock_rows` remain, keep only the most recently
        analysed ones (trim the tail).

    user_memory:
      - Delete ai_extracted facts older than `user_memory_days` (default 180 days).
        Core profile facts (category != 'ai_extracted') are kept indefinitely.

    Returns a dict with counts of deleted rows for logging.
    """
    deleted = {"stock_memory": 0, "user_memory": 0, "stock_trimmed": 0}
    try:
        with db.get_cursor() as (conn, c):
            # 1. Expire stale stock analyses
            c.execute(
                "DELETE FROM stock_memory WHERE last_analyzed < datetime('now', ?)",
                (f"-{stock_memory_days} days",)
            )
            deleted["stock_memory"] = c.rowcount

            # 2. Trim excess stock rows (keep only the freshest max_stock_rows)
            c.execute("SELECT COUNT(*) FROM stock_memory")
            total_stocks = c.fetchone()[0]
            if total_stocks > max_stock_rows:
                overflow = total_stocks - max_stock_rows
                c.execute(
                    "DELETE FROM stock_memory WHERE id IN ("
                    "  SELECT id FROM stock_memory ORDER BY last_analyzed ASC LIMIT ?"
                    ")",
                    (overflow,)
                )
                deleted["stock_trimmed"] = c.rowcount

            # 3. Expire stale ai_extracted user facts (keep hand-entered profile facts)
            c.execute(
                "DELETE FROM user_memory WHERE category = 'ai_extracted' "
                "AND updated_at < datetime('now', ?)",
                (f"-{user_memory_days} days",)
            )
            deleted["user_memory"] = c.rowcount

        logger.info(
            "[prune_old_memory] deleted stock=%d (trimmed=%d), user_facts=%d",
            deleted["stock_memory"], deleted["stock_trimmed"], deleted["user_memory"]
        )
    except Exception as e:
        logger.warning("[prune_old_memory] failed: %s", e)
    return deleted


# ── User Memory ───────────────────────────────────────────────────────────────

def save_user_fact(user_id: str, category: str, key: str, value: str):
    """Save a fact about the user (e.g. category='portfolio', key='risk', value='high')"""
    with db.get_cursor() as (conn, c):
        c.execute('''
            INSERT INTO user_memory (user_id, category, key, value, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, category, key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
        ''', (user_id, category, key, value, datetime.now()))


def get_user_memory(user_id: str) -> Dict:
    """Get all facts about a user, grouped by category"""
    with db.get_cursor() as (conn, c):
        c.execute('''
            SELECT category, key, value FROM user_memory
            WHERE user_id = ? ORDER BY category, updated_at DESC
        ''', (user_id,))
        rows = c.fetchall()

    result = {}
    for category, key, value in rows:
        if category not in result:
            result[category] = {}
        result[category][key] = value
    return result


def format_memory_for_prompt(user_id: str) -> str:
    """Format user memory as context for the AI prompt"""
    memory = get_user_memory(user_id)
    if not memory:
        return ""

    lines = ["[USER MEMORY — facts from previous sessions:]"]
    for category, facts in memory.items():
        for key, value in facts.items():
            lines.append(f"- {category}/{key}: {value}")
    return "\n".join(lines)


# ── Stock Memory ──────────────────────────────────────────────────────────────

def save_stock_analysis(ticker: str, verdict: str, price: float, summary: str = "", user_id: str = ""):
    """Save stock analysis result to SQLite + embed in vector memory for RAG.
    If user_id is provided, also saves per-user history for cross-session continuity."""
    with db.get_cursor() as (conn, c):
        c.execute('''
            INSERT INTO stock_memory (ticker, last_verdict, last_price, analysis_count, last_analyzed, summary)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                last_verdict=excluded.last_verdict,
                last_price=excluded.last_price,
                analysis_count=analysis_count + 1,
                last_analyzed=excluded.last_analyzed,
                summary=excluded.summary
        ''', (ticker, verdict, price, datetime.now(), summary))

    # Per-user history — enables "you analyzed this before" continuity
    if user_id:
        try:
            save_user_analysis(user_id, ticker, verdict, price, summary)
        except Exception as _ue:
            logger.debug("save_user_analysis failed: %s", _ue)

    # RAG: also embed into ChromaDB for semantic retrieval
    if summary:
        try:
            from core.vector_memory import embed_analysis
            embed_analysis(ticker, summary, {"verdict": verdict, "price": price})
        except Exception as e:
            logger.debug("[RAG] embed_analysis failed: %s", e)


def save_user_analysis(user_id: str, ticker: str, verdict: str, price: float, summary: str = ""):
    """
    Per-user stock analysis history.
    Stores the last analysis result for each (user, ticker) pair in user_memory.
    Called alongside save_stock_analysis — tracks WHO analyzed WHAT and WHEN.
    """
    import json as _j
    value = _j.dumps({
        "verdict": verdict,
        "price": price,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "summary": summary[:200] if summary else "",
    })
    save_user_fact(user_id, "history", f"analyzed_{ticker.upper()}", value)
    _invalidate_user_ctx_cache(user_id)


def get_user_ticker_history(user_id: str, ticker: str) -> Optional[Dict]:
    """
    Retrieve the last time this user analyzed this ticker.
    Returns dict with verdict, price, date — or None if never analyzed.
    """
    import json as _j
    try:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT value FROM user_memory WHERE user_id=? AND category='history' AND key=?",
                (user_id, f"analyzed_{ticker.upper()}")
            )
            row = c.fetchone()
        if not row:
            return None
        return _j.loads(row[0])
    except Exception as e:
        logger.debug("get_user_ticker_history failed: %s", e)
        return None


def get_user_recent_analyses(user_id: str, limit: int = 5) -> List[Dict]:
    """
    Return the most recent (user, ticker) analyses, sorted by updated_at DESC.
    Used to build a "Recent Activity" context block for the LLM.
    """
    import json as _j
    results = []
    try:
        with db.get_cursor() as (conn, c):
            c.execute(
                """SELECT key, value, updated_at FROM user_memory
                   WHERE user_id=? AND category='history' AND key LIKE 'analyzed_%'
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit)
            )
            rows = c.fetchall()
        for key, val, updated in rows:
            ticker = key.replace("analyzed_", "")
            try:
                data = _j.loads(val)
                data["ticker"] = ticker
                data["updated_at"] = updated
                results.append(data)
            except Exception:
                pass
    except Exception as e:
        logger.debug("get_user_recent_analyses failed: %s", e)
    return results


def get_stock_memory(ticker: str) -> Optional[Dict]:
    """Get previous analysis for a stock"""
    with db.get_cursor() as (conn, c):
        c.execute('''
            SELECT ticker, last_verdict, last_price, analysis_count, last_analyzed, summary
            FROM stock_memory WHERE ticker = ?
        ''', (ticker.upper(),))
        row = c.fetchone()

    if not row:
        return None
    return {
        "ticker": row[0],
        "last_verdict": row[1],
        "last_price": row[2],
        "analysis_count": row[3],
        "last_analyzed": row[4],
        "summary": row[5]
    }


def _invalidate_user_ctx_cache(user_id: str):
    """Evict a user's context from the cache (call after saving new facts)."""
    _ctx_cache.pop(user_id, None)


def get_rich_user_context(user_id: str) -> dict:
    """
    Aggregate user context from all sources with a 5-minute in-process cache.
    - user_memory table (ai-extracted facts, preferences, profile)
    - brain.py user_profiles table (risk_profile, watchlist, preferred_sectors)
    - session_manager user_profiles (tier)
    Returns a structured dict ready for prompt injection.
    """
    now = _time.time()
    cached = _ctx_cache.get(user_id)
    if cached and (now - cached[0]) < _CTX_TTL:
        return cached[1]

    ctx = _build_rich_user_context(user_id)
    _ctx_cache[user_id] = (now, ctx)
    return ctx


def _build_rich_user_context(user_id: str) -> dict:
    """Internal: fetch and build rich user context from all DB sources."""
    ctx = {
        "user_id": user_id,
        "name": None,
        "risk_profile": None,
        "preferred_sectors": [],
        "watchlist": [],
        "capital": None,
        "currency": None,
        "investment_goal": None,
        "time_horizon": None,
        "tier": "basic",
    }

    # 1. user_memory table (fast facts saved by AI extraction)
    try:
        mem = get_user_memory(user_id)
        if mem:
            profile = mem.get("profile", {})
            prefs = mem.get("preferences", {})
            interests = mem.get("interests", {})
            ai_ex = mem.get("ai_extracted", {})
            ctx["name"] = profile.get("name") or ai_ex.get("name")
            ctx["risk_profile"] = (prefs.get("risk_profile") or ai_ex.get("risk_profile"))
            ctx["capital"] = prefs.get("capital") or ai_ex.get("capital")
            ctx["currency"] = prefs.get("currency") or ai_ex.get("currency")
            ctx["investment_goal"] = ai_ex.get("investment_goal")
            ctx["time_horizon"] = ai_ex.get("time_horizon")
            sector = interests.get("sector_focus") or ai_ex.get("sector_focus")
            if sector:
                ctx["preferred_sectors"].append(sector)
    except Exception as e:
        logger.debug("user_memory read failed: %s", e)

    # 2. brain.py user_profiles (rich: watchlist, sectors, risk)
    try:
        import json as _j
        with brain_db.get_cursor() as (conn, c):
            c.execute(
                "SELECT name, risk_profile, preferred_sectors, watchlist FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = c.fetchone()
        if row:
            ctx["name"] = ctx["name"] or row[0]
            if row[1]:
                ctx["risk_profile"] = ctx["risk_profile"] or row[1]
            wl = _j.loads(row[3] or "[]")
            secs = _j.loads(row[2] or "[]")
            ctx["watchlist"] = list(set(ctx["watchlist"] + wl))
            ctx["preferred_sectors"] = list(set(ctx["preferred_sectors"] + secs))
    except Exception as e:
        logger.debug("brain user_profiles read failed: %s", e)

    # 3. tier from session_manager user_profiles
    try:
        with db.get_cursor() as (conn, c):
            c.execute(
                "SELECT tier FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            row = c.fetchone()
        if row and row[0]:
            ctx["tier"] = row[0]
    except Exception as e:
        logger.debug("tier read failed: %s", e)

    # 4. Recent stock analyses (per-user history)
    try:
        recent = get_user_recent_analyses(user_id, limit=5)
        if recent:
            ctx["recent_analyses"] = recent
    except Exception as e:
        logger.debug("recent_analyses read failed: %s", e)

    # Normalize risk profile
    if not ctx["risk_profile"]:
        ctx["risk_profile"] = "medium"

    return ctx


def format_ctx_for_prompt(ctx: dict, target_ticker: str = None) -> str:
    """
    Format a pre-built user context dict as a personalized prompt block.
    Injected into LLM prompts so the AI knows who it's talking to.
    """
    has_data = any([
        ctx.get("name"),
        ctx.get("risk_profile") not in (None, "medium"),
        ctx.get("capital"),
        ctx.get("watchlist"),
        ctx.get("preferred_sectors"),
        ctx.get("investment_goal"),
    ])
    if not has_data:
        return ""

    lines = []
    name = ctx.get("name")
    tier = (ctx.get("tier") or "basic").upper()

    if name:
        lines.append(f"USER: {name} | Tier: {tier}")
    else:
        lines.append(f"USER TIER: {tier}")

    risk = ctx.get("risk_profile", "medium")
    lines.append(f"Risk Profile: {risk.capitalize()}")

    if ctx.get("capital"):
        lines.append(f"Capital: {ctx['capital']}")
    if ctx.get("currency"):
        lines.append(f"Preferred Currency: {ctx['currency']}")
    if ctx.get("investment_goal"):
        lines.append(f"Investment Goal: {ctx['investment_goal']}")
    if ctx.get("time_horizon"):
        lines.append(f"Time Horizon: {ctx['time_horizon']}")
    if ctx.get("preferred_sectors"):
        lines.append(f"Sector Interests: {', '.join(ctx['preferred_sectors'])}")
    if ctx.get("watchlist"):
        wl = ctx["watchlist"][:12]
        lines.append(f"Watchlist: {', '.join(wl)}")
        if target_ticker and target_ticker.upper() in [t.upper() for t in wl]:
            lines.append(f"⭐ {target_ticker} IS ON THIS USER'S WATCHLIST — provide extra depth")

    # Personalization instruction
    if risk in ("aggressive", "high"):
        lines.append("TAILOR: Emphasize upside, growth catalysts, momentum — user accepts high risk")
    elif risk in ("conservative", "low"):
        lines.append("TAILOR: Emphasize downside risks, capital preservation, dividend yield — user prefers safety")
    elif risk == "moderate":
        lines.append("TAILOR: Balanced risk/reward focus — user wants both growth and protection")

    # Greeting personalization
    if name:
        lines.append(f"NOTE: User's name is '{name}' — use it naturally only when contextually appropriate, not in every response")

    # ── Recent analyses — critical for continuity ──────────────────────────
    recent = ctx.get("recent_analyses", [])
    if recent:
        lines.append("RECENT ANALYSES (user's history — reference when relevant):")
        for r in recent[:4]:
            tk = r.get("ticker", "?")
            verd = r.get("verdict", "?")
            px = r.get("price")
            dt = r.get("date", r.get("updated_at", "?"))[:10]
            px_str = f" @ {px:.2f}" if px else ""
            # Flag if this is the target ticker being analyzed now
            if target_ticker and tk.upper() == target_ticker.upper():
                lines.append(
                    f"  ⭐ {tk}: PREVIOUSLY analyzed {dt}{px_str} → verdict was {verd}. "
                    f"IMPORTANT: Acknowledge the prior analysis, note what has changed since then."
                )
            else:
                lines.append(f"  • {tk}: {dt}{px_str} → {verd}")

    header = "\n\n[USER PROFILE — personalize your response accordingly]\n"
    return header + "\n".join(lines) + "\n[END USER PROFILE]\n"


def format_memory_for_prompt_rich(user_id: str, target_ticker: str = None) -> str:
    """Enhanced version of format_memory_for_prompt that uses all data sources."""
    ctx = get_rich_user_context(user_id)
    return format_ctx_for_prompt(ctx, target_ticker=target_ticker)


def extract_and_save_user_facts(user_id: str, message: str, reply: str):
    """Auto-extract facts from conversation — keyword rules + AI extraction."""
    import os, re

    msg_lower = message.lower()

    # ── Risk profile ──────────────────────────────────────────────────────────
    _aggressive = ["aggressive", "high risk", "عدواني", "عدوانية", "مخاطرة عالية",
                   "مخاطر عالية", "عالي المخاطر", "high return", "max return"]
    _conservative = ["conservative", "low risk", "محافظ", "محافظة", "مخاطرة منخفضة",
                     "منخفض المخاطر", "مخاطر منخفضة", "capital preservation", "safe",
                     "أمان", "بدون مخاطرة"]
    _moderate = ["moderate", "medium", "medium risk", "معتدل", "معتدلة", "متوازن",
                 "متوازنة", "balanced", "مخاطرة متوسطة", "medium return"]
    if any(w in msg_lower for w in _aggressive):
        save_user_fact(user_id, "preferences", "risk_profile", "aggressive")
    elif any(w in msg_lower for w in _conservative):
        save_user_fact(user_id, "preferences", "risk_profile", "conservative")
    elif any(w in msg_lower for w in _moderate):
        save_user_fact(user_id, "preferences", "risk_profile", "moderate")

    # ── Currency ──────────────────────────────────────────────────────────────
    if any(w in msg_lower for w in ["usd", "dollar", "دولار", "$"]):
        save_user_fact(user_id, "preferences", "currency", "USD")
    elif any(w in msg_lower for w in ["aed", "درهم", "إماراتي"]):
        save_user_fact(user_id, "preferences", "currency", "AED")
    elif any(w in msg_lower for w in ["sar", "ريال سعودي", "riyal"]):
        save_user_fact(user_id, "preferences", "currency", "SAR")
    elif any(w in msg_lower for w in ["egp", "جنيه", "pound"]):
        save_user_fact(user_id, "preferences", "currency", "EGP")

    # ── Name ─────────────────────────────────────────────────────────────────
    name_match = re.search(
        r"(?:my name is|call me|i am|أنا|اسمي|اسمى|يسموني|اتصل بي)\s+([\w\u0600-\u06FF]+)",
        message, re.I | re.UNICODE
    )
    if name_match:
        save_user_fact(user_id, "profile", "name", name_match.group(1))

    # ── Capital / Investment amount ───────────────────────────────────────────
    # Arabic: "100 الف دولار", "مليون درهم", "500k"
    cap_patterns = [
        r"(\d[\d,\.]*)\s*(?:k|K)\s*(?:USD|AED|SAR|دولار|درهم|\$)",
        r"(\d[\d,\.]*)\s*(?:الف|ألف)\s*(?:دولار|درهم|ريال|\$)",
        r"(\d[\d,\.]*)\s*(?:مليون|million)\s*(?:دولار|درهم|ريال|\$)?",
        r"\$\s*(\d[\d,\.]+(?:[kKmM])?)",
        r"(\d[\d,]+)\s*(?:USD|AED|SAR)",
    ]
    for pat in cap_patterns:
        cap_m = re.search(pat, message, re.I)
        if cap_m:
            save_user_fact(user_id, "preferences", "capital", cap_m.group(0).strip())
            break

    # ── Time horizon ─────────────────────────────────────────────────────────
    if any(w in msg_lower for w in ["long term", "long-term", "طويل الأمد", "طويل المدى", "5 years", "10 years"]):
        save_user_fact(user_id, "ai_extracted", "time_horizon", "long-term")
    elif any(w in msg_lower for w in ["short term", "short-term", "قصير الأمد", "قصير المدى", "1 year", "months"]):
        save_user_fact(user_id, "ai_extracted", "time_horizon", "short-term")
    elif any(w in msg_lower for w in ["medium term", "medium-term", "متوسط الأمد", "3 years", "2-3"]):
        save_user_fact(user_id, "ai_extracted", "time_horizon", "medium-term")

    # ── Investment goal ───────────────────────────────────────────────────────
    if any(w in msg_lower for w in ["retirement", "تقاعد", "retire"]):
        save_user_fact(user_id, "ai_extracted", "investment_goal", "retirement")
    elif any(w in msg_lower for w in ["passive income", "دخل سلبي", "دخل شهري", "monthly income"]):
        save_user_fact(user_id, "ai_extracted", "investment_goal", "passive income")
    elif any(w in msg_lower for w in ["growth", "نمو", "capital growth", "wealth building"]):
        save_user_fact(user_id, "ai_extracted", "investment_goal", "capital growth")

    # ── Sector interests ─────────────────────────────────────────────────────
    _sector_map = {
        "Technology":   ["tech", "technology", "تكنولوجيا", "ai", "semiconductor", "software", "ذكاء اصطناعي"],
        "Energy":       ["energy", "oil", "طاقة", "نفط", "opec", "petrochemical", "بترول"],
        "Real Estate":  ["real estate", "عقارات", "property", "reit", "عقار"],
        "Finance":      ["bank", "banking", "بنوك", "بنك", "financial", "insurance", "تأمين"],
        "Crypto":       ["crypto", "bitcoin", "btc", "ethereum", "blockchain", "عملات رقمية", "بيتكوين"],
        "Healthcare":   ["health", "pharma", "صحة", "دواء", "pharmaceutical", "biotech"],
        "Gold":         ["gold", "ذهب", "precious metal", "xauusd"],
    }
    for sector, keywords in _sector_map.items():
        if any(w in msg_lower for w in keywords):
            save_user_fact(user_id, "interests", "sector_focus", sector)
            break

    # ── AI-powered deep extraction (runs in background thread — non-blocking) ──
    # Uses DeepSeek directly — Gemini keys are currently unavailable.
    def _ai_extract_bg():
        try:
            import json as _json, os as _os_mem, requests as _req_mem
            _ds_key = _os_mem.getenv("DEEPSEEK_API_KEY", "")
            if not _ds_key:
                return
            _prompt = (
                "Extract user facts from this conversation. Return ONLY valid JSON, no markdown.\n"
                f"User: {message[:500]}\nAssistant: {reply[:800]}\n\n"
                "Return JSON (null if not mentioned — be conservative, only extract clear signals):\n"
                '{"name":null,"risk_profile":null,"capital":null,"currency":null,'
                '"sector_focus":null,"investment_goal":null,"time_horizon":null,'
                '"preferred_markets":null,"language":"ar or en"}'
            )
            _r = _req_mem.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_ds_key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": _prompt}],
                      "max_tokens": 300, "temperature": 0.1},
                timeout=10,
            )
            _text = (_r.json().get("choices", [{}])[0]
                              .get("message", {})
                              .get("content", "") or "").strip()
            _text = _text.replace("```json","").replace("```","").strip()
            if not _text:
                return
            _facts = _json.loads(_text)
            _changed = False
            for key, value in _facts.items():
                if value and str(value).lower() not in ("null","none","","n/a"):
                    save_user_fact(user_id, "ai_extracted", key, str(value))
                    _changed = True
            if _changed:
                _invalidate_user_ctx_cache(user_id)
        except Exception as _e:
            logger.debug("AI memory extraction failed: %s", _e)

    # Fire-and-forget in a daemon thread so it never blocks the response
    import threading as _threading
    _t = _threading.Thread(target=_ai_extract_bg, daemon=True)
    _t.start()


def track_stock_interest(user_id: str, ticker: str):
    """
    Auto-track stocks a user asks about into their watchlist in brain DB.
    Called from orchestrator after STOCK_ANALYSIS queries.
    Keeps watchlist trimmed to the 20 most recent unique tickers.
    """
    if not ticker or len(ticker) < 2:
        return
    try:
        import json as _j
        ticker_up = ticker.upper()
        with brain_db.get_cursor() as (conn, c):
            # Ensure the row exists before the atomic update
            c.execute(
                "INSERT OR IGNORE INTO user_profiles "
                "(user_id, watchlist, risk_profile, total_interactions, first_seen, last_active) "
                "VALUES (?, '[]', 'medium', 0, date('now'), datetime('now'))",
                (user_id,)
            )

            row = c.execute(
                "SELECT watchlist FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            wl = _j.loads(row[0] or "[]") if row else []
            if ticker_up in wl:
                wl.remove(ticker_up)
            wl.insert(0, ticker_up)
            wl = wl[:20]

            c.execute(
                "UPDATE user_profiles SET watchlist = ? WHERE user_id = ?",
                (_j.dumps(wl), user_id)
            )
    except Exception as e:
        logger.debug("track_stock_interest failed: %s", e)
