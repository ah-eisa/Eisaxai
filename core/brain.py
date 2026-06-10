#!/usr/bin/env python3
"""
EisaX Brain - Autonomous Learning Engine
يبني معرفة حقيقية، يتذكر، يتعلم، ويشوف العالم كل يوم
"""
import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

DB_PATH = os.path.join(os.path.dirname(__file__), "investwise.db")
logger = logging.getLogger(__name__)


# ─── Database Setup ────────────────────────────────────────────────────────────
def init_brain_tables():
    """Create all brain tables if not exist"""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        -- معرفة تراكمية عن كل شركة/سهم
        CREATE TABLE IF NOT EXISTS stock_knowledge (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            summary TEXT,           -- ملخص تحليلي متراكم
            last_price REAL,
            last_verdict TEXT,      -- ACCUMULATE / HOLD / REDUCE
            last_sentiment TEXT,
            analysis_count INTEGER DEFAULT 0,
            first_seen DATE,
            last_updated DATETIME,
            tags TEXT               -- JSON array of tags
        );

        -- توقعات EisaX وتتبع دقتها
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            prediction_date DATE,
            verdict TEXT,           -- ACCUMULATE / HOLD / REDUCE
            price_at_prediction REAL,
            target_price REAL,
            horizon_days INTEGER DEFAULT 30,
            actual_price REAL,      -- يتملى بعد الـ horizon
            accuracy_pct REAL,      -- نسبة الدقة
            was_correct INTEGER,    -- 1 = صح, 0 = غلط
            notes TEXT,
            evaluated INTEGER DEFAULT 0  -- 0 = pending, 1 = evaluated
        );

        -- أخبار العالم اللي قرأها EisaX
        CREATE TABLE IF NOT EXISTS world_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            category TEXT,          -- markets / macro / geopolitics / tech / energy
            headline TEXT,
            summary TEXT,
            impact TEXT,            -- bullish / bearish / neutral
            affected_tickers TEXT,  -- JSON array
            source TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- ذاكرة المستخدمين
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            risk_profile TEXT DEFAULT 'medium',
            preferred_sectors TEXT,     -- JSON array
            watchlist TEXT,             -- JSON array of tickers
            total_interactions INTEGER DEFAULT 0,
            first_seen DATE,
            last_active DATETIME,
            notes TEXT                  -- ملاحظات EisaX عن المستخدم
        );

        -- سجل التعلم الذاتي
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            lesson TEXT,            -- ماذا تعلم EisaX
            category TEXT,          -- prediction_accuracy / market_pattern / user_behavior
            confidence REAL,        -- 0-1
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Brain tables initialized")


# ─── Stock Knowledge ───────────────────────────────────────────────────────────
def get_stock_memory(ticker: str) -> Optional[Dict]:
    """استرجاع كل ما يعرفه EisaX عن سهم معين"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stock_knowledge WHERE ticker = ?", (ticker.upper(),)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['tags'] = json.loads(d.get('tags') or '[]')
        return d
    return None


def update_stock_memory(ticker: str, analysis: Dict):
    """تحديث معرفة EisaX عن سهم بعد كل تحليل"""
    conn = sqlite3.connect(DB_PATH)
    ticker = ticker.upper()
    existing = conn.execute(
        "SELECT analysis_count, summary FROM stock_knowledge WHERE ticker = ?", (ticker,)
    ).fetchone()

    now = datetime.now().isoformat()
    today = datetime.now().date().isoformat()

    # Build cumulative summary
    new_summary = analysis.get('summary', '')
    if existing and existing[1]:
        # Keep last 3 summaries for context
        old_summaries = existing[1].split('|||')[-2:]
        combined = '|||'.join(old_summaries + [new_summary])
    else:
        combined = new_summary

    count = (existing[0] + 1) if existing else 1
    tags = json.dumps(analysis.get('tags', []))

    conn.execute("""
        INSERT INTO stock_knowledge
            (ticker, company_name, sector, summary, last_price, last_verdict,
             last_sentiment, analysis_count, first_seen, last_updated, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            summary = excluded.summary,
            last_price = excluded.last_price,
            last_verdict = excluded.last_verdict,
            last_sentiment = excluded.last_sentiment,
            analysis_count = excluded.analysis_count,
            last_updated = excluded.last_updated,
            tags = excluded.tags
    """, (
        ticker,
        analysis.get('company_name', ticker),
        analysis.get('sector', 'Unknown'),
        combined,
        analysis.get('price'),
        analysis.get('verdict'),
        analysis.get('sentiment'),
        count,
        today if not existing else conn.execute("SELECT first_seen FROM stock_knowledge WHERE ticker=?", (ticker,)).fetchone()[0],
        now,
        tags
    ))
    conn.commit()
    conn.close()
    logger.info(f"📚 Updated knowledge for {ticker} (analysis #{count})")


def save_prediction(ticker: str, verdict: str, price: float,
                    target: float = None, horizon: int = 30):
    """حفظ توقع EisaX لمتابعته لاحقاً"""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().date().isoformat()
    conn.execute("""
        INSERT INTO predictions (ticker, prediction_date, verdict, price_at_prediction, target_price, horizon_days)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker.upper(), today, verdict, price, target, horizon))
    conn.commit()
    conn.close()
    logger.info(f"🔮 Saved prediction: {ticker} → {verdict} @ ${price}")


def evaluate_old_predictions():
    """مراجعة التوقعات القديمة وحساب الدقة"""
    from market_data import get_realtime_quote
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().date()

    # Find predictions past their horizon
    pending = conn.execute("""
        SELECT id, ticker, prediction_date, verdict, price_at_prediction, horizon_days
        FROM predictions
        WHERE evaluated = 0
        AND date(prediction_date, '+' || horizon_days || ' days') <= date('now')
    """).fetchall()

    for pred in pending:
        pid, ticker, pred_date, verdict, old_price, horizon = pred
        quote = get_realtime_quote(ticker)
        current_price = quote.get('price')
        if not current_price:
            continue

        price_change = (current_price - old_price) / old_price
        was_correct = 0

        if verdict == 'ACCUMULATE' and price_change > 0.02:
            was_correct = 1
        elif verdict == 'REDUCE' and price_change < -0.02:
            was_correct = 1
        elif verdict == 'HOLD' and abs(price_change) <= 0.05:
            was_correct = 1

        conn.execute("""
            UPDATE predictions
            SET actual_price = ?, accuracy_pct = ?, was_correct = ?, evaluated = 1
            WHERE id = ?
        """, (current_price, price_change * 100, was_correct, pid))

        logger.info(f"📊 Evaluated {ticker}: {verdict} → {'✅' if was_correct else '❌'} ({price_change*100:+.1f}%)")

    conn.commit()

    # Calculate overall accuracy
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(was_correct) as correct,
            ROUND(AVG(was_correct)*100, 1) as accuracy
        FROM predictions WHERE evaluated = 1
    """).fetchone()
    conn.close()

    if stats and stats[0] > 0:
        lesson = f"Prediction accuracy: {stats[2]}% ({stats[1]}/{stats[0]} correct)"
        save_learning(lesson, 'prediction_accuracy', stats[2] / 100)

    return stats


# ─── World Knowledge ───────────────────────────────────────────────────────────
def save_world_news(category: str, headline: str, summary: str,
                    impact: str = 'neutral', tickers: List[str] = None):
    """حفظ خبر/معلومة عالمية تعلمها EisaX"""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().date().isoformat()
    conn.execute("""
        INSERT INTO world_knowledge (date, category, headline, summary, impact, affected_tickers)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (today, category, headline, summary, impact, json.dumps(tickers or [])))
    conn.commit()
    conn.close()


def get_recent_world_context(days: int = 7, category: str = None) -> List[Dict]:
    """استرجاع المعرفة العالمية الأخيرة"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    since = (datetime.now() - timedelta(days=days)).date().isoformat()

    if category:
        rows = conn.execute(
            "SELECT * FROM world_knowledge WHERE date >= ? AND category = ? ORDER BY date DESC LIMIT 20",
            (since, category)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM world_knowledge WHERE date >= ? ORDER BY date DESC LIMIT 20",
            (since,)
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ─── User Profiles ─────────────────────────────────────────────────────────────
def get_user_profile(user_id: str) -> Dict:
    """استرجاع ملف المستخدم"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['watchlist'] = json.loads(d.get('watchlist') or '[]')
        d['preferred_sectors'] = json.loads(d.get('preferred_sectors') or '[]')
        return d
    return {"user_id": user_id, "risk_profile": "medium", "watchlist": [], "total_interactions": 0}


def update_user_profile(user_id: str, updates: Dict):
    """تحديث ملف المستخدم"""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    today = datetime.now().date().isoformat()

    existing = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    count = (existing[6] + 1) if existing else 1

    watchlist = json.dumps(updates.get('watchlist', []))
    sectors = json.dumps(updates.get('preferred_sectors', []))

    conn.execute("""
        INSERT INTO user_profiles (user_id, name, risk_profile, preferred_sectors, watchlist,
                                   total_interactions, first_seen, last_active, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            risk_profile = COALESCE(excluded.risk_profile, risk_profile),
            watchlist = excluded.watchlist,
            total_interactions = excluded.total_interactions,
            last_active = excluded.last_active,
            notes = COALESCE(excluded.notes, notes)
    """, (
        user_id,
        updates.get('name', user_id),
        updates.get('risk_profile', 'medium'),
        sectors, watchlist, count, today, now,
        updates.get('notes')
    ))
    conn.commit()
    conn.close()


# ─── Learning Log ──────────────────────────────────────────────────────────────
def save_learning(lesson: str, category: str, confidence: float = 0.8):
    """تسجيل درس تعلمه EisaX"""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().date().isoformat()
    conn.execute("""
        INSERT INTO learning_log (date, lesson, category, confidence)
        VALUES (?, ?, ?, ?)
    """, (today, lesson, category, confidence))
    conn.commit()
    conn.close()
    logger.info(f"🎓 Learned: {lesson}")


def get_eisax_wisdom() -> str:
    """ملخص ما تعلمه EisaX حتى الآن"""
    conn = sqlite3.connect(DB_PATH)

    stocks = conn.execute("SELECT COUNT(*) FROM stock_knowledge").fetchone()[0]
    preds = conn.execute("SELECT COUNT(*), ROUND(AVG(was_correct)*100,1) FROM predictions WHERE evaluated=1").fetchone()
    lessons = conn.execute("SELECT COUNT(*) FROM learning_log").fetchone()[0]
    news = conn.execute("SELECT COUNT(*) FROM world_knowledge").fetchone()[0]

    conn.close()

    accuracy = preds[1] if preds[1] else 0
    total_preds = preds[0] if preds[0] else 0

    return (
        f"📊 **EisaX Knowledge Base:**\n"
        f"- Stocks analyzed: {stocks}\n"
        f"- Predictions made: {total_preds} (Accuracy: {accuracy}%)\n"
        f"- World events tracked: {news}\n"
        f"- Lessons learned: {lessons}"
    )


# ─── Daily World Reader ────────────────────────────────────────────────────────
def daily_world_update():
    """
    يشغل كل يوم تلقائياً - يقرأ الأخبار والمؤشرات ويحدث المعرفة
    """
    from market_data import get_macro_context, get_realtime_quote
    import requests

    logger.info("🌍 Starting daily world update...")
    today = datetime.now().date().isoformat()

    # 1. Macro Update
    macro = get_macro_context()
    t10y = macro.get('treasury_10y', {}).get('value')
    fed = macro.get('fed_funds', {}).get('value')
    inflation = macro.get('inflation', {}).get('value')
    unemployment = macro.get('unemployment', {}).get('value')

    if t10y:
        save_world_news(
            'macro',
            f"Macro Snapshot {today}",
            f"10Y Treasury: {t10y}% | Fed Rate: {fed}% | Unemployment: {unemployment}%",
            'neutral'
        )

    # 2. Update watchlist prices from DB
    conn = sqlite3.connect(DB_PATH)
    tickers = conn.execute("SELECT ticker FROM stock_knowledge").fetchall()
    conn.close()

    for (ticker,) in tickers:
        try:
            quote = get_realtime_quote(ticker)
            if quote.get('price'):
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "UPDATE stock_knowledge SET last_price = ?, last_updated = ? WHERE ticker = ?",
                    (quote['price'], datetime.now().isoformat(), ticker)
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to update {ticker}: {e}")

    # 3. Evaluate old predictions
    evaluate_old_predictions()

    logger.info("✅ Daily world update complete")


# ─── Context Builder for AI ───────────────────────────────────────────────────
def build_rich_context(ticker: str = None, user_id: str = None) -> str:
    """
    يبني context غني من كل المعرفة المتاحة
    يُستخدم قبل أي تحليل لإعطاء EisaX ذاكرة حقيقية
    """
    parts = []

    # Stock memory
    if ticker:
        mem = get_stock_memory(ticker)
        if mem:
            parts.append(
                f"📚 PREVIOUS KNOWLEDGE about {ticker}:\n"
                f"- Times analyzed: {mem['analysis_count']}\n"
                f"- Last verdict: {mem['last_verdict']}\n"
                f"- Last price: ${mem['last_price']}\n"
                f"- Summary: {mem['summary'].split('|||')[-1][:200] if mem['summary'] else 'None'}"
            )

        # Old predictions
        conn = sqlite3.connect(DB_PATH)
        preds = conn.execute("""
            SELECT verdict, price_at_prediction, prediction_date, was_correct, actual_price
            FROM predictions WHERE ticker = ? ORDER BY prediction_date DESC LIMIT 3
        """, (ticker.upper(),)).fetchall()
        conn.close()

        if preds:
            pred_text = "\n🔮 PAST PREDICTIONS:\n"
            for p in preds:
                verdict, old_price, date, correct, actual = p
                status = "✅" if correct == 1 else "❌" if correct == 0 else "⏳"
                pred_text += f"  {status} {date}: {verdict} @ ${old_price}"
                if actual:
                    pred_text += f" → actual ${actual:.2f}"
                pred_text += "\n"
            parts.append(pred_text)

    # Recent world news
    recent_news = get_recent_world_context(days=3)
    if recent_news:
        news_text = "🌍 RECENT WORLD CONTEXT:\n"
        for n in recent_news[:3]:
            news_text += f"  [{n['category'].upper()}] {n['headline']}: {n['summary'][:100]}\n"
        parts.append(news_text)

    # User profile
    if user_id:
        profile = get_user_profile(user_id)
        if profile.get('watchlist') or profile.get('risk_profile'):
            parts.append(
                f"👤 USER PROFILE:\n"
                f"  Risk: {profile['risk_profile']}\n"
                f"  Watchlist: {', '.join(profile['watchlist'][:5])}\n"
                f"  Interactions: {profile['total_interactions']}"
            )

    # RAG: semantically similar past analyses and world context
    try:
        from core.vector_memory import get_rag_context
        query = ticker or "market analysis"
        rag_block = get_rag_context(query, ticker=ticker, max_chars=800)
        if rag_block:
            parts.append(rag_block)
    except Exception as _rag_e:
        logger.debug("[RAG] build_rich_context vector search failed: %s", _rag_e)

    return "\n\n".join(parts) if parts else ""


# ─── Main: Initialize ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_brain_tables()
    print(get_eisax_wisdom())
    print("\nRunning daily update...")
    daily_world_update()
    print("\nDone!")
