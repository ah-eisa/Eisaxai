#!/usr/bin/env python3
import threading
import time
import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("LearningEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

DB_PATH = os.getenv("DB_PATH", "/home/ubuntu/investwise/investwise.db")

CYCLE_INTERVALS = {
    "evaluate_predictions": 60 * 60,
    "update_prices":        60 * 60 * 4,
    "extract_lessons":      60 * 60 * 12,
    "macro_snapshot":       60 * 60 * 24,
    "daily_analysis":       60 * 60 * 24,
    "cleanup_old_sessions": 60 * 60 * 24,
}

_engine_instance = None


class LearningEngine:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._stop_event = threading.Event()
        self._thread = None
        self._last_run = {}
        self._stats = {
            "predictions_evaluated": 0,
            "lessons_learned": 0,
            "prices_updated": 0,
            "errors": 0,
            "started_at": None,
        }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="EisaX-LearningEngine",
            daemon=True
        )
        self._thread.start()
        self._stats["started_at"] = datetime.now().isoformat()
        logger.info("🧠 LearningEngine started — EisaX is now autonomous.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("🛑 LearningEngine stopped.")

    def status(self):
        return {
            "running": self._thread.is_alive() if self._thread else False,
            "stats": self._stats,
            "last_run": self._last_run,
            "next_run": {task: self._next_run_in(task) for task in CYCLE_INTERVALS}
        }

    def _run_loop(self):
        logger.info("🔄 Learning loop started.")
        self._safe_run("evaluate_predictions", self._evaluate_predictions)
        while not self._stop_event.is_set():
            now = time.time()
            for task_name, interval in CYCLE_INTERVALS.items():
                last = self._last_run.get(task_name, 0)
                if now - last >= interval:
                    handler = getattr(self, f"_{task_name}", None)
                    if handler:
                        self._safe_run(task_name, handler)
            self._stop_event.wait(timeout=60)
        logger.info("🔄 Learning loop exited cleanly.")

    def _safe_run(self, task_name, fn):
        try:
            logger.info(f"▶️  Running: {task_name}")
            fn()
            self._last_run[task_name] = time.time()
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"❌ Task '{task_name}' failed: {e}", exc_info=True)

    def _next_run_in(self, task):
        last = self._last_run.get(task, 0)
        remaining = max(0, CYCLE_INTERVALS[task] - (time.time() - last))
        if remaining == 0:
            return "due now"
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m}m" if h else f"{m}m {s}s"

    # ── TASK 1: Evaluate Predictions ──────────────────────────────────────────

    def _evaluate_predictions(self):
        conn = self._get_conn()
        pending = conn.execute("""
            SELECT id, ticker, prediction_date, verdict, price_at_prediction, horizon_days
            FROM predictions
            WHERE evaluated = 0
              AND date(prediction_date, '+' || horizon_days || ' days') <= date('now')
        """).fetchall()
        conn.close()

        if not pending:
            logger.info("✅ No predictions to evaluate.")
            return

        logger.info(f"🔮 Evaluating {len(pending)} predictions...")
        evaluated_count = 0
        correct_count = 0

        for row in pending:
            pid, ticker, pred_date, verdict, old_price, horizon = row
            try:
                current_price = self._fetch_price(ticker)
                if not current_price or not old_price:
                    continue

                change_pct = (current_price - old_price) / old_price
                was_correct = 0

                if verdict in ("ACCUMULATE", "BUY") and change_pct > 0.02:
                    was_correct = 1
                elif verdict in ("REDUCE", "SELL") and change_pct < -0.02:
                    was_correct = 1
                elif verdict == "HOLD" and abs(change_pct) <= 0.05:
                    was_correct = 1

                conn = self._get_conn()
                conn.execute("""
                    UPDATE predictions
                    SET actual_price = ?, accuracy_pct = ?, was_correct = ?, evaluated = 1
                    WHERE id = ?
                """, (current_price, round(change_pct * 100, 2), was_correct, pid))
                conn.commit()
                conn.close()

                evaluated_count += 1
                correct_count += was_correct
                icon = "✅" if was_correct else "❌"
                logger.info(f"  {icon} {ticker} | {verdict} @ ${old_price:.2f} → ${current_price:.2f} ({change_pct:+.1%})")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to evaluate {ticker}: {e}")

        self._stats["predictions_evaluated"] += evaluated_count

        if evaluated_count > 0:
            accuracy = round(correct_count / evaluated_count * 100, 1)
            lesson = (
                f"Batch evaluation: {evaluated_count} predictions reviewed. "
                f"Accuracy: {accuracy}%. Correct: {correct_count}/{evaluated_count}."
            )
            self._save_lesson(lesson, "prediction_accuracy", confidence=accuracy / 100)
            logger.info(f"📊 Batch accuracy: {accuracy}%")

            if accuracy < 50:
                self._save_lesson(
                    f"Low accuracy warning ({accuracy}%). Review methodology for recent predictions.",
                    "self_correction", confidence=0.9
                )

    # ── TASK 2: Update Prices ─────────────────────────────────────────────────

    def _update_prices(self):
        conn = self._get_conn()
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM stock_knowledge ORDER BY last_updated ASC"
        ).fetchall()]
        conn.close()

        if not tickers:
            logger.info("📭 No tickers in knowledge base yet.")
            return

        logger.info(f"💹 Updating prices for {len(tickers)} tickers (parallel)...")

        def _fetch_and_store(ticker: str) -> int:
            try:
                price = self._fetch_price(ticker)
                if price:
                    c = self._get_conn()
                    c.execute(
                        "UPDATE stock_knowledge SET last_price = ?, last_updated = ? WHERE ticker = ?",
                        (price, datetime.now().isoformat(), ticker)
                    )
                    c.commit()
                    c.close()
                    return 1
            except Exception as e:
                logger.warning(f"  ⚠️ Price update failed for {ticker}: {e}")
            return 0

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_fetch_and_store, tickers))

        updated = sum(results)
        self._stats["prices_updated"] += updated
        logger.info(f"  ✅ Updated {updated}/{len(tickers)} prices.")

    # ── TASK 3: Extract Lessons ───────────────────────────────────────────────

    def _extract_lessons(self):
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT ticker, verdict, price_at_prediction, actual_price,
                   accuracy_pct, was_correct, prediction_date
            FROM predictions
            WHERE evaluated = 1
            ORDER BY prediction_date DESC
            LIMIT 20
        """).fetchall()

        stats = conn.execute("""
            SELECT COUNT(*), ROUND(AVG(was_correct)*100,1),
                   SUM(CASE WHEN verdict='ACCUMULATE' AND was_correct=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN verdict='REDUCE' AND was_correct=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN verdict='HOLD' AND was_correct=1 THEN 1 ELSE 0 END)
            FROM predictions WHERE evaluated = 1
        """).fetchone()

        recent_lessons = [r[0] for r in conn.execute(
            "SELECT lesson FROM learning_log ORDER BY created_at DESC LIMIT 5"
        ).fetchall()]
        conn.close()

        if not rows or (stats and stats[0] < 3):
            logger.info("📚 Not enough evaluated predictions for lesson extraction yet.")
            return

        total, accuracy, acc_correct, red_correct, hold_correct = stats
        accuracy = accuracy or 0

        predictions_summary = "\n".join([
            f"- {r[0]}: {r[1]} @ ${r[2]:.2f} → ${r[3]:.2f if r[3] else 0:.2f} "
            f"({'✅' if r[5] else '❌'}) | Δ{r[4]:+.1f}%"
            for r in rows if r[2]
        ])

        prompt = f"""You are EisaX, an autonomous AI investment analyst.
Analyze your recent prediction performance and extract 2-3 concrete lessons.

PERFORMANCE STATS:
- Total evaluated: {total}
- Overall accuracy: {accuracy}%
- ACCUMULATE calls correct: {acc_correct}
- REDUCE calls correct: {red_correct}
- HOLD calls correct: {hold_correct}

RECENT PREDICTIONS:
{predictions_summary}

ALREADY LEARNED:
{chr(10).join(recent_lessons) if recent_lessons else 'None yet'}

Return ONLY a JSON array — no explanation:
[{{"lesson": "...", "category": "prediction_accuracy|market_pattern|sector_insight|self_correction", "confidence": 0.0-1.0}}]
"""

        try:
            lessons_data = self._call_gemini(prompt)
            if lessons_data:
                for item in lessons_data:
                    lesson = item.get("lesson", "").strip()
                    category = item.get("category", "prediction_accuracy")
                    confidence = float(item.get("confidence", 0.7))
                    if lesson:
                        self._save_lesson(lesson, category, confidence)
                        self._stats["lessons_learned"] += 1
                        logger.info(f"🎓 [{category}]: {lesson[:80]}...")
        except Exception as e:
            logger.error(f"Lesson extraction failed: {e}")

    def _call_gemini(self, prompt: str) -> list:
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not ds_key:
            logger.warning("No DEEPSEEK_API_KEY — skipping lesson extraction.")
            return []
        try:
            import httpx
            r = httpx.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1200,
                    "temperature": 0.3,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            raw = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if "```" in raw:
                parts = raw.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        raw = p[4:].strip()
                        break
                    elif p.startswith("["):
                        raw = p
                        break
            return json.loads(raw)
        except Exception as e:
            logger.error(f"DeepSeek lesson call failed: {e}")
            return []

    # ── TASK 4: Macro Snapshot ────────────────────────────────────────────────

    def _macro_snapshot(self):
        try:
            import sys
            sys.path.insert(0, "/home/ubuntu/investwise")
            from market_data import get_macro_context
        except ImportError:
            logger.warning("market_data not importable — skipping macro snapshot.")
            return

        try:
            macro = get_macro_context()
            if not macro:
                return

            t10y = macro.get('treasury_10y', {}).get('value', 'N/A')
            fed  = macro.get('fed_funds', {}).get('value', 'N/A')
            cpi  = macro.get('inflation', {}).get('value', 'N/A')
            gdp  = macro.get('gdp_growth', {}).get('value', 'N/A')
            unem = macro.get('unemployment', {}).get('value', 'N/A')

            today = datetime.now().date().isoformat()
            summary = (
                f"10Y Treasury: {t10y}% | Fed Rate: {fed}% | "
                f"CPI YoY: {cpi}% | GDP Growth: {gdp}% | Unemployment: {unem}%"
            )

            impact = "neutral"
            try:
                if float(str(cpi)) > 4:
                    impact = "bearish"
                elif float(str(t10y)) < 3.5 and float(str(gdp)) > 2:
                    impact = "bullish"
            except Exception as _e:
                pass

            conn = self._get_conn()
            conn.execute("""
                INSERT INTO world_knowledge (date, category, headline, summary, impact, source)
                VALUES (?, 'macro', ?, ?, ?, 'FRED/EisaX')
            """, (today, f"Macro Snapshot {today}", summary, impact))
            conn.commit()
            conn.close()
            logger.info(f"🌍 Macro snapshot saved: {summary[:60]}...")

        except Exception as e:
            logger.error(f"Macro snapshot failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    # Mapping for tickers that use a different suffix in yfinance vs internal system
    _YFINANCE_REMAP = {
        ".DU": ".AE",   # DFM Dubai tickers: yfinance uses .AE suffix
    }

    def _fetch_price(self, ticker: str):
        try:
            import yfinance as yf
            # Remap internal ticker suffixes to yfinance-compatible ones
            yf_ticker = ticker
            for src, dst in self._YFINANCE_REMAP.items():
                if ticker.endswith(src):
                    yf_ticker = ticker[:-len(src)] + dst
                    break
            t = yf.Ticker(yf_ticker)
            info = t.info
            price = (
                info.get("regularMarketPrice") or
                info.get("currentPrice") or
                info.get("previousClose")
            )
            return round(float(price), 2) if price else None
        except Exception as _e:
            return None

    def _save_lesson(self, lesson: str, category: str, confidence: float = 0.8):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO learning_log (date, lesson, category, confidence)
            VALUES (date('now'), ?, ?, ?)
        """, (lesson, category, min(1.0, max(0.0, confidence))))
        conn.commit()
        conn.close()

    def _daily_analysis(self):
        """يحلل أهم الأسهم في قاعدة البيانات ويسجل predictions."""
        try:
            import sys
            sys.path.insert(0, "/home/ubuntu/investwise")
            from core.market_data import get_full_stock_profile
            from core.scorecard import calculate_score, get_verdict
        except ImportError as e:
            logger.warning(f"daily_analysis imports failed: {e}")
            return

        conn = self._get_conn()
        # Get top tickers from knowledge base
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM stock_knowledge ORDER BY last_updated DESC LIMIT 10"
        ).fetchall()]
        conn.close()

        if not tickers:
            logger.info("📭 No tickers to analyze.")
            return

        logger.info(f"🤖 Daily autonomous analysis: {tickers}")
        analyzed = 0

        for ticker in tickers:
            try:
                profile = get_full_stock_profile(ticker)
                quote = profile.get("quote", {})
                price = quote.get("price")
                if not price:
                    continue

                # Get fundamentals from yfinance directly
                try:
                    import yfinance as yf
                    info = yf.Ticker(ticker).info
                    _pct = lambda v: round(v * 100, 2) if v else None
                    fund = {
                        "price": price,
                        "pe_ratio": info.get("trailingPE"),
                        "forward_pe": info.get("forwardPE"),
                        "revenue_growth": _pct(info.get("revenueGrowth")),
                        "eps_growth": _pct(info.get("earningsGrowth")),
                        "gross_margin": _pct(info.get("grossMargins")),
                        "net_margin": _pct(info.get("profitMargins")),
                        "operating_margin": _pct(info.get("operatingMargins")),
                        "roe": _pct(info.get("returnOnEquity")),
                        "roic": _pct(info.get("returnOnAssets")),
                        "debt_equity": info.get("debtToEquity"),
                        "beta": info.get("beta"),
                        "analyst_target": info.get("targetMeanPrice"),
                        "sma200": info.get("twoHundredDayAverage"),
                        "sma50": info.get("fiftyDayAverage"),
                        "quality": min(100, int((
                            (_pct(info.get("grossMargins")) or 0) * 0.3 +
                            (_pct(info.get("profitMargins")) or 0) * 0.3 +
                            (_pct(info.get("returnOnEquity")) or 0) * 0.2 +
                            (50 if info.get("debtToEquity", 100) < 50 else 20)
                        ))),
                    }
                    sc_result = calculate_score(fund)
                    score = sc_result[1] if sc_result else 50
                except Exception as _e:
                    score = 50
                verdict = "ACCUMULATE" if score >= 75 else "REDUCE" if score < 40 else "HOLD"

                # Save prediction to DB
                conn = self._get_conn()
                conn.execute("""
                    INSERT INTO predictions
                    (ticker, verdict, price_at_prediction, prediction_date, horizon_days, evaluated, notes)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                """, (ticker, verdict, price, datetime.now().isoformat(), 30, f"EisaX Score: {score}/100"))
                conn.commit()
                conn.close()

                logger.info(f"🔮 {ticker}: {verdict} @ ${price:.2f} | Score: {score}/100")
                analyzed += 1

            except Exception as e:
                logger.warning(f"❌ Failed to analyze {ticker}: {e}")

        self._stats["predictions_evaluated"] += analyzed
        logger.info(f"✅ Daily analysis complete — {analyzed} stocks analyzed.")

    def _cleanup_old_sessions(self):
        """Delete sessions and chat history older than 90 days, then VACUUM the DB.
        Also prunes unbounded stock_memory and stale ai_extracted user facts."""
        try:
            import sys
            sys.path.insert(0, "/home/ubuntu/investwise")
            from core.session_manager import SessionManager
            sm = SessionManager(self.db_path)
            deleted = sm.cleanup_old_sessions(days=90)
            logger.info("🧹 DB cleanup complete — %d old chat rows removed.", deleted)
        except Exception as e:
            logger.warning("DB cleanup failed: %s", e)

        # Prune memory tables to prevent unbounded growth
        try:
            import sys
            sys.path.insert(0, "/home/ubuntu/investwise")
            from core.memory_manager import prune_old_memory
            stats = prune_old_memory(stock_memory_days=90, user_memory_days=180, max_stock_rows=500)
            logger.info("🧹 Memory pruned — stock=%d trimmed=%d user_facts=%d",
                        stats["stock_memory"], stats["stock_trimmed"], stats["user_memory"])
        except Exception as e:
            logger.warning("Memory prune failed: %s", e)

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn


# ── Singleton ─────────────────────────────────────────────────────────────────

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LearningEngine()
    return _engine_instance

def start_learning_engine():
    engine = get_engine()
    engine.start()
    return engine

def stop_learning_engine():
    global _engine_instance
    if _engine_instance:
        _engine_instance.stop()
