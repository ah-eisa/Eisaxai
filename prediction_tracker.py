"""
EisaX Prediction Tracker — logs report verdicts and checks accuracy after 30 days.
SQLite database: /home/ubuntu/investwise/predictions.db
"""
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / "predictions.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            verdict     TEXT NOT NULL,
            price_at    REAL,
            target      REAL,
            check_after TEXT NOT NULL,
            result      TEXT DEFAULT NULL,
            price_after REAL DEFAULT NULL,
            checked_at  TEXT DEFAULT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            fund_score  INTEGER,
            tech_score  INTEGER,
            blended     INTEGER,
            verdict     TEXT
        )
    """)
    conn.commit()
    return conn


def log_prediction(ticker: str, verdict: str, price_at: float, target: float):
    """Call this after every report generation."""
    try:
        conn = _get_conn()
        check_after = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO predictions (logged_at, ticker, verdict, price_at, target, check_after) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), ticker.upper(), verdict.upper(), price_at, target, check_after)
        )
        conn.commit()
        conn.close()
        logger.info(f"[PredTracker] Logged: {ticker} {verdict} @ {price_at} → target {target}")
    except Exception as e:
        logger.warning(f"[PredTracker] log_prediction failed: {e}")


def log_score(ticker: str, fund_score: int, tech_score: int, blended: int, verdict: str):
    # Reject zero/null score rows — they pollute get_score_velocity
    # and get_score_trend_chart by appearing as a -100% drop "vs last
    # analysis". Only log when at least one non-zero score is present.
    if not (fund_score or blended):
        logger.debug(f"[PredTracker] log_score skipped for {ticker}: all-zero row")
        return
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO score_history (logged_at, ticker, fund_score, tech_score, blended, verdict) VALUES (?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                ticker.upper(),
                int(fund_score or 0),
                int(tech_score or 0),
                int(blended or 0),
                verdict.upper(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[PredTracker] log_score failed: {e}")


def get_score_velocity(ticker: str, lookback: int = 7) -> dict:
    """Returns score change vs last analysis within lookback days."""
    try:
        conn = _get_conn()
        since = (datetime.now(timezone.utc) - timedelta(days=lookback)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute(
            "SELECT blended, verdict, logged_at FROM score_history WHERE ticker=? AND logged_at >= ? ORDER BY logged_at DESC LIMIT 5",
            (ticker.upper(), since),
        ).fetchall()
        conn.close()
        if len(rows) < 2:
            return {"change": 0, "direction": "stable", "arrow": "→", "prev_verdict": None}
        latest, prev = rows[0][0], rows[1][0]
        change = latest - prev
        if change >= 5:
            direction, arrow = "improving", "↑"
        elif change <= -5:
            direction, arrow = "deteriorating", "↓"
        else:
            direction, arrow = "stable", "→"
        return {
            "change": change,
            "direction": direction,
            "arrow": arrow,
            "prev_verdict": rows[1][1],
            "prev_score": prev,
            "current_score": latest,
        }
    except Exception as e:
        logger.warning(f"[PredTracker] get_score_velocity failed: {e}")
        return {"change": 0, "direction": "stable", "arrow": "→", "prev_verdict": None}


def get_portfolio_heatmap(ticker: str, sector: str, lookback_days: int = 30) -> dict:
    """
    Returns sector concentration data from stock_knowledge.
    Warns when multiple analyzed stocks belong to the same sector.
    """
    result = {"sector": sector, "peers_in_sector": [], "concentration_warning": False, "message": ""}
    if not sector or sector in ("Unknown", "N/A", ""):
        return result
    try:
        import sqlite3 as _sq
        import sys as _sys, os as _os
        _root = _os.path.dirname(_os.path.abspath(__file__))
        _db_paths = [
            _os.path.join(_root, "core", "investwise.db"),
            _os.path.join(_root, "investwise.db"),
        ]
        _db_path = next((p for p in _db_paths if _os.path.exists(p)), None)
        if not _db_path:
            return result
        _conn = _sq.connect(_db_path)
        _since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
        _rows = _conn.execute(
            "SELECT ticker, last_verdict, last_price FROM stock_knowledge "
            "WHERE sector=? AND ticker != ? AND last_updated >= ? ORDER BY last_updated DESC LIMIT 10",
            (sector, ticker.upper(), _since)
        ).fetchall()
        _conn.close()
        if _rows:
            result["peers_in_sector"] = [{"ticker": r[0], "verdict": r[1], "price": r[2]} for r in _rows]
            if len(_rows) >= 2:
                result["concentration_warning"] = True
                _peer_str = ", ".join(r[0] for r in _rows[:4])
                result["message"] = (
                    f"⚠️ **Portfolio Concentration:** {len(_rows)} other **{sector}** stocks recently analyzed "
                    f"({_peer_str}) — diversification check recommended."
                )
            elif len(_rows) == 1:
                result["message"] = (
                    f"ℹ️ **Sector Overlap:** {_rows[0][0]} also analyzed in **{sector}** recently."
                )
    except Exception as e:
        logger.warning(f"[PredTracker] get_portfolio_heatmap failed: {e}")
    return result


def get_score_trend_chart(ticker: str, lookback_days: int = 60, lang: str = "en") -> dict:
    """
    Returns ASCII sparkline of blended score history for the ticker.
    Requires at least 3 data points to be meaningful.

    ``lang='ar'`` produces an Arabic-localised message; the sparkline
    glyphs and numeric values are unchanged.
    """
    result = {"points": [], "chart": "", "message": ""}
    try:
        conn = _get_conn()
        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute(
            "SELECT blended, verdict, logged_at FROM score_history "
            "WHERE ticker=? AND logged_at >= ? ORDER BY logged_at ASC LIMIT 10",
            (ticker.upper(), since),
        ).fetchall()
        conn.close()
        if len(rows) < 3:
            return result
        scores = [r[0] for r in rows if r[0] is not None]
        verdicts = [r[1] for r in rows if r[0] is not None]
        if len(scores) < 3:
            return result

        # ASCII sparkline: map scores to 8 levels using Unicode blocks
        _SPARKS = " ▁▂▃▄▅▆▇█"
        _min_s, _max_s = min(scores), max(scores)
        _range = max(_max_s - _min_s, 1)
        spark = "".join(_SPARKS[int(round((s - _min_s) / _range * 8))] for s in scores)

        # Trend direction
        _first3_avg = sum(scores[:3]) / 3
        _last3_avg  = sum(scores[-3:]) / 3
        _trend_delta = _last3_avg - _first3_avg
        if _trend_delta >= 5:
            trend_label_en, trend_label_ar, trend_icon = "improving",     "تحسّن",   "📈"
        elif _trend_delta <= -5:
            trend_label_en, trend_label_ar, trend_icon = "deteriorating", "تراجع",   "📉"
        else:
            trend_label_en, trend_label_ar, trend_icon = "stable",        "مستقر",   "➡️"

        _AR_VERDICTS = {
            "STRONG BUY": "شراء قوي", "BUY": "شراء", "ACCUMULATE": "تجميع تدريجي",
            "HOLD": "احتفاظ", "REDUCE": "تخفيف", "SELL": "بيع", "AVOID": "تجنّب",
        }
        _last_v = (verdicts[-1] or "").upper()
        _last_v_ar = _AR_VERDICTS.get(_last_v, _last_v)

        result["points"] = scores
        result["chart"] = spark
        if lang == "ar":
            result["message"] = (
                f"{trend_icon} **اتجاه الدرجة ({len(scores)} تحليلات):** `{spark}` "
                f"({scores[0]}→{scores[-1]}) — **{trend_label_ar}** "
                f"| آخر قرار: **{_last_v_ar}**"
            )
        else:
            result["message"] = (
                f"{trend_icon} **Score Trend ({len(scores)} analyses):** `{spark}` "
                f"({scores[0]}→{scores[-1]}) — **{trend_label_en}** "
                f"| Last verdict: **{verdicts[-1]}**"
            )
    except Exception as e:
        logger.warning(f"[PredTracker] get_score_trend_chart failed: {e}")
    return result


def check_verdict_upgrade(ticker: str, prev_verdict: str, new_verdict: str, blended_score: int) -> dict:
    """
    Detects meaningful verdict upgrades (e.g. HOLD→BUY, REDUCE→HOLD).
    Returns alert dict if upgrade detected, else empty.
    Logs upgrade events to score_history notes via a simple alerts table.
    """
    _TIERS = {"SELL": 0, "AVOID": 0, "REDUCE": 1, "HOLD": 2, "ACCUMULATE": 3, "BUY": 3, "BUY (HIGH RISK)": 3, "STRONG BUY": 4}
    result = {"upgraded": False, "downgraded": False, "message": "", "tier_change": 0}
    if not prev_verdict or not new_verdict:
        return result
    _prev_t = _TIERS.get(prev_verdict.upper(), -1)
    _new_t  = _TIERS.get(new_verdict.upper(), -1)
    if _prev_t < 0 or _new_t < 0:
        return result

    _delta = _new_t - _prev_t
    result["tier_change"] = _delta

    if _delta >= 1:
        result["upgraded"] = True
        _icon = "🚀" if _delta >= 2 else "⬆️"
        result["message"] = (
            f"{_icon} **Verdict Upgrade Alert:** {ticker} upgraded from **{prev_verdict}** → **{new_verdict}** "
            f"(blended score: {blended_score}/100) — re-evaluate position sizing."
        )
        # Persist alert to DB
        try:
            conn = _get_conn()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS alerts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, logged_at TEXT, ticker TEXT, "
                "alert_type TEXT, prev_verdict TEXT, new_verdict TEXT, blended_score INTEGER, message TEXT)"
            )
            conn.execute(
                "INSERT INTO alerts (logged_at, ticker, alert_type, prev_verdict, new_verdict, blended_score, message) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), ticker.upper(), "UPGRADE",
                 prev_verdict.upper(), new_verdict.upper(), blended_score, result["message"])
            )
            conn.commit()
            conn.close()
        except Exception as _e:
            logger.debug(f"[PredTracker] alert persist failed: {_e}")

    elif _delta <= -1:
        result["downgraded"] = True
        _icon = "🔻" if _delta <= -2 else "⬇️"
        result["message"] = (
            f"{_icon} **Verdict Downgrade Alert:** {ticker} downgraded from **{prev_verdict}** → **{new_verdict}** "
            f"(blended score: {blended_score}/100) — review risk exposure."
        )

    return result


def check_due_predictions():
    """
    Check predictions whose check_after date has passed.
    Fetches current price via yfinance and marks Hit/Miss.
    Returns list of newly resolved predictions.
    """
    resolved = []
    try:
        import yfinance as yf
        conn = _get_conn()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        due = conn.execute(
            "SELECT id, ticker, verdict, price_at, target FROM predictions WHERE result IS NULL AND check_after <= ?",
            (today,)
        ).fetchall()

        for row in due:
            pred_id, ticker, verdict, price_at, target = row
            try:
                current = yf.Ticker(ticker).fast_info.get("last_price") or 0
                if not current or not target or not price_at:
                    continue
                if verdict in ("BUY", "ACCUMULATE"):
                    hit = current >= target
                elif verdict in ("SELL", "REDUCE", "AVOID"):
                    hit = current <= target
                else:  # HOLD — hit if within 5% of entry
                    hit = abs(current - price_at) / price_at < 0.05
                result = "Hit" if hit else "Miss"
                conn.execute(
                    "UPDATE predictions SET result=?, price_after=?, checked_at=? WHERE id=?",
                    (result, round(current, 4), today, pred_id)
                )
                resolved.append({"ticker": ticker, "verdict": verdict, "result": result,
                                  "price_at": price_at, "target": target, "price_after": current})
            except Exception as inner_e:
                logger.warning(f"[PredTracker] check failed for {ticker}: {inner_e}")

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[PredTracker] check_due_predictions failed: {e}")
    return resolved


def get_accuracy_stats(days: int = 90) -> dict:
    """Returns accuracy stats for the last N days."""
    try:
        conn = _get_conn()
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT verdict, result FROM predictions WHERE result IS NOT NULL AND checked_at >= ?",
            (since,)
        ).fetchall()
        conn.close()
        total = len(rows)
        if not total:
            return {"total": 0, "accuracy": None}
        hits = sum(1 for _, r in rows if r == "Hit")
        return {
            "total": total,
            "hits": hits,
            "misses": total - hits,
            "accuracy": round(hits / total * 100, 1),
        }
    except Exception as e:
        logger.warning(f"[PredTracker] get_accuracy_stats failed: {e}")
        return {"total": 0, "accuracy": None}
