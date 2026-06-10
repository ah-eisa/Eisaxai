"""
portfolio_memory.py — EisaX Portfolio History & Audit Trail
Stores every portfolio analysis: holdings, metrics, data sources, report hash.
Enables: performance tracking over time, reproducibility, user portfolio history.
"""
import sqlite3
import json
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "/home/ubuntu/investwise/investwise.db"
logger = logging.getLogger("portfolio_memory")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_portfolio_tables():
    """Create portfolio memory tables (idempotent)."""
    with _conn() as con:
        # ── Snapshots: one row per portfolio upload/analysis ──────────────
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id  TEXT    NOT NULL UNIQUE,
                user_id      TEXT    NOT NULL DEFAULT 'anonymous',
                timestamp    TEXT    NOT NULL,
                holdings     TEXT    NOT NULL,   -- JSON {ticker: weight}
                metrics      TEXT    NOT NULL,   -- JSON {sharpe, beta, cvar, vol, ...}
                data_sources TEXT    NOT NULL,   -- JSON [{source, tickers, period, fetched_at}]
                report_hash  TEXT    NOT NULL,   -- SHA-256 of full report markdown
                report_md    TEXT                -- full markdown (reproducibility)
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_ps_user
            ON portfolio_snapshots(user_id, timestamp DESC)
        """)

        # ── Daily performance tracking (updated by scheduler) ─────────────
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_performance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                tracked_at  TEXT NOT NULL,
                return_since_snap REAL,   -- % return since snapshot date
                current_holdings TEXT,    -- JSON {ticker: current_price}
                pnl_usd      REAL         -- P&L on $100k reference
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_pp_user
            ON portfolio_performance(user_id, tracked_at DESC)
        """)
        con.commit()
    logger.info("[PortfolioMemory] Tables ready")


def save_snapshot(
    user_id: str,
    holdings: dict,          # {ticker: weight (0-1)}
    metrics: dict,           # {sharpe, beta, cvar_95, ann_vol, total_return, ...}
    data_sources: list,      # [{source, tickers, period, fetched_at, price_as_of}]
    report_md: str,
) -> str:
    """
    Save a portfolio analysis snapshot. Returns the snapshot_id (UUID).
    Idempotent: if same holdings + same day → updates existing row.
    """
    snap_id    = str(uuid.uuid4())
    now        = datetime.now(timezone.utc).isoformat()
    today      = now[:10]   # YYYY-MM-DD

    holdings_j   = json.dumps(holdings,     sort_keys=True)
    metrics_j    = json.dumps(metrics,      sort_keys=True)
    sources_j    = json.dumps(data_sources, sort_keys=True)
    report_hash  = hashlib.sha256(report_md.encode()).hexdigest()

    try:
        with _conn() as con:
            # Check: same user, same holdings, same day → update instead of insert
            existing = con.execute(
                """SELECT snapshot_id FROM portfolio_snapshots
                   WHERE user_id = ? AND holdings = ? AND timestamp LIKE ?""",
                (user_id, holdings_j, f"{today}%")
            ).fetchone()

            if existing:
                snap_id = existing["snapshot_id"]
                con.execute(
                    """UPDATE portfolio_snapshots
                       SET metrics=?, data_sources=?, report_hash=?, report_md=?, timestamp=?
                       WHERE snapshot_id=?""",
                    (metrics_j, sources_j, report_hash, report_md, now, snap_id)
                )
            else:
                con.execute(
                    """INSERT INTO portfolio_snapshots
                       (snapshot_id, user_id, timestamp, holdings, metrics, data_sources,
                        report_hash, report_md)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (snap_id, user_id, now, holdings_j, metrics_j,
                     sources_j, report_hash, report_md)
                )
            con.commit()
        logger.info("[PortfolioMemory] Saved snapshot %s for user %s", snap_id, user_id)
        return snap_id
    except Exception as e:
        logger.error("[PortfolioMemory] Save failed: %s", e)
        return snap_id  # return ID even on error (audit trail written to report)


def get_user_snapshots(user_id: str, limit: int = 20) -> list[dict]:
    """Return most recent portfolio snapshots for a user (no report_md to keep response small)."""
    try:
        with _conn() as con:
            rows = con.execute(
                """SELECT snapshot_id, timestamp, holdings, metrics, data_sources, report_hash
                   FROM portfolio_snapshots
                   WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit)
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "snapshot_id":  r["snapshot_id"],
                "timestamp":    r["timestamp"],
                "holdings":     json.loads(r["holdings"]),
                "metrics":      json.loads(r["metrics"]),
                "data_sources": json.loads(r["data_sources"]),
                "report_hash":  r["report_hash"],
            })
        return result
    except Exception as e:
        logger.error("[PortfolioMemory] get_user_snapshots failed: %s", e)
        return []


def get_snapshot(snapshot_id: str) -> Optional[dict]:
    """Return full snapshot including report_md (for report reproduction)."""
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM portfolio_snapshots WHERE snapshot_id = ?",
                (snapshot_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "snapshot_id":  row["snapshot_id"],
            "user_id":      row["user_id"],
            "timestamp":    row["timestamp"],
            "holdings":     json.loads(row["holdings"]),
            "metrics":      json.loads(row["metrics"]),
            "data_sources": json.loads(row["data_sources"]),
            "report_hash":  row["report_hash"],
            "report_md":    row["report_md"],
        }
    except Exception as e:
        logger.error("[PortfolioMemory] get_snapshot failed: %s", e)
        return None


def get_performance_history(user_id: str, limit: int = 30) -> list[dict]:
    """
    Return Sharpe / Return / Beta over time across snapshots.
    Used for portfolio evolution chart.
    """
    try:
        with _conn() as con:
            rows = con.execute(
                """SELECT snapshot_id, timestamp, metrics
                   FROM portfolio_snapshots
                   WHERE user_id = ?
                   ORDER BY timestamp ASC LIMIT ?""",
                (user_id, limit)
            ).fetchall()
        result = []
        for r in rows:
            m = json.loads(r["metrics"])
            result.append({
                "snapshot_id": r["snapshot_id"],
                "date":        r["timestamp"][:10],
                "sharpe":      m.get("sharpe"),
                "beta":        m.get("beta"),
                "cvar_95":     m.get("cvar_95"),
                "total_return": m.get("total_return"),
                "ann_vol":     m.get("ann_vol"),
            })
        return result
    except Exception as e:
        logger.error("[PortfolioMemory] get_performance_history failed: %s", e)
        return []


def compare_snapshots(snap_id_a: str, snap_id_b: str) -> Optional[dict]:
    """
    Compare two snapshots: allocation drift + metric changes.
    Returns structured diff for display.
    """
    a = get_snapshot(snap_id_a)
    b = get_snapshot(snap_id_b)
    if not a or not b:
        return None

    all_tickers = sorted(set(a["holdings"]) | set(b["holdings"]))
    allocation_diff = {}
    for t in all_tickers:
        wa = a["holdings"].get(t, 0)
        wb = b["holdings"].get(t, 0)
        allocation_diff[t] = {
            "before": round(wa * 100, 2),
            "after":  round(wb * 100, 2),
            "delta":  round((wb - wa) * 100, 2),
        }

    def _mdiff(key):
        av = a["metrics"].get(key)
        bv = b["metrics"].get(key)
        if av is None or bv is None:
            return None
        return {"before": av, "after": bv, "delta": round(bv - av, 4)}

    return {
        "snapshot_a":       snap_id_a,
        "snapshot_b":       snap_id_b,
        "date_a":           a["timestamp"][:10],
        "date_b":           b["timestamp"][:10],
        "allocation_diff":  allocation_diff,
        "metric_diff": {
            "sharpe":       _mdiff("sharpe"),
            "beta":         _mdiff("beta"),
            "cvar_95":      _mdiff("cvar_95"),
            "ann_vol":      _mdiff("ann_vol"),
            "total_return": _mdiff("total_return"),
        },
    }


# Initialise on import
init_portfolio_tables()
