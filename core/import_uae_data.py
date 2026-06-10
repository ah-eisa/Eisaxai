"""
import_uae_data.py — يستورد داتا DFM وADX في SQLite
"""
import pandas as pd
import sqlite3
import re
from pathlib import Path

DB_PATH = "/home/ubuntu/investwise/core/investwise.db"
DFM_CSV = "/home/ubuntu/investwise/core/DFM.csv"
ADX_CSV = "/home/ubuntu/investwise/core/ADX.csv"

def parse_number(val):
    """يحول '25.25B' → 25250000000"""
    if not val or val.strip() == "":
        return None
    val = val.strip().replace(",", "")
    try:
        if val.endswith("B"): return float(val[:-1]) * 1_000_000_000
        if val.endswith("M"): return float(val[:-1]) * 1_000_000
        if val.endswith("K"): return float(val[:-1]) * 1_000
        return float(val)
    except:
        return None

def import_dfm(conn):
    df = pd.read_csv(DFM_CSV)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Name"] = df["Name"].str.strip().str.strip('"')

    conn.execute("""
        CREATE TABLE IF NOT EXISTS uae_fundamentals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            exchange    TEXT,
            avg_vol_3m  REAL,
            market_cap  REAL,
            revenue     REAL,
            pe_ratio    REAL,
            beta        REAL,
            updated_at  TEXT DEFAULT (date('now'))
        )
    """)
    conn.execute("DELETE FROM uae_fundamentals WHERE exchange='DFM'")

    rows = 0
    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO uae_fundamentals (name, exchange, avg_vol_3m, market_cap, revenue, pe_ratio, beta)
            VALUES (?, 'DFM', ?, ?, ?, ?, ?)
        """, (
            row["Name"],
            parse_number(str(row.get("Average Vol. (3m)", ""))),
            parse_number(str(row.get("Market Cap", ""))),
            parse_number(str(row.get("Revenue", ""))),
            parse_number(str(row.get("P/E Ratio", ""))),
            parse_number(str(row.get("Beta", ""))),
        ))
        rows += 1

    print(f"✅ DFM: {rows} شركة محفوظة")

def import_adx(conn):
    df = pd.read_csv(ADX_CSV)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Name"] = df["Name"].str.strip().str.strip('"')

    conn.execute("""
        CREATE TABLE IF NOT EXISTS uae_signals (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            exchange   TEXT,
            signal_hourly  TEXT,
            signal_daily   TEXT,
            signal_weekly  TEXT,
            signal_monthly TEXT,
            updated_at TEXT DEFAULT (date('now'))
        )
    """)
    conn.execute("DELETE FROM uae_signals WHERE exchange='ADX'")

    rows = 0
    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO uae_signals (name, exchange, signal_hourly, signal_daily, signal_weekly, signal_monthly)
            VALUES (?, 'ADX', ?, ?, ?, ?)
        """, (
            row["Name"],
            row.get("Hourly", ""),
            row.get("Daily", ""),
            row.get("Weekly", ""),
            row.get("Monthly", ""),
        ))
        rows += 1

    print(f"✅ ADX: {rows} شركة محفوظة")

def print_summary(conn):
    dfm_count = conn.execute("SELECT COUNT(*) FROM uae_fundamentals WHERE exchange='DFM'").fetchone()[0]
    adx_count = conn.execute("SELECT COUNT(*) FROM uae_signals WHERE exchange='ADX'").fetchone()[0]

    print(f"\n📊 DATABASE SUMMARY:")
    print(f"  DFM Fundamentals: {dfm_count} شركة")
    print(f"  ADX Signals:      {adx_count} شركة")

    print(f"\n🏆 أكبر 5 شركات DFM بالـ Market Cap:")
    rows = conn.execute("""
        SELECT name, market_cap, pe_ratio, beta
        FROM uae_fundamentals
        WHERE exchange='DFM' AND market_cap IS NOT NULL
        ORDER BY market_cap DESC LIMIT 5
    """).fetchall()
    for r in rows:
        mc = f"{r[1]/1e9:.1f}B" if r[1] else "N/A"
        pe = f"{r[2]:.1f}" if r[2] else "N/A"
        print(f"  {r[0]}: Market Cap={mc} AED | P/E={pe}")

    print(f"\n📈 Strong Buy اليوم في ADX:")
    rows = conn.execute("""
        SELECT name, signal_daily, signal_weekly
        FROM uae_signals
        WHERE exchange='ADX' AND signal_daily='Strong Buy'
        LIMIT 8
    """).fetchall()
    for r in rows:
        print(f"  {r[0]}: Daily={r[1]} | Weekly={r[2]}")

if __name__ == "__main__":
    # انسخ الـ CSV للـ data folder
    import shutil
    Path("/home/ubuntu/investwise/data").mkdir(parents=True, exist_ok=True)
    shutil.copy("/home/ubuntu/investwise/core/DFM.csv", DFM_CSV) if Path(DFM_CSV) != Path("/home/ubuntu/investwise/core/DFM.csv") else None

    conn = sqlite3.connect(DB_PATH)
    import_dfm(conn)
    import_adx(conn)
    conn.commit()
    print_summary(conn)
    conn.close()
    print("\n✅ Done!")