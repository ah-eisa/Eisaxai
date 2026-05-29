#!/usr/bin/env python3
"""
Phase 3 before/after capture for production /v1/report.
Read-only against the API: POSTs report requests, saves JSON + extracted
validation fields. Concurrency capped at 2 (workers=2) to avoid 429s.

Usage:
    phase3_capture.py <label> <outdir>
        label  : "before" | "after" (used in filenames)
        outdir : directory to write <label>_<TICKER>.json + summary
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

API = "http://127.0.0.1:8000/v1/report"
TICKERS = ["ADNOCGAS.AE", "EMAAR.AE", "COMI.CA", "AAPL", "BTC-USD", "GC=F"]

def _token():
    with open("/home/ubuntu/investwise/.env") as f:
        for line in f:
            if line.startswith("SECURE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

TOKEN = _token()

def _extract(body: str, rj: dict) -> dict:
    """Pull the fields we validate from the report body + report_json."""
    def srch(pat):
        m = re.search(pat, body, re.IGNORECASE)
        return m.group(1).strip() if m else None
    meta = (rj or {}).get("report_meta") or {}
    return {
        "h1_ticker": srch(r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)"),
        "live_price_line": srch(r"Live Price:\*\*\s*([^\n|]+)"),
        "sector_line": srch(r"Sector:\*\*\s*([^\n|]+)"),
        "verdict_meta": meta.get("verdict") or meta.get("recommendation"),
        "confidence_meta": meta.get("confidence_label") or meta.get("confidence"),
        "risk_meta": meta.get("overall_risk_label") or meta.get("risk_label"),
        "score_meta": meta.get("eisax_score") or meta.get("score"),
        "mentions_brent": bool(re.search(r"\bbrent\b", body, re.IGNORECASE)),
        "mentions_oil": bool(re.search(r"\boil\b", body, re.IGNORECASE)),
        "sma200_values": sorted(set(re.findall(r"SMA\s*200[^\d]{0,12}([\d,]+\.\d+)", body, re.IGNORECASE)))[:6],
        "currency_codes": sorted(set(re.findall(r"\b(AED|EGP|SAR|USD|KWD|QAR)\b", body)))[:6],
        "body_len": len(body),
    }

def run_one(ticker: str, label: str, outdir: str) -> dict:
    payload = json.dumps({
        "symbol": ticker, "market": "", "language": "en",
        "report_type": "pilot_report",
    }).encode()
    req = urllib.request.Request(
        API, data=payload, method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": TOKEN},
    )
    t0 = time.perf_counter()
    rec = {"ticker": ticker, "label": label}
    try:
        with urllib.request.urlopen(req, timeout=320) as resp:
            status = resp.status
            data = json.loads(resp.read().decode())
        body = data.get("html_report") or ""
        rj = data.get("report_json") or {}
        safe = ticker.replace(".", "_").replace("=", "_").replace("-", "_")
        with open(os.path.join(outdir, f"{label}_{safe}.json"), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        rec.update({"http": status, "ok": True,
                    "elapsed_s": round(time.perf_counter() - t0, 1)})
        rec.update(_extract(body, rj))
    except urllib.error.HTTPError as e:
        rec.update({"http": e.code, "ok": False,
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                    "error": e.read().decode()[:200]})
    except Exception as e:
        rec.update({"http": None, "ok": False,
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                    "error": repr(e)[:200]})
    return rec

def main():
    label, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_one, t, label, outdir): t for t in TICKERS}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    results.sort(key=lambda r: TICKERS.index(r["ticker"]))
    with open(os.path.join(outdir, f"{label}_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"=== {label} capture done: {len(results)} tickers ===", flush=True)

if __name__ == "__main__":
    main()
