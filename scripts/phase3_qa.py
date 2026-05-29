#!/usr/bin/env python3
"""
Phase 3 production report QA pack.
Read-only against /v1/report. Captures full JSON per ticker + extracts a
QA row covering price/currency/sector/subtype/verdict/confidence/risk/
SMA/news/json-vs-html consistency. Concurrency capped at 2.

Usage: phase3_qa.py <outdir>
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

API = "http://127.0.0.1:8000/v1/report"
TICKERS = [
    "ADNOCGAS.AE", "EMAAR.AE", "COMI.CA", "AAPL", "BTC-USD",
    "GC=F", "2222.SR", "QNBK.QA", "ALDAR.AE", "FAB.AE",
]

def _token():
    with open("/home/ubuntu/investwise/.env") as f:
        for line in f:
            if line.startswith("SECURE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

TOKEN = _token()

def _dom_verdict(body):
    """Most-frequent decision word in body."""
    counts = {w: len(re.findall(r"\b" + w + r"\b", body))
              for w in ["Buy", "Hold", "Sell", "Reduce", "Accumulate", "Overweight", "Underweight"]}
    counts = {k: v for k, v in counts.items() if v}
    return max(counts, key=counts.get) if counts else None, counts

def _extract(ticker, body, rj):
    meta = (rj or {}).get("report_meta") or {}
    def s(pat):
        m = re.search(pat, body, re.IGNORECASE)
        return m.group(1).strip() if m else None
    dom, vcounts = _dom_verdict(body)
    sma200 = sorted(set(re.findall(r"SMA\s*200[^\d]{0,14}([\d,]+\.\d+)", body, re.I)))[:5]
    sma50  = sorted(set(re.findall(r"SMA\s*50[^\d]{0,14}([\d,]+\.\d+)", body, re.I)))[:5]
    # off-thesis commodity language for non-energy names
    is_energy = bool(re.search(r"\b(adnoc|aramco|energy|oil & gas|petrochemical)\b", body, re.I)) \
                or ticker.startswith(("2222.", "ADNOC"))
    oil_brent_hits = len(re.findall(r"\b(brent|crude oil)\b", body, re.I))
    meta_verdict = (meta.get("verdict") or meta.get("recommendation") or "")
    return {
        "h1_ticker": s(r"#\s*EisaX\s+Intelligence\s+Report:\s*([A-Z0-9.=\-]+)"),
        "price_line": s(r"Live Price:\*\*\s*([^\n|]+)"),
        "sector_line": s(r"Sector:\*\*\s*([^\n|]+)"),
        "currencies": sorted(set(re.findall(r"\b(AED|EGP|SAR|QAR|KWD|USD)\b", body)))[:5],
        "meta_verdict": meta_verdict,
        "body_dom_verdict": dom,
        "verdict_counts": vcounts,
        "verdict_consistent": (str(meta_verdict).lower() == str(dom).lower()) if (meta_verdict and dom) else None,
        "meta_confidence": meta.get("confidence_label") or meta.get("confidence"),
        "meta_risk": meta.get("overall_risk_label") or meta.get("risk_label"),
        "meta_score": meta.get("eisax_score") or meta.get("score"),
        "sma200_body": sma200,
        "sma50_body": sma50,
        "is_energy_ctx": is_energy,
        "oil_brent_hits": oil_brent_hits,
        "offthesis_oil": (oil_brent_hits > 0 and not is_energy),
        "news_items": len(re.findall(r"(?im)^\s*[-*]\s|\bheadline\b|\bnews\b", body)),
        "body_len": len(body),
        "meta_keys": sorted(list(meta.keys()))[:25],
    }

def run_one(ticker, outdir):
    payload = json.dumps({"symbol": ticker, "market": "", "language": "en",
                          "report_type": "pilot_report"}).encode()
    req = urllib.request.Request(API, data=payload, method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": TOKEN})
    t0 = time.perf_counter()
    rec = {"ticker": ticker}
    try:
        with urllib.request.urlopen(req, timeout=320) as resp:
            status = resp.status
            data = json.loads(resp.read().decode())
        body = data.get("html_report") or ""
        rj = data.get("report_json") or {}
        safe = ticker.replace(".", "_").replace("=", "_").replace("-", "_")
        with open(os.path.join(outdir, f"qa_{safe}.json"), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        rec.update({"http": status, "ok": True, "elapsed_s": round(time.perf_counter()-t0, 1)})
        rec.update(_extract(ticker, body, rj))
    except urllib.error.HTTPError as e:
        rec.update({"http": e.code, "ok": False, "elapsed_s": round(time.perf_counter()-t0, 1),
                    "error": e.read().decode()[:200]})
    except Exception as e:
        rec.update({"http": None, "ok": False, "elapsed_s": round(time.perf_counter()-t0, 1),
                    "error": repr(e)[:200]})
    return rec

def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_one, t, outdir): t for t in TICKERS}
        for fut in cf.as_completed(futs):
            r = fut.result(); results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    results.sort(key=lambda r: TICKERS.index(r["ticker"]))
    with open(os.path.join(outdir, "qa_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"=== QA pack done: {len(results)} tickers ===", flush=True)

if __name__ == "__main__":
    main()
