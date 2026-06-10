from pathlib import Path


def main() -> None:
    p = Path("/home/ubuntu/investwise/arab_dashboard_fixed.py")
    s = p.read_text(encoding="utf-8")
    orig = s

    anchor = (
        'def now_dubai_str() -> str:\n'
        '    return pd.Timestamp.now(tz=DUBAI_TZ).strftime("%Y-%m-%d %H:%M:%S")\n'
        '\n'
        '\n'
        'def ask_eisa_ai(messages, market_context: str, stock_count: int, language: str) -> str:\n'
    )
    insert = (
        'def now_dubai_str() -> str:\n'
        '    return pd.Timestamp.now(tz=DUBAI_TZ).strftime("%Y-%m-%d %H:%M:%S")\n'
        '\n'
        '\n'
        'def _tokenize_query(text: str) -> list[str]:\n'
        '    """Simple tokenizer for relevance scoring (Arabic + English friendly)."""\n'
        '    if not text:\n'
        '        return []\n'
        '    cleaned = (\n'
        '        str(text).lower()\n'
        '        .replace(",", " ")\n'
        '        .replace(".", " ")\n'
        '        .replace(":", " ")\n'
        '        .replace(";", " ")\n'
        '        .replace("|", " ")\n'
        '        .replace("/", " ")\n'
        '        .replace("\\\\", " ")\n'
        '        .replace("(", " ")\n'
        '        .replace(")", " ")\n'
        '    )\n'
        '    raw_tokens = [t.strip() for t in cleaned.split() if t.strip()]\n'
        '    stop = {\n'
        '        "the", "and", "for", "with", "from", "that", "this", "what", "which",\n'
        '        "how", "are", "is", "in", "on", "to", "of", "a", "an",\n'
        '        "stock", "stocks", "market", "markets",\n'
        '        "في", "من", "على", "الى", "إلى", "ما", "هو", "هي", "عن", "مع", "او", "أو", "كل",\n'
        '        "سهم", "اسهم", "الاسهم", "السوق", "الأسواق",\n'
        '    }\n'
        '    return [t for t in raw_tokens if len(t) > 1 and t not in stop]\n'
        '\n'
        '\n'
        'def build_ai_market_context(df: "pd.DataFrame", user_query: str, max_rows: int = 18) -> tuple[str, int]:\n'
        '    """Build relevance-ranked context instead of sending arbitrary top rows."""\n'
        '    if df is None or df.empty:\n'
        '        return "No market data available.", 0\n'
        '\n'
        '    cols = [\n'
        '        "name", "market", "close", "change", "RSI",\n'
        '        "price_earnings_ttm", "dividend_yield_recent", "sector", "SMA50", "SMA200"\n'
        '    ]\n'
        '    use_cols = [c for c in cols if c in df.columns]\n'
        '    work = df[use_cols].copy()\n'
        '\n'
        '    for c in ("close", "change", "RSI", "price_earnings_ttm", "dividend_yield_recent", "SMA50", "SMA200"):\n'
        '        if c in work.columns:\n'
        '            work[c] = pd.to_numeric(work[c], errors="coerce")\n'
        '\n'
        '    q = (user_query or "").lower()\n'
        '    tokens = _tokenize_query(user_query)\n'
        '\n'
        '    wants_oversold = any(k in q for k in ("oversold", "ذروة بيع", "undervalued rsi"))\n'
        '    wants_overbought = any(k in q for k in ("overbought", "ذروة شراء"))\n'
        '    wants_dividend = any(k in q for k in ("dividend", "yield", "توزيع", "عائد"))\n'
        '    wants_value = any(k in q for k in ("pe", "p/e", "valuation", "value", "قيمة", "تقييم"))\n'
        '    wants_momentum = any(k in q for k in ("momentum", "trend", "زخم", "اتجاه"))\n'
        '    wants_gainers = any(k in q for k in ("gainer", "top gain", "ارتفاع", "صاعد"))\n'
        '    wants_losers = any(k in q for k in ("loser", "drop", "هبوط", "هابط"))\n'
        '\n'
        '    score = pd.Series(0.0, index=work.index)\n'
        '    if "change" in work.columns:\n'
        '        score += work["change"].fillna(0).abs() * 0.08\n'
        '    if "dividend_yield_recent" in work.columns:\n'
        '        score += work["dividend_yield_recent"].fillna(0) * 0.05\n'
        '\n'
        '    if wants_oversold and "RSI" in work.columns:\n'
        '        score += ((35 - work["RSI"].fillna(50)).clip(lower=0, upper=25) / 8.0)\n'
        '    if wants_overbought and "RSI" in work.columns:\n'
        '        score += ((work["RSI"].fillna(50) - 65).clip(lower=0, upper=25) / 8.0)\n'
        '    if wants_dividend and "dividend_yield_recent" in work.columns:\n'
        '        score += (work["dividend_yield_recent"].fillna(0).clip(lower=0, upper=12) / 2.0)\n'
        '    if wants_value and "price_earnings_ttm" in work.columns:\n'
        '        pe = work["price_earnings_ttm"].replace(0, pd.NA)\n'
        '        score += ((22 - pe.fillna(22)).clip(lower=0, upper=22) / 4.0)\n'
        '    if wants_momentum and "change" in work.columns:\n'
        '        score += (work["change"].fillna(0).abs().clip(lower=0, upper=8) / 1.5)\n'
        '    if wants_gainers and "change" in work.columns:\n'
        '        score += work["change"].fillna(0).clip(lower=0, upper=10) / 1.2\n'
        '    if wants_losers and "change" in work.columns:\n'
        '        score += ((-work["change"].fillna(0)).clip(lower=0, upper=10) / 1.2)\n'
        '\n'
        '    if tokens:\n'
        '        name_blob = (\n'
        '            work.get("name", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()\n'
        '            + " "\n'
        '            + work.get("sector", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()\n'
        '            + " "\n'
        '            + work.get("market", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()\n'
        '        )\n'
        '        for tk in tokens[:8]:\n'
        '            score += name_blob.str.contains(tk, regex=False).astype(float) * 2.5\n'
        '\n'
        '    work["_score"] = score\n'
        '    ranked = work.sort_values("_score", ascending=False).head(max_rows).copy()\n'
        '    ranked = ranked.drop(columns=["_score"], errors="ignore")\n'
        '\n'
        '    preview = ranked.copy()\n'
        '    for c in ("close", "change", "RSI", "price_earnings_ttm", "dividend_yield_recent"):\n'
        '        if c in preview.columns:\n'
        '            preview[c] = preview[c].round(2)\n'
        '\n'
        '    csv_context = preview.to_csv(index=False)\n'
        '    fact_lines = []\n'
        '    for _, row in preview.head(10).iterrows():\n'
        '        fact_lines.append(\n'
        '            f"- {row.get(\'name\',\'N/A\')} ({row.get(\'market\',\'N/A\')}) | "\n'
        '            f"Price {row.get(\'close\',\'N/A\')} | Change {row.get(\'change\',\'N/A\')}% | "\n'
        '            f"RSI {row.get(\'RSI\',\'N/A\')} | P/E {row.get(\'price_earnings_ttm\',\'N/A\')} | "\n'
        '            f"Div {row.get(\'dividend_yield_recent\',\'N/A\')}% | Sector {row.get(\'sector\',\'N/A\')}"\n'
        '        )\n'
        '\n'
        '    context_block = (\n'
        '        f"Relevant market slice ({len(preview)} rows selected from {len(df)} filtered rows):\\n"\n'
        '        f"{csv_context}\\n"\n'
        '        f"Data Fact Cards:\\n" + "\\n".join(fact_lines)\n'
        '    )\n'
        '    return context_block, len(preview)\n'
        '\n'
        '\n'
        'def ask_eisa_ai(messages, market_context: str, stock_count: int, language: str) -> str:\n'
    )

    if anchor not in s:
        raise SystemExit("anchor_not_found_for_insert")
    s = s.replace(anchor, insert, 1)

    s = s.replace(
        "You are EISA AI, the official market intelligence assistant for EisaX.",
        "You are Eisax, the official market intelligence assistant for EisaX.",
        1,
    )
    s = s.replace(
        "If asked who you are, reply that you are EISA AI.",
        "If asked who you are, reply that you are Eisax.",
        1,
    )
    s = s.replace(
        "Use the provided market data context when mentioning prices, RSI, daily change, sectors, or valuation metrics.",
        "Use ONLY the provided market data context for prices, RSI, daily change, sector, dividend yield, and valuation.",
        1,
    )
    s = s.replace(
        "When citing specific stocks, include price, RSI, and daily change when available.",
        "Every stock mention MUST include inline evidence in this style:\n[Data: Price=..., Change=...%, RSI=..., P/E=..., Div=...%].",
        1,
    )
    s = s.replace(
        "If the data is missing or insufficient, say that clearly instead of guessing.\n\nMarket data (filtered, {stock_count} stocks):",
        "If the data is missing or insufficient, say that clearly instead of guessing.\nUse a strict structured format:\n1) Executive Summary\n2) Top Opportunities (max 3, each with data evidence)\n3) Risks / Watchouts\n4) Monitoring Checklist (non-execution, what to track next)\nKeep the answer concise and data-first.\n\nMarket data (filtered, {stock_count} stocks):",
        1,
    )

    old_context_block = (
        '    # Build compact market context (top 50 rows to stay within token limits)\n'
        '    context_cols = ["name","market","close","change","RSI","price_earnings_ttm","dividend_yield_recent","sector"]\n'
        '    ctx_df = filtered_df[[c for c in context_cols if c in filtered_df.columns]].head(50).round(2)\n'
        '    market_context = ctx_df.to_csv(index=False)\n'
        '\n'
    )
    if old_context_block not in s:
        raise SystemExit("old_context_block_not_found")
    s = s.replace(old_context_block, "", 1)

    old_messages = '        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ai_history]\n\n'
    new_messages = (
        '        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ai_history]\n'
        '        market_context, selected_count = build_ai_market_context(\n'
        '            filtered_df,\n'
        '            user_query,\n'
        '            max_rows=18,\n'
        '        )\n'
        '\n'
    )
    if old_messages not in s:
        raise SystemExit("old_messages_not_found")
    s = s.replace(old_messages, new_messages, 1)

    s = s.replace("                    stock_count=len(filtered_df),", "                    stock_count=selected_count,", 1)

    old_append = '        st.session_state.ai_history.append({"role": "assistant", "content": ai_reply})\n'
    new_append = (
        '        note = f"_Context used: {selected_count} relevant rows from {len(filtered_df)} filtered stocks._"\n'
        '        ai_reply = f"{ai_reply}\\n\\n{note}"\n'
        '        st.session_state.ai_history.append({"role": "assistant", "content": ai_reply})\n'
    )
    if old_append not in s:
        raise SystemExit("old_append_not_found")
    s = s.replace(old_append, new_append, 1)

    if s == orig:
        raise SystemExit("no_change")

    p.write_text(s, encoding="utf-8")
    print("patched_ok", len(orig), "->", len(s))


if __name__ == "__main__":
    main()
