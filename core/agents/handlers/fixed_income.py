# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.intent_classifier import IntentClassifier
logger = logging.getLogger(__name__)


class FixedIncomeMixin:
    def _handle_optimize(self, sid: str, mem: Dict[str, Any], msg: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Wraps _handle_optimize_inner with a 240-second timeout to prevent hangs."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        try:
            with ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(self._handle_optimize_inner, sid, mem, msg, settings)
                return _fut.result(timeout=240)
        except _TE:
            logger.warning("[_handle_optimize] Timed out after 240s — returning fallback")
            return {
                "type": "chat.reply",
                "reply": (
                    "⏱️ Portfolio optimization timed out (>4 min). This usually means:\n"
                "- Market data APIs are slow right now\n"
                "- Too many simultaneous requests\n\n"
                "**Please try again in 30 seconds.** If this keeps happening, try a simpler request like: 'optimize NVDA MSFT AAPL GOOGL'\n"
                    "**Quick suggestion while we optimize:**\n"
                    "- For aggressive growth: QQQ (35%), NVDA (20%), MSFT (15%), AMZN (15%), TSLA (15%)\n"
                    "- For balanced: SPY (40%), QQQ (20%), BND (20%), GLD (10%), VNQ (10%)\n"
                    "- For conservative: BND (50%), SPY (30%), GLD (10%), VYM (10%)\n\n"
                    "Try again in a moment for a full optimized analysis."
                ),
                "data": {}
            }
        except Exception as e:
            logger.error("[_handle_optimize] Error: %s", e)
            return {"type": "error", "reply": f"Portfolio optimization error: {e}", "data": {}}

    def _handle_egypt_bonds(self, message: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch live Egyptian government bond yield curve + macro context,
        then ask the LLM to produce a CIO-style analysis.
        """
        try:
            bond_data = get_egypt_bond_data()
            bond_context = format_egypt_bonds_for_prompt(bond_data)

            model = settings.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)

            system_prompt = (
                "You are EisaX, an institutional CIO specialising in emerging-market fixed income. "
                "You have been provided with live data for Egyptian government bonds and T-bills. "
                "Produce a concise, data-driven analysis covering:\n"
                "1. Current yield curve shape (normal / inverted / flat) and what it signals\n"
                "2. Real yield vs. inflation context (if data available)\n"
                "3. Relative value: short-end vs. long-end opportunity\n"
                "4. EGP currency risk for foreign investors\n"
                "5. Key risks and catalysts (CBE rate path, IMF programme, FX reserves)\n"
                "6. Investment verdict: who should buy, in what maturity, and why\n\n"
                "Use markdown with headers. Be specific with numbers. "
                "Today's date: " + datetime.now().strftime("%B %d, %Y")
            )

            client = self.client_factory()
            user_message = f"{bond_context}\n\nUser question: {message}"
            response = client.create_completion(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ]
            )
            reply = response.choices[0].message.content if hasattr(response, "choices") else str(response)

            # Attach raw data for potential export
            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {
                    "egypt_bonds": bond_data,
                    "source": bond_data.get("source"),
                    "fetched_at": bond_data.get("fetched_at"),
                }
            }

        except Exception as e:
            logger.error("[_handle_egypt_bonds] Error: %s", e)
            return {
                "type": "chat.reply",
                "reply": (
                    "⚠️ Could not fetch live Egyptian bond data right now. "
                    "Common data points:\n\n"
                    "- **CBE Overnight Deposit Rate**: ~27.25% (as of early 2025)\n"
                    "- **91-day T-bill**: ~27–28%\n"
                    "- **1-year T-bill**: ~28–29%\n"
                    "- **5-year bond**: ~27.5–29%\n"
                    "- **10-year bond**: ~27–28.5%\n\n"
                    "Please check [CBE](https://www.cbe.org.eg) or "
                    "[Investing.com Egypt Bonds](https://www.investing.com/rates-bonds/egypt-government-bonds) "
                    "for the latest figures."
                ),
                "data": {}
            }

    def _handle_fixed_income(self, message: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full Sukuk & Bond analysis by ISIN.
        1. Extract ISIN from message
        2. Fetch metadata via OpenFIGI + FMP + FRED benchmarks
        3. Compute EisaX Fixed Income Score
        4. Generate CIO-style report via LLM
        Falls back gracefully when no ISIN is found (general fixed-income question).
        """
        try:
            model = settings.get("model") or os.getenv("MODEL_NAME", config.DEFAULT_MODEL)
            lang  = detect_sukuk_query_language(message)
            isin  = extract_isin(message)

            # ── Case A: specific ISIN provided ───────────────────────────────
            if isin:
                logger.info("[fixed_income] ISIN detected: %s", isin)
                data  = get_instrument_data(isin, hint_text=message)
                score = compute_fi_score(data)
                fi_context = format_fi_for_prompt(data, score)

                is_sukuk   = data.get("is_sukuk", False)
                name_str   = data.get("name") or isin

                # ── Detect Bond ETF — fundamentally different methodology ──
                sec_type_raw = (data.get("security_type") or "").lower()
                is_bond_etf  = any(kw in sec_type_raw for kw in ("etf", "fund", "exchange traded"))

                if is_bond_etf:
                    # Bond ETF — NO seniority, covenants, YTM-in-isolation analysis
                    # Focus on: fund mechanics, expense ratio, duration, credit spread index, peers
                    is_hy_etf = any(kw in (name_str + sec_type_raw).lower()
                                    for kw in ("high yield", "junk", "hyg", "jnk", "faln", "angl", "bb", "b rated"))
                    peer_block = (
                        "\n### 4. Peer Comparison (HY Bond ETF Universe)\n"
                        "   Compare against primary HY ETF peers:\n"
                        "   - HYG (iShares iBoxx $ High Yield): largest, most liquid, ~8yr duration\n"
                        "   - JNK (SPDR Bloomberg HY Bond): similar but slightly higher yield / lower quality tilt\n"
                        "   - FALN (iShares Fallen Angels): higher quality bias (BB, recently downgraded)\n"
                        "   - ANGL (VanEck Fallen Angel): similar fallen angel approach, different index\n"
                        "   Discuss: yield spread vs peers, AUM, expense ratio, index methodology differences\n\n"
                    ) if is_hy_etf else (
                        "\n### 4. Peer Comparison (Bond ETF Universe)\n"
                        "   Compare against relevant IG bond ETF peers:\n"
                        "   - LQD (iShares IG Corp): broad IG corporate benchmark\n"
                        "   - VCIT (Vanguard IG Corp): lower expense ratio alternative\n"
                        "   - IGIB (iShares Intermediate IG): intermediate duration focus\n"
                        "   Discuss: yield spread vs peers, duration difference, expense ratio, index coverage\n\n"
                    )

                    system_prompt = (
                        f"You are EisaX, an institutional CIO specialising in fixed income ETF analysis.\n"
                        f"You have been given live data for **{name_str}**, which is a Bond ETF — NOT an individual bond.\n\n"
                        f"⚠️ IMPORTANT METHODOLOGY NOTE:\n"
                        f"  This is a Bond ETF (a diversified fund). Do NOT apply individual bond analysis:\n"
                        f"  - Do NOT discuss seniority or covenants (those apply to individual bonds)\n"
                        f"  - Do NOT apply YTM the same way (ETFs hold hundreds of bonds; use SEC 30-day yield)\n"
                        f"  - DO focus on: expense ratio, duration, index tracked, AUM/liquidity, credit spread index\n\n"
                        f"Produce a CIO-grade Bond ETF report with EXACTLY these sections:\n\n"
                        f"## 📊 Bond ETF Analysis — {name_str}\n\n"
                        f"### 1. Fund Overview\n"
                        f"   - Fund type (HY / IG / Treasury / TIPS / Fallen Angel etc.)\n"
                        f"   - Index tracked, number of holdings, AUM, expense ratio\n"
                        f"   - Exchange listing and daily volume (liquidity)\n\n"
                        f"### 2. Yield & Income Analysis\n"
                        f"   - SEC 30-day yield vs distribution yield (not YTM)\n"
                        f"   - Credit spread of underlying index vs benchmark (US10Y or IG index)\n"
                        f"   - Income attractiveness relative to risk-free rate\n\n"
                        f"### 3. Duration & Rate Risk\n"
                        f"   - Effective duration of the fund (not maturity of individual bonds)\n"
                        f"   - Price sensitivity: estimated NAV impact per 1% rate move\n"
                        f"   - Positioning in current rate environment\n\n"
                        f"{peer_block}"
                        f"### 5. Credit Quality & Default Risk\n"
                        f"   - Weighted average credit rating and HY/IG split\n"
                        f"   - Default rate sensitivity in recession vs base case\n"
                        f"   - Spread widening risk in a credit crunch\n\n"
                        f"### 6. Investment Verdict\n"
                        f"   - Clear BUY / HOLD / AVOID with conviction level\n"
                        f"   - Target investor profile (income, tactical, institutional)\n"
                        f"   - Entry conditions and key risk triggers\n\n"
                        f"## 🎯 EisaX Fixed Income Score: {score['total']}/100  {score['verdict_label']} {score['verdict']}\n\n"
                        f"Copy the scorecard table from the data block EXACTLY — do not add a weighted column.\n\n"
                        f"Use clear markdown. Be specific with numbers from the data. "
                        f"Today: {datetime.now().strftime('%B %d, %Y')}"
                    )

                else:
                    # ── Individual Bond / Sukuk ──────────────────────────────
                    instrument_type = "Sukuk" if is_sukuk else "Bond"

                    system_prompt = (
                        f"You are EisaX, an institutional CIO specialising in fixed income and Islamic finance.\n"
                        f"You have been given live instrument data for {name_str}.\n\n"
                        f"Produce a professional CIO-grade report with EXACTLY these sections:\n\n"
                        f"## 📊 {instrument_type} Analysis — {name_str}\n\n"
                        f"### 1. Instrument Overview\n"
                        f"   - Summarise key terms (ISIN, issuer, coupon, maturity, currency, exchange)\n"
                        f"   - Security type and market sector\n\n"
                        f"### 2. Yield Analysis\n"
                        f"   - Current coupon rate vs benchmark yields provided\n"
                        f"   - Spread in basis points (bps) and what it implies\n"
                        f"   - Estimated YTM commentary (based on time to maturity)\n\n"
                        f"### 3. Credit Risk Assessment\n"
                        f"   - Issuer credit profile and sovereign/corporate context\n"
                        f"   - Country rating — NOTE: if the rating has a staleness warning in the data, explicitly flag it\n"
                        f"   - Key downside risks (default, FX, liquidity)\n\n"
                    )

                    if is_sukuk:
                        system_prompt += (
                            f"### 4. Sukuk Structure\n"
                            f"   - Structure type (Ijara / Murabaha / Wakala / Mudarabah / Musharaka)\n"
                            f"   - Asset backing and SPV mechanics (if inferable)\n"
                            f"   - Sharia compliance confidence level\n"
                            f"   - How periodic distributions compare to conventional coupon\n\n"
                        )
                    else:
                        system_prompt += (
                            f"### 4. Bond Structure\n"
                            f"   - Seniority (Senior Unsecured / Subordinated / Secured)\n"
                            f"   - Covenant and call/put features if available in the data\n"
                            f"   - If seniority is not in the data, state 'Not available from ISIN lookup'\n\n"
                        )

                    system_prompt += (
                        f"### 5. FX & Currency Risk\n"
                        f"   - Issue currency vs likely investor base currency\n"
                        f"   - FX peg status (for AED/SAR/QAR etc.)\n"
                        f"   - Hedging context\n\n"
                        f"### 6. Liquidity Assessment\n"
                        f"   - Exchange listing and secondary market depth\n"
                        f"   - Typical bid-ask spread estimate\n"
                        f"   - Suitable investor type (retail / institutional / HNWI)\n\n"
                        f"### 7. Investment Verdict\n"
                        f"   - Clear BUY / HOLD / AVOID recommendation\n"
                        f"   - Who should invest (income seekers, Islamic funds, GCC investors, etc.)\n"
                        f"   - Entry conditions and risk limits\n\n"
                        f"## 🎯 EisaX Fixed Income Score: {score['total']}/100  {score['verdict_label']} {score['verdict']}\n\n"
                        f"Copy the scorecard table from the data block EXACTLY — do not add a weighted column.\n"
                        f"If any factor shows N/A, note that it was excluded from scoring and the total was rescaled.\n\n"
                        f"Use clear markdown. Be specific with numbers from the data. "
                        f"Today: {datetime.now().strftime('%B %d, %Y')}"
                    )

                user_msg = f"{fi_context}\n\nUser question: {message}"

            # ── Case B: general fixed-income question (no ISIN) ──────────────
            else:
                logger.info("[fixed_income] General fixed-income query (no ISIN)")
                system_prompt = (
                    "You are EisaX, an institutional CIO specialising in fixed income, "
                    "Sukuk, government bonds, and Islamic finance products.\n\n"
                    "Answer the user's question with CIO-level depth. Include:\n"
                    "- Relevant yield context for the GCC/MENA region\n"
                    "- Sukuk structures (Ijara, Wakala, Murabaha, etc.) where relevant\n"
                    "- Credit quality and ratings context\n"
                    "- Investment suitability (retail, institutional, Islamic funds)\n\n"
                    "For specific ISIN analysis, ask the user to provide the ISIN "
                    "(e.g. XS1234567890 — 12-character code starting with 2 letters).\n\n"
                    "Use markdown with headers. Be concise and data-driven. "
                    f"Today: {datetime.now().strftime('%B %d, %Y')}"
                )
                user_msg = message
                data  = {}
                score = {}
                fi_context = ""

            # ── LLM call ─────────────────────────────────────────────────────
            client = self.client_factory()
            response = client.create_completion(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            )
            reply = response.choices[0].message.content if hasattr(response, "choices") else str(response)

            return {
                "type": "chat.reply",
                "reply": reply,
                "data": {
                    "instrument": data,
                    "fi_score":   score,
                    "isin":       isin,
                },
            }

        except Exception as e:
            logger.error("[_handle_fixed_income] Error: %s", e, exc_info=True)
            isin_hint = extract_isin(message) or ""
            return {
                "type": "chat.reply",
                "reply": (
                    f"⚠️ Could not complete fixed income analysis"
                    f"{f' for **{isin_hint}**' if isin_hint else ''}.\n\n"
                    f"**Error:** {e}\n\n"
                    f"**For Sukuk/Bond analysis, please provide:**\n"
                    f"- The 12-character ISIN (e.g. `XS1234567890`, `AE000A1RKDU1`)\n"
                    f"- Found on the term sheet, prospectus, or your broker platform\n\n"
                    f"**Free ISIN lookups:**\n"
                    f"- [OpenFIGI](https://www.openfigi.com/search)\n"
                    f"- [ISIN.net](https://www.isin.net)\n"
                    f"- [Nasdaq Dubai](https://www.nasdaqdubai.com)\n"
                ),
                "data": {},
            }

    def _handle_optimize_inner(self, sid: str, mem: Dict[str, Any], msg: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        # ── Clean Pipeline (Step 1-5 architecture) ────────────────────────────
        try:
            from portfolio_pipeline import is_pipeline_request, run as pipeline_run
            if is_pipeline_request(msg):
                logger.info("[Portfolio] Pipeline request detected — routing to clean pipeline")
                report = pipeline_run(msg)
                return {
                    "type": "chat.reply",
                    "reply": report,
                    "data": {"agent": "finance", "analysis_type": "pipeline_report"},
                }
        except Exception as _pe:
            logger.warning("[Portfolio] Pipeline routing failed: %s — falling back to legacy path", _pe)

        # Logic copied/adapted from Orchestrator._handle_optimize

        tickers = IntentClassifier.extract_tickers(msg)
        # "Fresh Start" detection — English + Arabic (ابنى/انشئ/اعمل/جديد)
        fresh_start = any(v in msg.lower() for v in [
            "build", "create", "make", "generate", "new", "start",
            "ابنى", "ابني", "ابن ", "انشئ", "اعمل", "جديد", "بناء", "انشاء", "أنشئ"
        ])
        
        if not tickers and not fresh_start:
            # Only use memory tickers if they match the same market context
            mem_tickers = mem.get("tickers", [])
            if mem_tickers:
                # Detect market of message vs memory tickers
                msg_has_local = any(x in msg.upper() for x in [".SR", ".CA", ".DU", ".AE", ".KW", ".QA", "ARAMCO", "SABIC", "CIB", "EMAAR"])
                mem_has_local = any(t.upper().endswith((".SR", ".CA", ".DU", ".AE", ".KW", ".QA")) for t in mem_tickers)
                # Only use memory if market context matches
                if msg_has_local == mem_has_local:
                    tickers = mem_tickers
                else:
                    logger.info("[Portfolio] Skipping memory tickers — market context mismatch (msg=%s, mem=%s)", 
                                "local" if msg_has_local else "US", "local" if mem_has_local else "US")
        
        # Parse explicit constraints from message
        from core.portfolio import parse_constraints
        constraints = parse_constraints(msg)
        target_return = constraints.get("target_return")
        max_drawdown_val = constraints.get("max_drawdown")

        if not tickers and not state.get_artifact(sid):
            rp = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
            
            if rp == "high" or "aggressive" in msg.lower():
                tickers = pm.recommend_etfs("high")
                method = "max_sharpe"
            elif rp == "low" or "conservative" in msg.lower():
                tickers = pm.recommend_etfs("low")
                method = "min_vol"
            else:
                tickers = pm.recommend_etfs("medium")
                method = "max_sharpe"
            
            start = str(pm.get_param(mem, msg, "start", config.DEFAULT_START))
            w_raw, perf = pm.optimize_and_get_data(
                tickers=tickers, start=start, end=None, method=method,
                min_w=0.0, max_w=0.20, min_assets=4, seed_w=config.DEFAULT_SEED_W, rf=config.DEFAULT_RF,
                target_return=target_return, max_drawdown=max_drawdown_val
            )
            
            # Use RICH STRATEGY GUIDE for generic requests
            guide_md = pm.generate_strategy_guide_llm(
                risk_profile=rp,
                tickers=tickers,
                weights=w_raw,
                performance=perf
            )
            
            # Fix date
            import re as _re
            from datetime import datetime as _dt
            _correct = _dt.now().strftime("%B %d, %Y")
            guide_md = _re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}', _correct, guide_md)
            guide_md = _re.sub(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}', _correct, guide_md)
            guide_md = _re.sub(r'20\d{2}-\d{2}-\d{2}', _dt.now().strftime('%Y-%m-%d'), guide_md)

            # SAVE ARTIFACT
            state.set_artifact(sid, {
                "type": "strategy",
                "content": guide_md,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })
            
            extra_mem = {
                "tickers": tickers, "method": method, "start": start, "end": None,
                "weights": w_raw, "performance": perf,
                "risk": rp,
                "min_w": 0.0, "max_w": 0.35
            }
            
            return {
                "type": "chat.reply",
                "reply": guide_md,
                "data": extra_mem
            }

        # Standard optimization
        start = str(pm.get_param(mem, msg, "start", config.DEFAULT_START))
        end = pm.get_param(mem, msg, "end", None)
        end = None if end in (None, "", "none", "null") else str(end)
        method = str(pm.get_param(mem, msg, "method", mem.get("method") or "max_sharpe")).lower()
        min_w = pm.parse_float(pm.get_param(mem, msg, "min_w", config.DEFAULT_MIN_W), config.DEFAULT_MIN_W)
        max_w = pm.parse_float(pm.get_param(mem, msg, "max_w", config.DEFAULT_MAX_W), config.DEFAULT_MAX_W)
        min_assets = pm.parse_int(pm.get_param(mem, msg, "min_assets", config.DEFAULT_MIN_ASSETS), config.DEFAULT_MIN_ASSETS)
        seed_w = pm.parse_float(pm.get_param(mem, msg, "seed_w", config.DEFAULT_SEED_W), config.DEFAULT_SEED_W)
        rf = pm.parse_float(pm.get_param(mem, msg, "rf", config.DEFAULT_RF), config.DEFAULT_RF)

        # Only call smart_expand when we have fewer than 3 explicit tickers
        if len(tickers) < 3:
            tickers = pm.smart_expand_tickers(msg, tickers)

        # Last-resort fallback: if still < 2 tickers, use risk-based ETF list
        if len(tickers) < 2:
            rp_fb = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
            tickers = pm.recommend_etfs(rp_fb)
            logger.info(f"[Optimize] Ticker fallback triggered → using recommend_etfs({rp_fb}): {tickers}")

        w_raw, perf = pm.optimize_and_get_data(
            tickers=tickers, start=start, end=end, method=method,
            min_w=min_w, max_w=max_w, min_assets=min_assets, seed_w=seed_w, rf=rf,
            target_return=target_return, max_drawdown=max_drawdown_val,
        )
        rp = pm.detect_risk_pref(msg) or mem.get("risk") or "medium"
        reply_text = pm.generate_strategy_guide_llm(
            risk_profile=rp,
            tickers=tickers,
            weights=w_raw,
            performance=perf,
            target_return=target_return,
            max_drawdown=max_drawdown_val,
        )
        
        # SAVE ARTIFACT
        state.set_artifact(sid, {
            "type": "portfolio",
            "content": reply_text,
            "source": "self_generated",
            "exportable": True,
            "timestamp": datetime.now()
        })

        extra_mem = {
            "tickers": tickers, "method": method, "start": start, "end": end,
            "weights": w_raw, "performance": perf,
            "metrics": {},
            "min_w": min_w, "max_w": max_w
        }
            
        return {
            "type": "chat.reply", 
            "reply": reply_text,
            "data": extra_mem
        }


