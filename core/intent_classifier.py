from __future__ import annotations
from typing import Dict, Any, List, Optional
import re
import json
import config
from core.llm import get_client

# ── ISIN pattern (2-letter country code + 9 alphanumeric + 1 check digit) ─────
ISIN_RE = re.compile(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b')

# ============================================================
# LOCAL MARKET SUPPORT
# ============================================================
from core.ticker_resolver import (
    TickerResolver,
    extract_local_tickers_from_text,
    has_arabic_stock_context,
    COMBINED_LOCAL_PATTERN,
)

# Initialize resolver once at module level
_resolver = TickerResolver()

# ============================================================
# CONSTANTS & REGEX
# ============================================================
# Updated TICKER_RE to include local market patterns
TICKER_RE = re.compile(
    r'(?:'
    r'\b[A-Za-z]{1,6}(?:-USD)?\b'               # US tickers: AAPL, BTC-USD
    r'|\b\d{4}\.SR\b'                             # Saudi: 2222.SR
    r'|\b[A-Za-z]{2,6}\.CA\b'                     # Egypt: COMI.CA
    r'|\b[A-Za-z]{2,15}\.(?:AE|DU)\b'             # UAE: FAB.AE, ADNOCDIST.AE, ALPHADHABI.AE
    r'|\b[A-Za-z]{2,15}\.KW\b'                    # Kuwait: KFH.KW, BOUBYAN.KW
    r'|\b[A-Za-z]{2,12}\.QA\b'                    # Qatar: QNBK.QA, IQCD.QA
    r'|\^[A-Za-z]{3,10}\b'                        # Indices: ^TASI, ^EGX30
    r')'
)

# Words that should NEVER be treated as tickers
COMMON_WORDS = {
    "A", "AN", "THE", "AND", "OR", "TO", "OF", "IN", "ON", "AT", "BY", "WITH", "FROM",
    "FOR", "ME", "MY", "YOU", "YOUR", "WE", "US", "IT", "IS", "BE", "AM", "ARE", "WAS",
    "WERE", "BEEN", "BEING", "HAVE", "HAS", "HAD", "DO", "DOES", "DID", "WILL", "WOULD",
    "COULD", "SHOULD", "MAY", "MIGHT", "MUST", "SHALL", "CAN", "HE", "SHE", "THEY", "THEM",
    "HIS", "HER", "ITS", "OUR", "THEIR", "THIS", "THAT", "THESE", "THOSE", "WHAT", "WHICH",
    "WHO", "WHOM", "WHOSE", "WHERE", "WHEN", "WHY", "HOW", "IF", "THEN", "ELSE", "BUT",
    "HI", "HEY", "HELLO", "BYE", "GOODBYE", "YES", "NO", "OK", "OKAY", "THANKS", "THANK",
    "PLEASE", "SORRY", "GOOD", "BAD", "GREAT", "NICE", "FINE", "WELL", "SURE", "RIGHT",
    "WRONG", "TRUE", "FALSE", "NEW", "OLD", "BIG", "SMALL", "MUCH", "MANY", "MORE", "LESS",
    "MOST", "LEAST", "SOME", "ANY", "ALL", "EACH", "EVERY", "BOTH", "FEW", "OTHER", "ANOTHER",
    "SUCH", "ONLY", "JUST", "ALSO", "VERY", "TOO", "SO", "NOW", "HERE", "THERE", "UP", "DOWN",
    "TODAY", "TOMORROW", "YESTERDAY", "ALWAYS", "NEVER", "SOMETIMES", "OFTEN", "STILL",
    "ABOUT", "TELL", "KNOW", "THINK", "SEE", "LOOK", "FIND", "GIVE", "TAKE", "COME", "GO",
    "SAY", "SAID", "LIKE", "LOVE", "HATE", "WANT", "NEED", "TRY", "USE", "WORK", "CALL",
    "ASK", "SEEM", "FEEL", "LEAVE", "PUT", "MEAN", "KEEP", "LET", "BEGIN", "SHOW", "HEAR",
    "PLAY", "RUN", "MOVE", "LIVE", "BELIEVE", "HOLD", "BRING", "HAPPEN", "WRITE", "PROVIDE",
    "SIT", "STAND", "LOSE", "PAY", "MEET", "INCLUDE", "CONTINUE", "LEARN", "CHANGE", "LEAD",
    "UNDERSTAND", "WATCH", "FOLLOW", "STOP", "CREATE", "SPEAK", "READ", "ALLOW", "GROW",
    "OPEN", "WALK", "WIN", "OFFER", "REMEMBER", "CONSIDER", "APPEAR", "BUY", "WAIT", "SERVE",
    "DIE", "SEND", "EXPECT", "BUILD", "STAY", "FALL", "CUT", "REACH", "KILL", "REMAIN",
    "STOCK", "STOCKS", "SHARE", "SHARES", "INVEST", "INVESTING", "INVESTMENT", "INVESTOR",
    "MARKET", "MARKETS", "MONEY", "FUND", "FUNDS", "ASSET", "ASSETS", "BOND", "BONDS",
    "RETURN", "RETURNS", "PROFIT", "LOSS", "GAIN", "GAINS", "PRICE", "PRICES", "VALUE",
    "TRADE", "TRADING", "TRADER", "EQUITY", "WEALTH", "CAPITAL", "INCOME", "GROWTH",
    "YIELD", "DIVIDEND", "INTEREST", "RATE", "RATES", "INDEX", "SECTOR", "INDUSTRY",
    "COMPANY", "BUSINESS", "FINANCE", "FINANCIAL", "ECONOMIC", "ECONOMY",
    "QUICK", "BRIEF", "SUMMARY", "SHORT", "FAST", "FULL", "DETAILED", "DEEP", "CIO", "MEMO",
    "DESIGN", "SUGGEST", "RECOMMEND", "CHOOSE", "SELECT", "PICK", "BEST", "TOP", "HIGH",
    "LOW", "MEDIUM", "STRONG", "WEAK", "FREE", "USING", "BASED", "ANALYSIS", "ANALYZE",
    "MAKE", "GET", "HELP", "BALANCED", "DIVERSIFIED", "DIVERSIFY", "PORTFOLIO", 
    "ALLOCATION", "ALLOCATE", "REBALANCE", "HEDGE", "RISK", "SAFE", "CONSERVATIVE", "AGGRESSIVE",
    "OPTIMIZE", "REPORT", "METRICS",
    "MAX", "MIN", "RATIO", "VOL", "VOLATILITY", "SHARPE", "WEIGHTS", "METHOD",
    "START", "END", "RF", "MAX_W", "MIN_W", "MIN_ASSETS", "SEED_W",
    "ADD", "REMOVE", "DELETE", "DROP", "CAP", "SET",
    "EXPORT", "PDF", "DOCX", "WORD", "DOWNLOAD", "SAVE", "PRINT",
    "DELTA", "THETA", "GAMMA", "VEGA", "RHO", "STRIKE", "SPOT", "MODEL", "MONTH", "MONTHS", "IV", 
    "BLACK", "SCHOLES", "OPTIONS", "OPTION", "PUT", "CALL", "FRIEND", "AGREE", "AGREED", "TALK", "TALKED",
    "TELL", "CHARTS", "CHART", "LOOKING", "WITHOUT", "THESE", "THOSE", "WHAT", "WHICH", "ABOUT",
}

CIO_TRIGGER_KEYWORDS = [
    "investment", "portfolio", "trade", "crypto", "asset", "financial",
    "stock", "market", "trading", "crypto", "dividend", "yield"
]

# Arabic keywords that indicate financial/stock intent
ARABIC_FINANCIAL_KEYWORDS = [
    "سهم", "أسهم", "اسهم", "سعر", "أسعار", "اسعار",
    "تحليل", "حلل", "بورصة", "سوق", "تداول", "تاسي",
    "محفظة", "شراء", "بيع", "ارتفاع", "انخفاض", "هبوط",
    "أرباح", "ارباح", "توزيعات", "مكرر", "ربحية",
    "قطاع", "مؤشر", "استثمار", "استثمر",
    "عوائد", "مخاطر", "تنويع", "تحسين",
    # Fixed income / Sukuk (Arabic)
    "صكوك", "صك", "سندات", "سند", "دخل ثابت", "عائد ثابت",
    "كوبون", "استحقاق", "إجارة", "مرابحة", "مضاربة", "وكالة",
    "سندات حكومية", "سندات شركات",
]

# Fixed income English trigger keywords
FIXED_INCOME_KEYWORDS = [
    "sukuk", "isin", "bond analysis", "fixed income", "coupon rate",
    "yield to maturity", "ytm", "trust certificate", "eurobond",
    "sovereign bond", "corporate bond", "islamic bond",
]

PORTFOLIO_SOFT_WORDS = [
    "portfolio", "invest", "allocation", "diversify", "balance",
    "محفظة", "استثمار", "تنويع", "توزيع",
]


class IntentClassifier:
    """
    Centralized intent classification logic.
    Combines strict rule-based detection with LLM-based inference.
    Now supports Arabic/English local market queries.
    """
    
    @staticmethod
    def extract_tickers(text: str) -> List[str]:
        """
        Extract tickers from text — supports:
        1. US tickers (AAPL, MSFT)
        2. Crypto (BTC-USD)
        3. Saudi (2222.SR)
        4. Egypt (COMI.CA)
        5. UAE (FAB.AE, EMAAR.DU)
        6. Indices (^TASI, ^EGX30)
        7. Arabic company names (أرامكو → 2222.SR)
        """
        if not text:
            return []
        
        out: List[str] = []
        
        # ── Step 1: Extract explicit ticker patterns ──────────────
        toks = [t.upper() for t in TICKER_RE.findall(text)]
        for t in toks:
            if t in COMMON_WORDS:
                continue
            if len(t) < 2:
                continue
            # If it's a local ticker pattern, keep as-is
            if any(t.endswith(s) for s in [".SR", ".CA", ".AE", ".DU"]) or t.startswith("^"):
                out.append(t)
            else:
                out.append(t)
        
        # ── Step 2: Resolve Arabic/English names via TickerResolver ──
        words = text.split()
        for i, word in enumerate(words):
            # Skip if word is very short or a common word
            if len(word) < 2:
                continue
            if word.upper() in COMMON_WORDS:
                continue
                
            # Single word lookup
            result = _resolver.resolve_single(word)
            if result and result not in out:
                out.append(result)
            
            # Bigram lookup (two consecutive words)
            if i < len(words) - 1:
                bigram = f"{word} {words[i+1]}"
                result = _resolver.resolve_single(bigram)
                if result and result not in out:
                    out.append(result)
            
            # Trigram (three words — for names like "بنك أبوظبي الأول")
            if i < len(words) - 2:
                trigram = f"{word} {words[i+1]} {words[i+2]}"
                result = _resolver.resolve_single(trigram)
                if result and result not in out:
                    out.append(result)
        
        # ── Step 3: Deduplicate preserving order ──────────────────
        seen = set()
        uniq = []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        
        # ── Step 4: Remove fragments that are parts of compound tickers ──
        # e.g., if we have both "COMI.CA" and "COMI", remove "COMI"
        compound_tickers = [t for t in uniq if '.' in t or '-' in t or t.startswith('^')]
        fragments_to_remove = set()
        for ct in compound_tickers:
            parts = re.split(r'[.\-]', ct)
            for part in parts:
                if part in uniq and part != ct:
                    fragments_to_remove.add(part)
        
        return [t for t in uniq if t not in fragments_to_remove]

    @staticmethod
    def has_arabic_financial_intent(text: str) -> bool:
        """Check if text contains Arabic financial/stock keywords."""
        if not text:
            return False
        return any(kw in text for kw in ARABIC_FINANCIAL_KEYWORDS)

    @staticmethod
    def is_soft_portfolio_request(text: str) -> bool:
        low = (text or "").lower()
        return any(w in low for w in PORTFOLIO_SOFT_WORDS)

    @staticmethod
    def detect_primary_intent(text: str, mem: dict = None) -> str | None:
        """
        Legacy rule-based detection from agent.py.
        Returns: 'portfolio_optimize', 'portfolio_report', etc.
        Now also detects Arabic stock queries.
        """
        low = (text or "").lower().strip()
        mem = mem or {}

        # Hard commands win
        if low.startswith("optimize"):
            return "portfolio_optimize"
        if low.startswith("report"):
            return "portfolio_report"
        if low.startswith("metrics"):
            return "portfolio_metrics"
        if low.startswith(("add ", "remove ", "cap ", "set ")):
            return "portfolio_edit"
        
        # ── Fixed Income / Sukuk / ISIN ──────────────────────────────────────
        # Check BEFORE generic investment keywords to avoid misrouting ISIN queries
        if ISIN_RE.search(text.upper()):
            return "fixed_income"
        if any(kw in low for kw in FIXED_INCOME_KEYWORDS):
            return "fixed_income"
        # Arabic Sukuk/bond keywords
        if any(kw in text for kw in ["صكوك", "صك", "سندات حكومية", "سندات شركات", "دخل ثابت"]):
            return "fixed_income"

        # IRR calculation
        if low.startswith("irr") or "internal rate" in low or "calculate irr" in low:
            return "irr_calc"
        
        # VaR calculation
        if "var " in low or "value at risk" in low or low.startswith("var") or "calculate var" in low:
            return "var_calc"

        # Trade Execution
        if low.startswith(("buy ", "sell ", "close ", "trade ")):
            return "trade_execution"

        # Export / Download
        export_keywords = ["export", "download", "save", "pdf", "word", "docx", "صدر", "تصدير", "ملف"]
        if any(k in low for k in export_keywords):
            return "report_export"

        # ── NEW: Arabic stock/analysis intent ──────────────────
        # Check if the message has Arabic financial keywords + resolvable tickers
        if IntentClassifier.has_arabic_financial_intent(text):
            tickers = IntentClassifier.extract_tickers(text)
            if tickers:
                # Arabic analysis keywords → treat as stock analysis
                analysis_words = ["تحليل", "حلل", "سعر", "أسعار", "اسعار"]
                if any(w in text for w in analysis_words):
                    return "investment_query"
                # Arabic portfolio keywords → optimize
                portfolio_words = ["محفظة", "تنويع", "تحسين", "استثمر"]
                if any(w in text for w in portfolio_words):
                    return "portfolio_optimize"
                # Default: investment query
                return "investment_query"
        
        # Check if message contains Arabic company names without explicit keywords
        # (e.g., user just types "أرامكو" or "الراجحي")
        if not any(c.isascii() and c.isalpha() for c in text.replace(" ", "")):
            # Message is mostly Arabic/non-ASCII
            tickers = IntentClassifier.extract_tickers(text)
            if tickers:
                return "investment_query"

        # CIO Mode Trigger Check (Keywords)
        if any(k in low for k in CIO_TRIGGER_KEYWORDS):
            return "investment_query"

        # Extract tickers
        tickers = IntentClassifier.extract_tickers(text)
        
        # Soft intent rules: only optimize if 2+ tickers AND intent keyword or purely tickers
        optimize_keywords = ["optimize", "build", "create", "allocation", "portfolio", "weights", "rebalance"]
        is_optimize_msg = any(k in low for k in optimize_keywords)
        
        if len(tickers) >= 2:
            if is_optimize_msg or len(text.split()) < 5:
                return "portfolio_optimize"

        return None

    @staticmethod
    def classify_intent_hybrid(text: str, mem: dict = None) -> str:
        """
        Hybrid classification: Rule-based first, then LLM for ambiguity.
        Returns: 'investment' or 'general'.
        """
        # Rule-based check
        rule_intent = IntentClassifier.detect_primary_intent(text, mem)
        
        # Define which rule-intents trigger CIO (Executive) Mode
        CIO_INTENTS = {
            "portfolio_optimize", "portfolio_report", "portfolio_metrics",
            "portfolio_edit", "irr_calc", "var_calc", "investment_query",
            "fixed_income",   # Sukuk / bond / ISIN analysis
        }
        
        if rule_intent in CIO_INTENTS:
            return "investment"
            
        # If it's a utility intent like 'report_export' or 'generic_action', return 'general'
        if rule_intent == "report_export":
            return "general"
        
        # ── NEW: Catch remaining Arabic financial context ─────
        if IntentClassifier.has_arabic_financial_intent(text):
            return "investment"
        if has_arabic_stock_context(text):
            return "investment"
        
        # Rule-based didn't catch it → treat as general conversation.
        return "general"