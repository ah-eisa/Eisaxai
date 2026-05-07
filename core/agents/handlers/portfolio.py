# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import re
from core.intent_classifier import IntentClassifier
from core.broker import BrokerClient
logger = logging.getLogger(__name__)


class PortfolioMixin:
    def _handle_portfolio_show(self, sid: str = None, mem: dict = None, msg: str = None) -> dict:
        """جلب وعرض بيانات المحفظة الحقيقية من Alpaca"""
        broker = BrokerClient()
        if not broker.is_active():
            return {"type": "chat.reply", "reply": "❌ لا يمكن الاتصال بالوسيط. تأكد من إعداد المفاتيح في ملف .env"}
        
        acct = broker.get_account()
        pos = broker.get_positions()
        
        reply = "## 📊 ملخص المحفظة\n\n"
        reply += f"**حالة الحساب:** {acct.get('status', 'N/A').upper()}\n"
        reply += f"**إجمالي القيمة (Equity):** ${acct.get('equity', 0):,.2f}\n"
        reply += f"**القوة الشرائية:** ${acct.get('buying_power', 0):,.2f}\n\n"
        
        if pos:
            reply += "## 📈 الصفقات المفتوحة\n"
            for p in pos:
                reply += f"- **{p['symbol']}**: {p['qty']} سهم | القيمة: ${p['market_value']:,.2f} | الربح/الخسارة: {p['unrealized_plpc']*100:+.2f}%\n"
        else:
            reply += "*لا توجد صفقات مفتوحة حالياً.*"
            
        return {"type": "chat.reply", "reply": reply}

    def _handle_account_display(self) -> dict:
        """عرض ملخص الحساب والمحفظة — wrapper لـ _handle_portfolio_show"""
        return self._handle_portfolio_show()

    def _handle_portfolio_add(self, sid: str, mem: dict, msg: str) -> dict:
        """إضافة صفقة للمحفظة المحلية — يدعم الأسهم المحلية"""
        import re
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers:
            return {"type": "chat.reply", "reply": "يرجى تحديد السهم المراد إضافته. مثال: 'add 10 shares NVDA at $130' أو 'أضف 10 أسهم أرامكو'"}
        
        ticker = tickers[0].upper()
        
        # Try to resolve via local ticker resolver if not a known format
        if not any(ticker.endswith(s) for s in ['.SR', '.CA', '.AE', '.DU', '.KW', '.QA', '-USD']):
            from core.agents.finance import _ticker_resolver
            local = _ticker_resolver.resolve_single(ticker)
            if local:
                ticker = local

        # Parse quantity
        qty_match = re.search(r'(\d+\.?\d*)\s*(?:share|سهم|stock)', msg.lower())
        qty = float(qty_match.group(1)) if qty_match else 1.0
        
        # Parse price
        price_match = re.search(r'(?:at|@|بسعر|price)\s*\$?(\d+\.?\d*)', msg.lower())
        if price_match:
            price = float(price_match.group(1))
        else:
            # Try to get live price — fast_info is ~3x faster than .info for price-only lookup
            try:
                import yfinance as yf
                _fi = yf.Ticker(ticker).fast_info
                price = float(getattr(_fi, "last_price", None) or 0)
            except Exception as _e:
                price = 0
        
        try:
            self.portfolio_tracker.add_position(ticker, qty, price)
            price_str = self._format_local_price(price, ticker)
            total_str = self._format_local_price(qty * price, ticker)
            name = self._get_local_display_name(ticker)
            return {
                "type": "chat.reply",
                "reply": f"✅ تم إضافة **{qty:.0f} سهم {name} ({ticker})** بسعر **{price_str}** للمحفظة.\n\nالقيمة الإجمالية: **{total_str}**"
            }
        except Exception as e:
            return {"type": "error", "reply": f"فشل إضافة الصفقة: {e}"}

    def _handle_portfolio_remove(self, sid: str, mem: dict, msg: str) -> dict:
        """إزالة/بيع صفقة من المحفظة المحلية"""
        import re
        tickers = IntentClassifier.extract_tickers(msg)
        if not tickers:
            return {"type": "chat.reply", "reply": "يرجى تحديد السهم المراد بيعه. مثال: 'sell 5 shares AAPL' أو 'بيع 5 أسهم أرامكو'"}
        
        ticker = tickers[0].upper()
        
        # Resolve local tickers
        if not any(ticker.endswith(s) for s in ['.SR', '.CA', '.AE', '.DU', '.KW', '.QA', '-USD']):
            from core.agents.finance import _ticker_resolver
            local = _ticker_resolver.resolve_single(ticker)
            if local:
                ticker = local
        
        # Parse quantity
        qty_match = re.search(r'(\d+\.?\d*)\s*(?:share|سهم|stock)', msg.lower())
        qty = float(qty_match.group(1)) if qty_match else None
        
        try:
            self.portfolio_tracker.remove_position(ticker, qty)
            name = self._get_local_display_name(ticker)
            qty_str = f"{qty:.0f} سهم من " if qty else "كل أسهم "
            return {
                "type": "chat.reply",
                "reply": f"✅ تم بيع {qty_str}**{name} ({ticker})** من المحفظة."
            }
        except Exception as e:
            return {"type": "error", "reply": f"فشل إزالة الصفقة: {e}"}


