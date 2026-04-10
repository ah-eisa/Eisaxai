import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import stripe

from core.db import db

logger = logging.getLogger(__name__)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
PRICE_IDS = {
    "pro": os.getenv("STRIPE_PRICE_PRO", ""),
    "vip": os.getenv("STRIPE_PRICE_VIP", ""),
}


class StripeBilling:
    def __init__(self) -> None:
        stripe.api_key = STRIPE_SECRET_KEY
        self.base_url = (
            os.getenv("APP_BASE_URL")
            or os.getenv("PUBLIC_BASE_URL")
            or os.getenv("SITE_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self.success_url = os.getenv(
            "STRIPE_SUCCESS_URL",
            f"{self.base_url}/static/pricing.html?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        )
        self.cancel_url = os.getenv(
            "STRIPE_CANCEL_URL",
            f"{self.base_url}/static/pricing.html?checkout=cancelled",
        )
        self.portal_return_url = os.getenv(
            "STRIPE_PORTAL_RETURN_URL",
            f"{self.base_url}/static/pricing.html?portal=returned",
        )
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with db.get_cursor() as (conn, c):
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS stripe_customers (
                    user_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    subscription_id TEXT,
                    tier TEXT,
                    updated_at TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY
                )
                """
            )
            existing = [row[1] for row in c.execute("PRAGMA table_info(user_profiles)").fetchall()]
            if "tier" not in existing:
                c.execute('ALTER TABLE user_profiles ADD COLUMN tier TEXT DEFAULT "basic"')

    def _require_secret_key(self) -> None:
        if not STRIPE_SECRET_KEY:
            raise RuntimeError("STRIPE_SECRET_KEY is not configured")

    def _set_user_tier(self, user_id: str, tier: str) -> None:
        with db.get_cursor() as (conn, c):
            c.execute("INSERT OR IGNORE INTO user_profiles (user_id) VALUES (?)", (user_id,))
            c.execute("UPDATE user_profiles SET tier = ? WHERE user_id = ?", (tier, user_id))

    def _find_customer_record(
        self,
        *,
        customer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
    ) -> Optional[dict]:
        self._ensure_tables()
        if not customer_id and not subscription_id:
            return None

        clauses = []
        params = []
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if subscription_id:
            clauses.append("subscription_id = ?")
            params.append(subscription_id)

        query = (
            "SELECT user_id, customer_id, subscription_id, tier "
            f"FROM stripe_customers WHERE {' OR '.join(clauses)} LIMIT 1"
        )
        with db.get_cursor() as (conn, c):
            row = c.execute(query, tuple(params)).fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "customer_id": row[1],
            "subscription_id": row[2],
            "tier": row[3],
        }

    def create_checkout_session(self, user_id: str, email: str, tier: str) -> str:
        self._require_secret_key()
        tier = (tier or "").strip().lower()
        price_id = PRICE_IDS.get(tier, "")
        if tier not in PRICE_IDS or not price_id:
            raise ValueError(f"Unsupported or unconfigured billing tier: {tier}")

        customer_id = self.get_customer_id(user_id)
        metadata = {"user_id": user_id, "tier": tier, "email": email}
        payload = {
            "mode": "subscription",
            "success_url": self.success_url,
            "cancel_url": self.cancel_url,
            "client_reference_id": user_id,
            "line_items": [{"price": price_id, "quantity": 1}],
            "metadata": metadata,
            "subscription_data": {"metadata": metadata},
        }
        if customer_id:
            payload["customer"] = customer_id
        else:
            payload["customer_email"] = email

        session = stripe.checkout.Session.create(**payload)
        checkout_url = getattr(session, "url", "") or ""
        if not checkout_url:
            raise RuntimeError("Stripe did not return a checkout URL")
        return checkout_url

    def create_portal_session(self, customer_id: str) -> str:
        self._require_secret_key()
        if not customer_id:
            raise ValueError("customer_id is required")
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=self.portal_return_url,
        )
        portal_url = getattr(session, "url", "") or ""
        if not portal_url:
            raise RuntimeError("Stripe did not return a portal URL")
        return portal_url

    def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        self._ensure_tables()
        if STRIPE_WEBHOOK_SECRET and sig_header:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
            event_data = event.to_dict_recursive() if hasattr(event, "to_dict_recursive") else event
        else:
            event_data = json.loads(payload.decode("utf-8"))

        event_type = event_data.get("type", "")
        data = (event_data.get("data") or {}).get("object") or {}

        if event_type == "checkout.session.completed":
            metadata = data.get("metadata") or {}
            user_id = metadata.get("user_id") or data.get("client_reference_id") or ""
            tier = (metadata.get("tier") or "basic").lower()
            customer_id = data.get("customer") or ""
            subscription_id = data.get("subscription") or ""
            if user_id:
                self.save_customer(user_id, customer_id, subscription_id, tier)
                self._set_user_tier(user_id, tier)
            return {"event": event_type, "user_id": user_id, "tier": tier}

        if event_type == "customer.subscription.deleted":
            customer_id = data.get("customer") or ""
            subscription_id = data.get("id") or ""
            record = self._find_customer_record(customer_id=customer_id, subscription_id=subscription_id)
            user_id = record["user_id"] if record else ""
            if user_id:
                self.save_customer(user_id, customer_id or record.get("customer_id", ""), "", "basic")
                self._set_user_tier(user_id, "basic")
            return {"event": event_type, "user_id": user_id, "tier": "basic"}

        return {"event": event_type, "user_id": None, "tier": None}

    def get_customer_id(self, user_id: str) -> Optional[str]:
        self._ensure_tables()
        with db.get_cursor() as (conn, c):
            row = c.execute(
                "SELECT customer_id FROM stripe_customers WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return str(row[0])

    def save_customer(
        self,
        user_id: str,
        customer_id: Optional[str],
        subscription_id: Optional[str],
        tier: str,
    ) -> None:
        self._ensure_tables()
        updated_at = datetime.now(timezone.utc).isoformat()
        with db.get_cursor() as (conn, c):
            c.execute(
                """
                INSERT INTO stripe_customers (
                    user_id, customer_id, subscription_id, tier, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    subscription_id = excluded.subscription_id,
                    tier = excluded.tier,
                    updated_at = excluded.updated_at
                """,
                (user_id, customer_id or "", subscription_id or "", tier, updated_at),
            )
