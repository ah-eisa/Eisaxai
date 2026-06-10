"""
Tests for the SessionManager (core/session_manager.py).
"""
import pytest


class TestSessionManager:
    """Test SessionManager CRUD operations."""

    def test_create_session(self, test_session_manager):
        """Can create a new session."""
        sid = test_session_manager.get_or_create_session(
            user_id="user1", session_id="test-session-1", ip="127.0.0.1"
        )
        assert sid == "test-session-1"

    def test_auto_generate_session_id(self, test_session_manager):
        """Auto-generates session_id if none provided."""
        sid = test_session_manager.get_or_create_session(user_id="user1")
        assert sid.startswith("session_")

    def test_save_and_get_messages(self, test_session_manager):
        """Can save and retrieve chat messages."""
        sid = test_session_manager.get_or_create_session("user1", "msg-test")
        test_session_manager.save_message(sid, "user1", "user", "Hello")
        test_session_manager.save_message(sid, "user1", "assistant", "Hi there!")

        history = test_session_manager.get_chat_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_user_sessions(self, test_session_manager):
        """Can list all sessions for a user."""
        test_session_manager.get_or_create_session("user1", "s1")
        test_session_manager.get_or_create_session("user1", "s2")
        test_session_manager.get_or_create_session("user2", "s3")

        sessions = test_session_manager.get_user_sessions("user1")
        session_ids = {s["session_id"] for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids
        assert "s3" not in session_ids  # belongs to user2

    def test_delete_session(self, test_session_manager):
        """Deleting a session removes it and its messages."""
        sid = test_session_manager.get_or_create_session("user1", "del-test")
        test_session_manager.save_message(sid, "user1", "user", "test")
        test_session_manager.delete_session(sid)

        history = test_session_manager.get_chat_history(sid)
        assert len(history) == 0


class TestBlockingSystem:
    """Test user and IP blocking."""

    def test_block_unblock_user(self, test_session_manager):
        """Can block and unblock a user."""
        test_session_manager.get_or_create_session("user1", "block-test")
        assert not test_session_manager.is_user_blocked("user1")

        test_session_manager.set_user_blocked("user1", True)
        assert test_session_manager.is_user_blocked("user1")

        test_session_manager.set_user_blocked("user1", False)
        assert not test_session_manager.is_user_blocked("user1")

    def test_ip_blocking(self, test_session_manager):
        """Can block and unblock IPs."""
        assert not test_session_manager.is_ip_blocked("192.168.1.1")

        test_session_manager.block_ip("192.168.1.1", "spam")
        assert test_session_manager.is_ip_blocked("192.168.1.1")

        test_session_manager.unblock_ip("192.168.1.1")
        assert not test_session_manager.is_ip_blocked("192.168.1.1")

    def test_empty_ip_not_blocked(self, test_session_manager):
        """Empty/unknown IPs should never be considered blocked."""
        assert not test_session_manager.is_ip_blocked("")
        assert not test_session_manager.is_ip_blocked("—")
        assert not test_session_manager.is_ip_blocked("unknown")


class TestRateLimiting:
    """Test rate limiting logic."""

    def test_no_limit_by_default(self, test_session_manager):
        """Users are not rate limited by default (limit=0 means unlimited)."""
        test_session_manager.get_or_create_session("user1", "rate-test")
        assert not test_session_manager.is_user_rate_limited("user1")

    def test_rate_limit_enforced(self, test_session_manager):
        """User is rate limited when daily count >= limit."""
        test_session_manager.get_or_create_session("user1", "rate-test")
        test_session_manager.set_user_profile("user1", daily_limit=2)

        # Send 2 messages
        test_session_manager.save_message("rate-test", "user1", "user", "msg1")
        test_session_manager.save_message("rate-test", "user1", "user", "msg2")

        assert test_session_manager.is_user_rate_limited("user1")


class TestAdminFeatures:
    """Test admin-related functionality."""

    def test_admin_messages(self, test_session_manager):
        """Can queue and retrieve admin messages."""
        test_session_manager.get_or_create_session("user1", "admin-test")
        test_session_manager.queue_admin_message("user1", "Welcome!")

        pending = test_session_manager.get_pending_admin_messages("user1")
        assert len(pending) == 1
        assert pending[0]["content"] == "Welcome!"

        test_session_manager.mark_admin_messages_delivered("user1")
        pending = test_session_manager.get_pending_admin_messages("user1")
        assert len(pending) == 0

    def test_admin_stats(self, test_session_manager):
        """Admin stats returns correct aggregates."""
        test_session_manager.get_or_create_session("user1", "stats-test")
        test_session_manager.save_message("stats-test", "user1", "user", "hi")

        stats = test_session_manager.get_admin_stats()
        assert stats["total_users"] >= 1
        assert stats["total_messages"] >= 1

    def test_audit_log(self, test_session_manager):
        """Admin actions are logged."""
        test_session_manager.log_admin_action("test_action", "user1", "details")
        log = test_session_manager.get_audit_log(limit=10)
        assert len(log) >= 1
        assert log[0]["action"] == "test_action"

    def test_user_profile_tier(self, test_session_manager):
        """Can set and get user tier."""
        test_session_manager.set_user_profile("user1", tier="vip")
        profile = test_session_manager.get_user_profile("user1")
        assert profile["tier"] == "vip"

    def test_session_state(self, test_session_manager):
        """Can save and load session state JSON."""
        test_session_manager.get_or_create_session("user1", "state-test")
        test_session_manager.save_session_state("state-test", {"mode": "admin", "step": 3})

        state = test_session_manager.get_session_state("state-test")
        assert state["mode"] == "admin"
        assert state["step"] == 3

    def test_broadcast(self, test_session_manager):
        """Broadcast sends to all users."""
        test_session_manager.get_or_create_session("user1", "bc1")
        test_session_manager.get_or_create_session("user2", "bc2")

        count = test_session_manager.broadcast_admin_message("System update!")
        assert count == 2
