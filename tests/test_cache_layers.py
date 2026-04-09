import importlib

import pytest


@pytest.fixture
def analysis_cache(tmp_path):
    module = importlib.import_module("core.analysis_cache")
    module.configure(str(tmp_path / "analysis_cache_test.db"))
    module.reset(clear_persistent=True, clear_stats=True)
    yield module
    module.reset(clear_persistent=True, clear_stats=True)


def test_cache_uses_l1_l2_l3_in_order(analysis_cache, monkeypatch):
    current = {"ts": 1_700_000_000.0}
    monkeypatch.setattr(analysis_cache, "_now", lambda: current["ts"])

    key_session_1 = analysis_cache.make_key(
        "aapl",
        analysis_type="full",
        user_id="user-1",
        session_id="session-1",
    )
    analysis_cache.set(key_session_1, {"reply": "AAPL analysis", "model": "deepseek"}, level="all")

    l1_hit = analysis_cache.get(key_session_1)
    assert l1_hit["cache_level"] == "l1"

    key_session_2 = analysis_cache.make_key(
        "aapl",
        analysis_type="full",
        timestamp_bucket=key_session_1.timestamp_bucket,
        user_id="user-1",
        session_id="session-2",
    )
    l2_hit = analysis_cache.get(key_session_2)
    assert l2_hit["cache_level"] == "l2"

    key_other_user = analysis_cache.make_key(
        "aapl",
        analysis_type="full",
        timestamp_bucket=key_session_1.timestamp_bucket,
        user_id="user-2",
        session_id="session-9",
    )
    l3_hit = analysis_cache.get(key_other_user)
    assert l3_hit["cache_level"] == "l3"

    promoted_l1_hit = analysis_cache.get(key_other_user)
    assert promoted_l1_hit["cache_level"] == "l1"


def test_cache_falls_back_as_each_level_expires(analysis_cache, monkeypatch):
    current = {"ts": 2_000_000_000.0}
    monkeypatch.setattr(analysis_cache, "_now", lambda: current["ts"])

    key = analysis_cache.make_key(
        "msft",
        analysis_type="quick",
        user_id="user-1",
        session_id="session-1",
    )
    analysis_cache.set(key, {"reply": "MSFT quick take", "model": "deepseek"}, level="all")

    current["ts"] += analysis_cache.L1_TTL_SECONDS + 1
    assert analysis_cache.get(key)["cache_level"] == "l2"

    current["ts"] = 2_000_000_000.0 + analysis_cache.L2_TTL_SECONDS + 1
    assert analysis_cache.get(key)["cache_level"] == "l3"

    current["ts"] = 2_000_000_000.0 + analysis_cache.L3_TTL_SECONDS + 1
    assert analysis_cache.get(key) is None


def test_invalidate_can_clear_single_ticker_or_all(analysis_cache, monkeypatch):
    current = {"ts": 1_800_000_000.0}
    monkeypatch.setattr(analysis_cache, "_now", lambda: current["ts"])

    aapl_key = analysis_cache.make_key("aapl", user_id="user-1", session_id="session-1")
    msft_key = analysis_cache.make_key("msft", user_id="user-1", session_id="session-1")

    analysis_cache.set(aapl_key, {"reply": "AAPL", "model": "deepseek"}, level="all")
    analysis_cache.set(msft_key, {"reply": "MSFT", "model": "deepseek"}, level="all")

    analysis_cache.invalidate("AAPL")
    assert analysis_cache.get(aapl_key) is None
    assert analysis_cache.get(msft_key)["reply"] == "MSFT"

    analysis_cache.invalidate()
    assert analysis_cache.get(msft_key) is None


def test_stats_report_hit_and_miss_rates_per_level(analysis_cache, monkeypatch):
    current = {"ts": 1_900_000_000.0}
    monkeypatch.setattr(analysis_cache, "_now", lambda: current["ts"])

    key = analysis_cache.make_key("nvda", user_id="user-1", session_id="session-1")
    analysis_cache.set(key, {"reply": "NVDA", "model": "deepseek"}, level="all")

    assert analysis_cache.get(key)["cache_level"] == "l1"
    analysis_cache.invalidate("NVDA")
    assert analysis_cache.get(key) is None

    stats = analysis_cache.stats()
    assert stats["l1"]["requests"] == 2
    assert stats["l1"]["hits"] == 1
    assert stats["l1"]["misses"] == 1
    assert stats["l1"]["hit_rate"] == 0.5
    assert stats["l2"]["requests"] == 1
    assert stats["l2"]["misses"] == 1
    assert stats["l3"]["requests"] == 1
    assert stats["l3"]["misses"] == 1
