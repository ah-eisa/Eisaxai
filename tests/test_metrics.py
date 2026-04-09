import json
import logging

import pytest

import core.metrics as metrics


@pytest.fixture
def isolated_metrics(monkeypatch):
    tracker = metrics.PerformanceTracker()
    monkeypatch.setattr(metrics, "performance_tracker", tracker)

    logger = logging.getLogger(metrics._STRUCTURED_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    monkeypatch.setattr(metrics, "_STRUCTURED_LOGGER", None)

    yield tracker

    logger = logging.getLogger(metrics._STRUCTURED_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_performance_tracker_aggregates_metrics():
    tracker = metrics.PerformanceTracker()

    tracker.track_latency("core.op", 100.0)
    tracker.track_latency("core.op", 300.0)
    tracker.track_cache_hit("memory")
    tracker.track_cache_miss("memory")
    tracker.track_llm_call("openai", True, 120.0)
    tracker.track_llm_call("openai", False, 240.0)
    tracker.track_request("/chat", 200, 50.0)
    tracker.track_request("/chat", 500, 150.0)

    summary = tracker.get_summary()

    assert summary["latencies"]["core.op"]["count"] == 2
    assert summary["latencies"]["core.op"]["avg_ms"] == 200.0
    assert summary["cache_stats"]["memory"]["hits"] == 1
    assert summary["cache_stats"]["memory"]["misses"] == 1
    assert summary["cache_stats"]["memory"]["hit_rate"] == 0.5
    assert summary["llm_calls"]["openai"]["count"] == 2
    assert summary["llm_calls"]["openai"]["failures"] == 1
    assert summary["request_counts"]["/chat"] == 2
    assert summary["requests"]["/chat"]["status_codes"]["500"] == 1
    assert summary["error_counts"]["llm:openai"] == 1
    assert summary["error_counts"]["request:/chat:500"] == 1


def test_log_structured_outputs_json(capsys, isolated_metrics):
    metrics.log_structured("cache_event", level="memory", hit=True)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines

    payload = json.loads(lines[-1])
    assert payload["event"] == "cache_event"
    assert payload["fields"]["level"] == "memory"
    assert payload["fields"]["hit"] is True
    assert "timestamp" in payload


def test_track_performance_records_request_and_slow_warning(monkeypatch, capsys, isolated_metrics):
    perf_counter_values = iter([10.0, 12.5])
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(perf_counter_values))

    @metrics.track_performance
    def handle_request():
        return {
            "type": "error",
            "data": {"error": {"http_status_code": 503}},
        }

    response = handle_request()
    assert response["type"] == "error"

    operation = f"{handle_request.__module__}.{handle_request.__qualname__}"
    summary = isolated_metrics.get_summary()

    assert summary["request_counts"][operation] == 1
    assert summary["requests"][operation]["status_codes"]["503"] == 1
    assert summary["error_counts"][f"request:{operation}:503"] == 1

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    events = [payload["event"] for payload in payloads]
    assert "performance" in events
    assert "slow_call" in events


@pytest.mark.asyncio
async def test_track_performance_supports_async_functions(monkeypatch, isolated_metrics):
    perf_counter_values = iter([20.0, 20.25])
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(perf_counter_values))

    @metrics.track_performance
    async def process_message():
        return {"reply": "ok"}

    result = await process_message()
    assert result["reply"] == "ok"

    operation = f"{process_message.__module__}.{process_message.__qualname__}"
    summary = isolated_metrics.get_summary()

    assert summary["latencies"][operation]["avg_ms"] == 250.0
    assert summary["request_counts"][operation] == 1
    assert summary["requests"][operation]["status_codes"]["200"] == 1


def test_get_metrics_summary_returns_endpoint_payload(isolated_metrics):
    isolated_metrics.track_latency("core.worker", 40.0)
    isolated_metrics.track_request("/health", 200, 15.0)
    isolated_metrics.track_cache_hit("l1")
    isolated_metrics.track_cache_miss("l1")
    isolated_metrics.track_error("exception:core.worker:RuntimeError")

    payload = metrics.get_metrics_summary()

    assert set(payload) == {"request_counts", "avg_latencies", "cache_stats", "error_counts"}
    assert payload["request_counts"]["/health"] == 1
    assert payload["avg_latencies"]["requests"]["/health"] == 15.0
    assert payload["cache_stats"]["l1"]["hit_rate"] == 0.5
    assert payload["error_counts"]["exception:core.worker:RuntimeError"] == 1
