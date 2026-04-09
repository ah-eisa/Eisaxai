from __future__ import annotations

import inspect
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")

SLOW_CALL_THRESHOLD_MS = 2000.0
_STRUCTURED_LOGGER_NAME = "investwise.structured"
_STRUCTURED_LOGGER: logging.Logger | None = None


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _get_structured_logger() -> logging.Logger:
    global _STRUCTURED_LOGGER
    if _STRUCTURED_LOGGER is not None:
        return _STRUCTURED_LOGGER

    logger = logging.getLogger(_STRUCTURED_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    _STRUCTURED_LOGGER = logger
    return logger


def _emit_structured_log(log_level: int, event: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "fields": fields,
    }
    _get_structured_logger().log(
        log_level,
        json.dumps(payload, default=_json_default, ensure_ascii=True, sort_keys=True),
    )
    return payload


def log_structured(event: str, **fields: Any) -> dict[str, Any]:
    return _emit_structured_log(logging.INFO, event, **fields)


class PerformanceTracker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._latencies: dict[str, dict[str, float | int | None]] = defaultdict(
                lambda: {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": None,
                    "max_ms": 0.0,
                }
            )
            self._cache_hits: dict[str, int] = defaultdict(int)
            self._cache_misses: dict[str, int] = defaultdict(int)
            self._llm_calls: dict[str, dict[str, float | int]] = defaultdict(
                lambda: {
                    "count": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_latency_ms": 0.0,
                }
            )
            self._requests: dict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "count": 0,
                    "total_latency_ms": 0.0,
                    "status_codes": defaultdict(int),
                }
            )
            self._error_counts: dict[str, int] = defaultdict(int)

    def _record_latency(self, store: dict[str, dict[str, Any]], key: str, duration_ms: float) -> None:
        bucket = store[key]
        bucket["count"] += 1
        bucket["total_ms"] += duration_ms
        bucket["max_ms"] = max(float(bucket["max_ms"]), duration_ms)

        current_min = bucket["min_ms"]
        if current_min is None:
            bucket["min_ms"] = duration_ms
        else:
            bucket["min_ms"] = min(float(current_min), duration_ms)

    def track_latency(self, operation: str, duration_ms: float) -> None:
        with self._lock:
            self._record_latency(self._latencies, operation, duration_ms)

    def track_cache_hit(self, level: str) -> None:
        with self._lock:
            self._cache_hits[level] += 1

    def track_cache_miss(self, level: str) -> None:
        with self._lock:
            self._cache_misses[level] += 1

    def track_llm_call(self, provider: str, success: bool, latency_ms: float) -> None:
        with self._lock:
            bucket = self._llm_calls[provider]
            bucket["count"] += 1
            bucket["total_latency_ms"] += latency_ms
            if success:
                bucket["successes"] += 1
            else:
                bucket["failures"] += 1
                self._error_counts[f"llm:{provider}"] += 1

    def track_request(self, path: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            bucket = self._requests[path]
            bucket["count"] += 1
            bucket["total_latency_ms"] += duration_ms
            bucket["status_codes"][str(status_code)] += 1
            if status_code >= 400:
                self._error_counts[f"request:{path}:{status_code}"] += 1

    def track_error(self, name: str) -> None:
        with self._lock:
            self._error_counts[name] += 1

    @staticmethod
    def _avg(total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return round(total / count, 2)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            latencies = {
                operation: {
                    "count": int(stats["count"]),
                    "total_ms": round(float(stats["total_ms"]), 2),
                    "avg_ms": self._avg(float(stats["total_ms"]), int(stats["count"])),
                    "min_ms": round(float(stats["min_ms"]), 2) if stats["min_ms"] is not None else None,
                    "max_ms": round(float(stats["max_ms"]), 2),
                }
                for operation, stats in self._latencies.items()
            }

            all_cache_levels = sorted(set(self._cache_hits) | set(self._cache_misses))
            cache_stats = {}
            for level in all_cache_levels:
                hits = self._cache_hits[level]
                misses = self._cache_misses[level]
                total = hits + misses
                cache_stats[level] = {
                    "hits": hits,
                    "misses": misses,
                    "total": total,
                    "hit_rate": round(hits / total, 4) if total else 0.0,
                }

            llm_calls = {
                provider: {
                    "count": int(stats["count"]),
                    "successes": int(stats["successes"]),
                    "failures": int(stats["failures"]),
                    "success_rate": round(
                        int(stats["successes"]) / int(stats["count"]),
                        4,
                    )
                    if int(stats["count"])
                    else 0.0,
                    "avg_latency_ms": self._avg(
                        float(stats["total_latency_ms"]),
                        int(stats["count"]),
                    ),
                }
                for provider, stats in self._llm_calls.items()
            }

            requests = {
                path: {
                    "count": int(stats["count"]),
                    "avg_latency_ms": self._avg(
                        float(stats["total_latency_ms"]),
                        int(stats["count"]),
                    ),
                    "status_codes": dict(stats["status_codes"]),
                }
                for path, stats in self._requests.items()
            }

            request_counts = {
                path: stats["count"]
                for path, stats in requests.items()
            }
            request_avg_latencies = {
                path: stats["avg_latency_ms"]
                for path, stats in requests.items()
            }
            operation_avg_latencies = {
                operation: stats["avg_ms"]
                for operation, stats in latencies.items()
            }
            llm_avg_latencies = {
                provider: stats["avg_latency_ms"]
                for provider, stats in llm_calls.items()
            }

            return {
                "totals": {
                    "requests": sum(request_counts.values()),
                    "operations": sum(stats["count"] for stats in latencies.values()),
                    "llm_calls": sum(stats["count"] for stats in llm_calls.values()),
                    "cache_hits": sum(self._cache_hits.values()),
                    "cache_misses": sum(self._cache_misses.values()),
                },
                "latencies": latencies,
                "cache_stats": cache_stats,
                "llm_calls": llm_calls,
                "requests": requests,
                "request_counts": request_counts,
                "avg_latencies": {
                    "operations": operation_avg_latencies,
                    "requests": request_avg_latencies,
                    "llm": llm_avg_latencies,
                },
                "error_counts": dict(self._error_counts),
            }


performance_tracker = PerformanceTracker()


def get_metrics_summary() -> dict[str, Any]:
    summary = performance_tracker.get_summary()
    return {
        "request_counts": summary["request_counts"],
        "avg_latencies": summary["avg_latencies"],
        "cache_stats": summary["cache_stats"],
        "error_counts": summary["error_counts"],
    }


def _operation_name(func: Any) -> str:
    return f"{func.__module__}.{func.__qualname__}"


def _should_track_request(func: Any) -> bool:
    return func.__name__ in {"handle_request", "process_message"}


def _status_code_from_result(result: Any) -> int:
    if not isinstance(result, dict):
        return 200

    error = result.get("data", {}).get("error") if isinstance(result.get("data"), dict) else None
    if isinstance(error, dict):
        code = error.get("http_status_code")
        if isinstance(code, int):
            return code

    if result.get("type") == "error":
        return 500

    if result.get("agent_name") == "ErrorHandler" or result.get("model") == "error":
        return 500

    return 200


def _record_call(
    operation: str,
    duration_ms: float,
    func: Any,
    result: Any = None,
    error: Exception | None = None,
) -> None:
    rounded_duration = round(duration_ms, 2)
    performance_tracker.track_latency(operation, duration_ms)

    status_code: int | None = None
    if _should_track_request(func):
        status_code = 500 if error is not None else _status_code_from_result(result)
        performance_tracker.track_request(operation, status_code, duration_ms)

    if error is not None:
        performance_tracker.track_error(f"exception:{operation}:{type(error).__name__}")
        _emit_structured_log(
            logging.ERROR,
            "performance_error",
            operation=operation,
            duration_ms=rounded_duration,
            status_code=status_code,
            error_type=type(error).__name__,
            error=str(error),
        )
    else:
        log_structured(
            "performance",
            operation=operation,
            duration_ms=rounded_duration,
            status_code=status_code,
            success=True,
        )

    if duration_ms > SLOW_CALL_THRESHOLD_MS:
        _emit_structured_log(
            logging.WARNING,
            "slow_call",
            operation=operation,
            duration_ms=rounded_duration,
            status_code=status_code,
        )


def track_performance(func: Any):
    operation = _operation_name(func)

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                _record_call(operation, (time.perf_counter() - start) * 1000, func, error=exc)
                raise

            _record_call(operation, (time.perf_counter() - start) * 1000, func, result=result)
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            _record_call(operation, (time.perf_counter() - start) * 1000, func, error=exc)
            raise

        _record_call(operation, (time.perf_counter() - start) * 1000, func, result=result)
        return result

    return sync_wrapper


def perf_metrics(daily_returns: "pd.Series", rf: float = 0.0) -> dict[str, float]:
    import pandas as pd
    import empyrical as ep
    from scipy.stats import kurtosis, skew

    if not isinstance(daily_returns, pd.Series):
        daily_returns = pd.Series(daily_returns)

    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        raise ValueError("daily_returns is empty")

    rf_daily = (1 + rf) ** (1 / 252) - 1

    return {
        "cagr": float(ep.annual_return(daily_returns)),
        "vol": float(ep.annual_volatility(daily_returns)),
        "sharpe": float(ep.sharpe_ratio(daily_returns, risk_free=rf_daily)),
        "sortino": float(ep.sortino_ratio(daily_returns, required_return=rf_daily)),
        "max_drawdown": float(ep.max_drawdown(daily_returns)),
        "calmar": float(ep.calmar_ratio(daily_returns)),
        "skew": float(skew(daily_returns, bias=False)),
        "kurtosis": float(kurtosis(daily_returns, fisher=True, bias=False)),
    }
