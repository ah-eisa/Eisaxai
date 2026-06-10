"""
Gunicorn configuration for EisaX staging worker.

Addresses the intermittent futex_do_wait worker-boot deadlock:
When Python modules create threading.Lock / threading.RLock at module-level,
and the master process holds one of those locks at fork() time (e.g. during
module import or cache flush), the child worker inherits the OS-level futex in
locked state. The thread that held the lock is NOT present in the child
(only the calling thread survives fork). The child then deadlocks trying to
re-acquire the lock.

The post_fork hook below resets every known module-level lock in the project
before the uvicorn event loop starts.
"""

import os
import sys
import random
import logging

# ── Basic settings ────────────────────────────────────────────────────────────
bind = "127.0.0.1:8001"
workers = 2                   # production-shape test (was 1)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300
graceful_timeout = 30
keepalive = 5
worker_tmp_dir = "/dev/shm"   # avoids inotify/filesystem heartbeat issues on ARM64

# ── Proxy trust ───────────────────────────────────────────────────────────────
# Trust X-Forwarded-* ONLY from the local nginx peer. With this set, uvicorn
# rewrites request.client.host to the real client IP, so app-layer logic that
# checks for a loopback peer (e.g. _resolve_staging_access) cannot be fooled by
# external traffic. Without it, every proxied request would look like 127.0.0.1.
forwarded_allow_ips = "127.0.0.1"

loglevel = "info"
errorlog = "/home/ubuntu/investwise/logs/gunicorn_staging_test_error.log"
accesslog = "/home/ubuntu/investwise/logs/gunicorn_staging_test_access.log"

# ── Scheduler dedup lock ──────────────────────────────────────────────────────
# APScheduler (news engine) starts at module-import time inside api_bridge_v2.
# With workers >= 2, both children would start it → news collected ×N.
# post_fork uses O_EXCL on this lock file: first worker to fork wins and
# becomes the scheduler owner; later workers see the file exists and skip.
# api_bridge_v2 reads EISAX_SCHEDULER_OWNER (default "1") to decide.
SCHEDULER_LOCK = "/tmp/eisax-staging-scheduler.lock"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_lock(lock):
    """Force-release a threading.Lock or threading.RLock that may be stuck."""
    import threading
    try:
        if isinstance(lock, type(threading.Lock())):
            # Plain Lock: release if locked (ignore RuntimeError if not locked)
            try:
                lock.release()
            except RuntimeError:
                pass
        else:
            # RLock: drain the count
            while True:
                try:
                    lock.release()
                except RuntimeError:
                    break
    except Exception:
        pass


# ── Fork safety ───────────────────────────────────────────────────────────────

def when_ready(server):
    """Called once in master after it's bound but before any worker forks.
    Clear any stale scheduler lock from a previous crash so this run can
    elect a fresh owner."""
    try:
        if os.path.exists(SCHEDULER_LOCK):
            os.unlink(SCHEDULER_LOCK)
    except Exception:
        pass


def post_fork(server, worker):
    """
    Called in the child process immediately after fork(), before the uvicorn
    event loop starts. Resets all module-level threading primitives that could
    have been inherited in a locked state from the master.
    """
    # 0. Scheduler-owner election (must run BEFORE app import so api_bridge_v2
    #    can read EISAX_SCHEDULER_OWNER at module-load time).
    try:
        fd = os.open(SCHEDULER_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        os.environ["EISAX_SCHEDULER_OWNER"] = "1"
    except FileExistsError:
        # Another worker already claimed ownership in a prior fork
        os.environ["EISAX_SCHEDULER_OWNER"] = "0"
    except Exception:
        # Fail-safe: default to owner if filesystem misbehaves on first boot.
        os.environ["EISAX_SCHEDULER_OWNER"] = "1"

    # 1. Re-seed randomness (each worker gets fresh entropy)
    random.seed()

    # 2. Reset logging handlers (avoid sharing file-descriptor buffers)
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
    logging.root.handlers.clear()

    # 3. Reset every known module-level lock in the project.
    #    We work through sys.modules to avoid re-importing.

    # core.analysis_cache._lock  (RLock — wraps parquet cache r/w)
    if "core.analysis_cache" in sys.modules:
        try:
            _reset_lock(sys.modules["core.analysis_cache"]._lock)
        except Exception:
            pass

    # core.news_engine_client._cache_lock  (Lock — wraps news cache dict)
    if "core.news_engine_client" in sys.modules:
        try:
            _reset_lock(sys.modules["core.news_engine_client"]._cache_lock)
        except Exception:
            pass

    # core.ticker_index._INDEX_LOCK  (Lock — wraps ticker symbol index)
    if "core.ticker_index" in sys.modules:
        try:
            _reset_lock(sys.modules["core.ticker_index"]._INDEX_LOCK)
        except Exception:
            pass

    # core.utils — RateLimiter._lock and similar class-level locks
    if "core.utils" in sys.modules:
        mod = sys.modules["core.utils"]
        for attr in dir(mod):
            try:
                obj = getattr(mod, attr, None)
                if obj is not None and hasattr(obj, "_lock"):
                    _reset_lock(obj._lock)
            except Exception:
                pass

    # 4. Flush import lock (Python's internal _imp lock can also deadlock).
    #    We do NOT try to manipulate _imp directly — instead we call
    #    importlib.invalidate_caches() which safely re-initialises finder
    #    state without touching the lock.
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass


def worker_exit(server, worker):
    """Called when a worker exits — flush pending log records."""
    logging.shutdown()
