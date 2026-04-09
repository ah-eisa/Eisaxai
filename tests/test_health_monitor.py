import importlib
from types import SimpleNamespace

import scripts.health_monitor as health_monitor


def reload_health_monitor():
    module = importlib.reload(health_monitor)
    module.gunicorn_restart_samples.clear()
    return module


def test_check_health_ignores_optional_gemini(monkeypatch):
    module = reload_health_monitor()

    class FakeResponse:
        def __init__(self):
            self.elapsed = SimpleNamespace(total_seconds=lambda: 0.123)

        def json(self):
            return {
                "services": {
                    "database": {"status": "ok"},
                    "deepseek": {"status": "ok"},
                    "kimi": {"status": "ok"},
                    "gemini": {"status": "down"},
                }
            }

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: FakeResponse())

    status, details = module.check_health()

    assert status == "ok"
    assert details["failed"] == []
    assert details["optional_failed"] == ["Gemini (optional) ⚠️"]


def test_restart_counter_tracks_last_hour_delta():
    module = reload_health_monitor()

    assert module.get_gunicorn_restart_count_last_hour(10, now=0) == 0
    assert module.get_gunicorn_restart_count_last_hour(25, now=1800) == 15
    assert module.get_gunicorn_restart_count_last_hour(70, now=3500) == 60
    assert module.get_gunicorn_restart_count_last_hour(90, now=7201) == 20


def test_zombie_loop_threshold_ignores_normal_gunicorn_worker_count():
    module = reload_health_monitor()

    assert not module.is_zombie_loop(6, 0)
    assert not module.is_zombie_loop(7, 50)
    assert module.is_zombie_loop(8, 0)
    assert module.is_zombie_loop(6, 51)


def test_format_diag_only_warns_on_new_threshold():
    module = reload_health_monitor()

    normal = module.format_diag(
        {
            "port_8000_procs": 6,
            "gunicorn_restarts": 12,
            "gunicorn_restarts_last_hour": 0,
            "memory": "1MB / 2MB (50%)",
            "disk": "1G used / 2G total (50%)",
        }
    )
    zombie = module.format_diag(
        {
            "port_8000_procs": 8,
            "gunicorn_restarts": 12,
            "gunicorn_restarts_last_hour": 0,
            "memory": "1MB / 2MB (50%)",
            "disk": "1G used / 2G total (50%)",
        }
    )

    assert "zombie loop!" not in normal
    assert "zombie loop!" in zombie
