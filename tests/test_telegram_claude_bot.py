import types

import pytest

import telegram_claude_bot as bot


class DummyProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self._stdout = stdout.encode("utf-8")
        self._stderr = stderr.encode("utf-8")

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        return None


@pytest.mark.asyncio
async def test_run_claude_retries_with_resume_on_in_use(monkeypatch):
    calls = []
    procs = [
        DummyProc(1, stderr="Error: Session ID old-sid is already in use."),
        DummyProc(0, stdout="OK"),
    ]

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return procs.pop(0)

    monkeypatch.setattr(bot.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bot, "_session_file_exists", lambda sid: False)

    result = await bot.run_claude("Hi", "old-sid", 123)

    assert result == "OK"
    assert calls[0][-2:] == ["--session-id", "old-sid"]
    assert calls[1][-2:] == ["--resume", "old-sid"]


@pytest.mark.asyncio
async def test_run_claude_rotates_session_when_existing_one_is_stuck(monkeypatch):
    calls = []
    stored = []
    procs = [
        DummyProc(1, stderr="Error: Session ID old-sid is already in use."),
        DummyProc(1, stderr="Error: Session ID old-sid is already in use."),
        DummyProc(0, stdout="Fresh session works"),
    ]

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return procs.pop(0)

    monkeypatch.setattr(bot.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bot, "_session_file_exists", lambda sid: False)
    monkeypatch.setattr(bot.uuid, "uuid4", lambda: "new-sid")
    monkeypatch.setattr(bot.session_store, "set", lambda chat_id, session_id: stored.append((chat_id, session_id)))

    result = await bot.run_claude("Hi", "old-sid", 456)

    assert result == "Fresh session works"
    assert stored == [(456, "new-sid")]
    assert calls[2][-2:] == ["--session-id", "new-sid"]


def test_session_store_persists_values(tmp_path):
    path = tmp_path / "claude_sessions.db"
    store_a = bot.SessionStore(path)
    store_a.set(10, "sid-1")

    store_b = bot.SessionStore(path)
    assert store_b.get_or_create(10) == "sid-1"
    assert store_b.clear(10) != "sid-1"
