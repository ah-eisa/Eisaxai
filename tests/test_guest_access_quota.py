import sqlite3

import api_bridge_v2 as api


def _create_guest_db(path):
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE guest_trial (
            username TEXT PRIMARY KEY,
            analyses_used INTEGER NOT NULL DEFAULT 0,
            portfolios_used INTEGER NOT NULL DEFAULT 0,
            max_analyses INTEGER NOT NULL DEFAULT 5,
            max_portfolios INTEGER NOT NULL DEFAULT 3,
            created_at TEXT NOT NULL,
            last_used TEXT
        )
        """
    )
    con.commit()
    con.close()


def _patch_sqlite_connect(monkeypatch, path):
    real_connect = sqlite3.connect

    def connect(_db, *args, **kwargs):
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)


def test_test_guest_counts_only_successful_analysis(tmp_path, monkeypatch):
    db_path = tmp_path / "guest.db"
    _create_guest_db(db_path)
    _patch_sqlite_connect(monkeypatch, db_path)

    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO guest_trial VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)",
        ("Test", 5, 0, 6, 0),
    )
    con.commit()
    con.close()

    allowed, message = api._guest_trial_check("Test", is_portfolio=False)
    assert allowed is True
    assert message == ""

    con = sqlite3.connect(db_path)
    assert con.execute("SELECT analyses_used FROM guest_trial WHERE username='Test'").fetchone()[0] == 5
    con.close()

    status = api._guest_trial_increment_success("Test", is_portfolio=False)
    assert status["analysis_limit"] == 6
    assert status["analyses_remaining"] == 0

    allowed, message = api._guest_trial_check("Test", is_portfolio=False)
    assert allowed is False
    assert message == api._GUEST_LIMIT_MESSAGE


def test_test_guest_portfolio_scope_is_blocked(tmp_path, monkeypatch):
    db_path = tmp_path / "guest.db"
    _create_guest_db(db_path)
    _patch_sqlite_connect(monkeypatch, db_path)

    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO guest_trial VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)",
        ("Test", 0, 0, 6, 0),
    )
    con.commit()
    con.close()

    allowed, message = api._guest_trial_check("Test", is_portfolio=True)
    assert allowed is False
    assert message == api._GUEST_LIMIT_MESSAGE


def test_existing_guest_quota_remains_independent(tmp_path, monkeypatch):
    db_path = tmp_path / "guest.db"
    _create_guest_db(db_path)
    _patch_sqlite_connect(monkeypatch, db_path)

    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO guest_trial VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)",
        ("alan.talib", 1, 0, 10, 3),
    )
    con.execute(
        "INSERT INTO guest_trial VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)",
        ("Test", 6, 0, 6, 0),
    )
    con.commit()
    con.close()

    allowed, _ = api._guest_trial_check("alan.talib", is_portfolio=False)
    assert allowed is True
    api._guest_trial_increment_success("alan.talib", is_portfolio=False)

    con = sqlite3.connect(db_path)
    assert con.execute("SELECT analyses_used FROM guest_trial WHERE username='alan.talib'").fetchone()[0] == 2
    assert con.execute("SELECT analyses_used FROM guest_trial WHERE username='Test'").fetchone()[0] == 6
    con.close()
