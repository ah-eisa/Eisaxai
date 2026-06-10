import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "investwise.db"):
        self.db_path = db_path
        self._create_tables()
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT UNIQUE NOT NULL, user_id TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'active', metadata TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, agent_name TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, metadata TEXT, FOREIGN KEY (session_id) REFERENCES sessions(session_id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS user_profile (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT UNIQUE NOT NULL, name TEXT, email TEXT, interests TEXT, risk_tolerance TEXT, investment_goals TEXT, preferences TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.commit()
        conn.close()
    def create_session(self, session_id: str, user_id: str, metadata: Optional[Dict] = None) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sessions (session_id, user_id) VALUES (?, ?)", (session_id, user_id))
            conn.commit()
            conn.close()
            return True
        except: return False
    def add_message(self, session_id: str, role: str, content: str, agent_name: Optional[str] = None) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO chat_history (session_id, role, content, agent_name) VALUES (?, ?, ?, ?)", (session_id, role, content, agent_name))
            conn.commit()
            conn.close()
            return True
        except: return False
    def get_chat_history(self, session_id: str) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role, content, agent_name FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
