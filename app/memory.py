import sqlite3
from datetime import datetime


class Memory:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    ai_output TEXT,
                    created_at TEXT
                )
            """)

    def save(self, user_input: str, ai_output: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO memory (user_input, ai_output, created_at) VALUES (?, ?, ?)",
                (user_input, ai_output, datetime.utcnow().isoformat())
            )

    def recent(self, limit: int = 5):
        cursor = self.conn.execute(
            "SELECT user_input, ai_output FROM memory ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()
