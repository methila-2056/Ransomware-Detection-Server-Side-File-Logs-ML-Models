"""
Ransomware Detection - Database Operations

SQLite-based storage for:
- File operation logs
- Detection alerts
- Attack history
- System statistics
"""

import logging
import sqlite3
from typing import List, Dict
from contextlib import contextmanager

import config

log = logging.getLogger("ransomware.db")


class Database:
    """SQLite database manager for the ransomware detection system."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self._init_database()

    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    nc INTEGER NOT NULL,
                    nr INTEGER NOT NULL,
                    nu INTEGER NOT NULL,
                    att INTEGER NOT NULL,
                    user TEXT,
                    family TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    xgb_prediction INTEGER,
                    xgb_probability REAL,
                    xgb_confidence REAL,
                    all_models_prediction TEXT,
                    actual_label INTEGER,
                    user TEXT,
                    family TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attack_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attack_id INTEGER,
                    family TEXT,
                    start_second INTEGER,
                    end_second INTEGER,
                    duration INTEGER,
                    detected_by TEXT,
                    detection_delay_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_ticks INTEGER DEFAULT 0,
                    total_attacks INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    total_false_positives INTEGER DEFAULT 0,
                    total_false_negatives INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("SELECT COUNT(*) FROM system_stats")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO system_stats (total_ticks, total_attacks) VALUES (0, 0)")

            conn.commit()
            log.debug("Database initialized: %s", self.db_path)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_file_operation(self, tick: Dict):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO file_operations (timestamp, nc, nr, nu, att, user, family)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tick["timestamp"], tick["nc"], tick["nr"], tick["nu"],
                 tick["att"], tick.get("user", ""), tick.get("family", "")),
            )
            conn.commit()

    def log_detection_alert(self, tick: Dict, predictions: Dict):
        xgb = predictions.get("xgb", {})
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO detection_alerts
                   (timestamp, xgb_prediction, xgb_probability, xgb_confidence,
                    all_models_prediction, actual_label, user, family)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tick["timestamp"],
                    xgb.get("prediction", 0),
                    xgb.get("probability", 0),
                    xgb.get("confidence", 0),
                    str(predictions),
                    tick.get("att", 0),
                    tick.get("user", ""),
                    tick.get("family", ""),
                ),
            )
            conn.commit()

    def log_attack_session(self, attack_info: Dict):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO attack_sessions
                   (attack_id, family, start_second, end_second, duration,
                    detected_by, detection_delay_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    attack_info.get("attack_id"),
                    attack_info.get("family"),
                    attack_info.get("start_second"),
                    attack_info.get("end_second"),
                    attack_info.get("duration"),
                    attack_info.get("detected_by", ""),
                    attack_info.get("detection_delay", 0),
                ),
            )
            conn.commit()

    def update_stats(self, **kwargs):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            values = []
            for key, value in kwargs.items():
                if key in ("total_ticks", "total_attacks", "total_detections",
                           "total_false_positives", "total_false_negatives"):
                    updates.append(f"{key} = {key} + ?")
                    values.append(value)

            if updates:
                updates.append("last_updated = CURRENT_TIMESTAMP")
                query = f"UPDATE system_stats SET {', '.join(updates)} WHERE id = 1"
                cursor.execute(query, values)
                conn.commit()

    def get_stats(self) -> Dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_stats WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {
                "total_ticks": 0,
                "total_attacks": 0,
                "total_detections": 0,
                "total_false_positives": 0,
                "total_false_negatives": 0,
            }

    def get_recent_operations(self, limit: int = 60) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, nc, nr, nu, att, user, family
                   FROM file_operations
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, xgb_probability, xgb_confidence,
                          actual_label, user, family
                   FROM detection_alerts
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_attack_sessions(self, limit: int = 10) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM attack_sessions
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def clear_old_data(self, keep_last_n: int = 1000):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """DELETE FROM file_operations
                   WHERE id NOT IN (
                       SELECT id FROM file_operations
                       ORDER BY id DESC LIMIT ?
                   )""",
                (keep_last_n,),
            )
            cursor.execute(
                """DELETE FROM detection_alerts
                   WHERE id NOT IN (
                       SELECT id FROM detection_alerts
                       ORDER BY id DESC LIMIT ?
                   )""",
                (keep_last_n // 5,),
            )
            conn.commit()
            log.debug("Cleaned old data, kept last %d operations", keep_last_n)


def get_database(db_path: str = None) -> Database:
    return Database(db_path)
