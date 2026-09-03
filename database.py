"""
Ransomware Detection - Database Operations

SQLite-based storage for:
- File operation logs
- Detection alerts
- Attack history
- System statistics
"""

import os
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager


DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "ransomware_detection.db")


class Database:
    """SQLite database manager for the ransomware detection system."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # File operations log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    timestamp_ns INTEGER,
                    nc INTEGER NOT NULL,
                    nw INTEGER NOT NULL,
                    nr INTEGER NOT NULL,
                    nm INTEGER NOT NULL,
                    nu INTEGER NOT NULL,
                    att INTEGER NOT NULL,
                    user TEXT,
                    family TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Detection alerts
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

            # Attack sessions
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

            # System statistics
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

            # Initialize stats if empty
            cursor.execute("SELECT COUNT(*) FROM system_stats")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO system_stats (total_ticks, total_attacks) VALUES (0, 0)")

            # Migration: add missing 5-operation columns to existing file_operations table
            self._migrate_file_operations(conn, cursor)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _migrate_file_operations(self, conn, cursor):
        """
        Migrate an existing file_operations table from the legacy 3-column
        (nc, nr=rename, nu) schema to the paper-aligned 5-column schema
        (nc, nw, nr=read, nm=rename, nu).
        """
        cursor.execute("PRAGMA table_info(file_operations)")
        cols = [row[1] for row in cursor.fetchall()]

        # Already fully migrated (nw and nm present means 5-column schema).
        if "nw" in cols and "nm" in cols:
            if "timestamp_ns" not in cols:
                cursor.execute("ALTER TABLE file_operations ADD COLUMN timestamp_ns INTEGER")
            return

        # New table created with full schema but missing helpers - ensure timestamp_ns.
        if "nw" in cols and "nm" not in cols:
            cursor.execute("ALTER TABLE file_operations ADD COLUMN nm INTEGER DEFAULT 0")
            if "timestamp_ns" not in cols:
                cursor.execute("ALTER TABLE file_operations ADD COLUMN timestamp_ns INTEGER")
            return

        # Legacy schema: nc, nr(=rename), nu. Rebuild to add nw (write),
        # nm (rename), and reinterpret nr as read. Preserve existing rows by
        # mapping legacy rename counts into nm and leaving read as 0.
        cursor.execute("ALTER TABLE file_operations RENAME TO file_operations_legacy")
        cursor.execute("""
            CREATE TABLE file_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                timestamp_ns INTEGER,
                nc INTEGER NOT NULL,
                nw INTEGER NOT NULL,
                nr INTEGER NOT NULL,
                nm INTEGER NOT NULL,
                nu INTEGER NOT NULL,
                att INTEGER NOT NULL,
                user TEXT,
                family TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO file_operations
                (id, timestamp, timestamp_ns, nc, nw, nr, nm, nu, att, user, family, created_at)
            SELECT id, timestamp, NULL, nc, 0, 0, nr, nu, att, user, family, created_at
            FROM file_operations_legacy
        """)
        cursor.execute("DROP TABLE file_operations_legacy")

    def log_file_operation(self, tick: Dict):
        """Log a file operation tick to the database (server-side log)."""
        timestamp_ns = tick.get("timestamp_ns", None)
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO file_operations
                   (timestamp, timestamp_ns, nc, nw, nr, nm, nu, att, user, family)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tick["timestamp"],
                    timestamp_ns,
                    tick["nc"], tick["nw"], tick["nr"], tick["nm"], tick["nu"],
                    tick["att"],
                    tick.get("user", ""),
                    tick.get("family", ""),
                ),
            )
            conn.commit()

    def log_detection_alert(self, tick: Dict, predictions: Dict):
        """Log a detection alert when attack is predicted."""
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
        """Log a completed attack session."""
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
        """Update system statistics."""
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
        """Get current system statistics."""
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
        """Get recent file operations for the timeline."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, timestamp_ns, nc, nw, nr, nm, nu, att, user, family
                   FROM file_operations
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_training_data(self, limit: int = 20000) -> List[Dict]:
        """
        Fetch logged file operations as training samples.

        Returns a list of dicts with the 5 operation features and the
        attack label, usable directly by ml_engine.prepare_training_data.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT nc, nw, nr, nm, nu, att
                   FROM file_operations
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent detection alerts."""
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
        """Get attack session history."""
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
        """Clear old data to prevent database bloat."""
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


def get_database(db_path: str = None) -> Database:
    """Factory function to get database instance."""
    return Database(db_path)
