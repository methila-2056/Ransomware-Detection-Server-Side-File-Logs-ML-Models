"""
Tests for the SQLite storage layer.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import Database


class TestDatabase:
    """Test database schema, writes, and queries."""

    @staticmethod
    def _tick(ts=1, nc=5, nr=3, nu=2, att=0):
        return {"timestamp": ts, "nc": nc, "nr": nr, "nu": nu,
                "att": att, "user": "Secretary", "family": None}

    def test_schema_created(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        assert db.get_stats()["total_ticks"] == 0

    def test_log_file_operation_and_ordering(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        for i in range(3):
            db.log_file_operation(self._tick(ts=i))
        ops = db.get_recent_operations()
        assert [o["timestamp"] for o in ops] == [0, 1, 2]
        assert ops[0]["nc"] == 5

    def test_log_detection_alert(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        preds = {"xgb": {"prediction": 1, "probability": 0.9, "confidence": 95.0}}
        db.log_detection_alert(self._tick(att=1), preds)
        alerts = db.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["actual_label"] == 1
        assert alerts[0]["xgb_probability"] == 0.9

    def test_update_stats_accumulates(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        db.update_stats(total_ticks=5, total_attacks=2)
        db.update_stats(total_ticks=3)
        stats = db.get_stats()
        assert stats["total_ticks"] == 8
        assert stats["total_attacks"] == 2

    def test_clear_old_data_keeps_latest(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        for i in range(30):
            db.log_file_operation(self._tick(ts=i))
        db.clear_old_data(keep_last_n=10)
        ops = db.get_recent_operations(limit=100)
        assert len(ops) == 10
        assert [o["timestamp"] for o in ops] == list(range(20, 30))

    def test_attack_session_roundtrip(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        db.log_attack_session({
            "attack_id": 1, "family": "Ryuk", "start_second": 10,
            "end_second": 25, "duration": 15,
            "detected_by": "XGBoost", "detection_delay": 1.5,
        })
        sessions = db.get_attack_sessions()
        assert len(sessions) == 1
        assert sessions[0]["family"] == "Ryuk"
        assert sessions[0]["duration"] == 15
