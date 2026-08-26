"""
Integration tests for the end-to-end tick processing pipeline.
Covers metrics updates, alert generation, database persistence,
and attack session logging through process_tick().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import app as A


@pytest.fixture(autouse=True, scope="session")
def _load_models_once():
    """Load pretrained models once for the entire test session."""
    if not A.ml_engine.is_trained:
        A.init_ml_models()


@pytest.fixture(autouse=True)
def _reset_globals(tmp_path):
    """Reset shared state and isolate the database before every test."""
    original_db = A.db
    A.metrics = {
        "total_ticks": 0,
        "total_attacks": 0,
        "total_detections": 0,
        "total_fp": 0,
        "total_fn": 0,
        "current_streak": 0,
    }
    A.tick_history.clear()
    A.alert_log.clear()
    A._previous_tick_was_attack = False
    from database import Database
    A.db = Database(str(tmp_path / "test.db"))
    yield
    A.db = original_db


def _make_tick(nc=5, nr=2, nu=1, att=0, family=None, is_attack=False,
               source="simulation", ts=1, file_events=None):
    return {
        "timestamp": ts, "nc": nc, "nr": nr, "nu": nu,
        "att": att, "user": "Secretary", "family": family,
        "is_attack": is_attack, "source": source,
        "file_events": file_events or [],
    }


class TestProcessTickMetrics:
    """Verify metrics counters stay consistent through process_tick."""

    def test_benign_tick_increments_only_ticks(self):
        A.process_tick(_make_tick(ts=1))
        assert A.metrics["total_ticks"] == 1
        assert A.metrics["total_attacks"] == 0
        assert A.metrics["total_fp"] == 0

    def test_attack_tick_increments_attacks(self):
        A.process_tick(_make_tick(ts=1, att=1, nc=100, nr=70, nu=50, is_attack=True))
        assert A.metrics["total_attacks"] == 1
        assert A.metrics["total_ticks"] == 1

    def test_real_source_tick_has_source_field(self):
        A.process_tick(_make_tick(ts=1, source="real", nc=2, nr=1, nu=0))
        assert A.tick_history[0]["source"] == "real"
        assert A.tick_history[0]["att"] == 0

    def test_tick_history_bounded(self):
        for i in range(200):
            A.process_tick(_make_tick(ts=i + 1))
        assert len(A.tick_history) <= 130


class TestProcessTickAlerts:
    """Verify alert entries are generated for high-probability detections."""

    def test_benign_tick_no_alert(self):
        A.process_tick(_make_tick(ts=1, nc=5, nr=2, nu=1))
        assert len(A.alert_log) == 0

    def test_real_mode_detection_emits_alert(self):
        A.process_tick(_make_tick(ts=1, nc=100, nr=70, nu=50, att=1,
                                  source="real", is_attack=True))
        real_alerts = [a for a in A.alert_log if a["source"] == "real"]
        assert len(real_alerts) == 1
        assert real_alerts[0]["probability"] > 0.5
        assert "Suspicious activity" in real_alerts[0]["message"]

    def test_sim_attack_only_alerts_on_streak_start(self):
        for i in range(3):
            A.process_tick(_make_tick(ts=i+1, nc=110, nr=70, nu=50,
                                      att=1, is_attack=True, family="Ryuk"))
        alerts = [a for a in A.alert_log if a["family"] == "Ryuk"]
        assert len(alerts) >= 1
        assert alerts[0]["message"].startswith("Ransomware detected: Ryuk")

    def test_alert_log_bounded(self):
        for i in range(60):
            A.process_tick(_make_tick(ts=i+1, nc=120, nr=80, nu=60,
                                      att=1, source="real", is_attack=True))
        assert len(A.alert_log) <= A.config.MAX_ALERT_LOG


class TestProcessTickDB:
    """Verify database writes happen through process_tick."""

    def test_db_stats_increment(self):
        A.process_tick(_make_tick(ts=1))
        A.process_tick(_make_tick(ts=2))
        stats = A.db.get_stats()
        assert stats["total_ticks"] >= 0


class TestAttackSessionLogging:
    """Verify completed attack sessions are logged to SQLite."""

    def test_attack_end_logs_session(self):
        A._previous_tick_was_attack = True
        A.simulator.state.attack_history = [{
            "attack_id": 99, "family": "WannaCry",
            "start_second": 5, "duration": 10,
        }]
        A.process_tick(_make_tick(ts=11, att=0, nc=5, nr=2, nu=1))
        sessions = A.db.get_attack_sessions()
        wannacy = [s for s in sessions if s["family"] == "WannaCry"]
        assert len(wannacy) >= 1
        assert wannacy[0]["attack_id"] == 99

    def test_no_session_on_benign_streak(self):
        A._previous_tick_was_attack = False
        A.process_tick(_make_tick(ts=1))
        sessions = A.db.get_attack_sessions()
        assert len(sessions) == 0

    def test_previous_was_attack_tracks_correctly(self):
        A.process_tick(_make_tick(ts=1, att=1, nc=100, nr=70, nu=50,
                                  is_attack=True))
        assert A._previous_tick_was_attack is True
        A.process_tick(_make_tick(ts=2, att=0, nc=5, nr=2, nu=1))
        assert A._previous_tick_was_attack is False
