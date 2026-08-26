"""
Tests for the real folder monitor module.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from real_monitor import (
    FileCountHandler,
    RealFolderMonitor,
    resolve_known_folders,
)


class FakeEvent:
    """Minimal stand-in for watchdog file events (duck-typed)."""

    def __init__(self, src, dest=None, is_directory=False):
        self.src_path = src
        self.dest_path = dest
        self.is_directory = is_directory


class TestResolveKnownFolders:
    """Test Windows known-folder resolution (OneDrive-aware)."""

    def test_returns_common_folders(self):
        known = resolve_known_folders()
        assert "Desktop" in known
        assert "Downloads" in known

    def test_resolved_paths_exist(self):
        for name, path in resolve_known_folders().items():
            assert os.path.isdir(path), f"{name} -> {path} does not exist"

    def test_paths_under_user_profile(self):
        home = os.path.expanduser("~")
        for path in resolve_known_folders().values():
            assert path.lower().startswith(home.lower())


class TestFileCountHandler:
    """Test operation counting and per-tick event draining."""

    def test_counts_operations(self):
        h = FileCountHandler()
        h.on_created(FakeEvent("C:\\x\\a.txt"))
        h.on_moved(FakeEvent("C:\\x\\a.txt", dest="C:\\x\\b.txt"))
        h.on_deleted(FakeEvent("C:\\x\\b.txt"))
        h.on_modified(FakeEvent("C:\\x\\b.txt"))
        counts = h.get_and_reset_counts()
        assert counts["nc"] == 1
        assert counts["nr"] == 1
        assert counts["nu"] == 1
        assert counts["nm"] == 1

    def test_reset_after_read(self):
        h = FileCountHandler()
        h.on_created(FakeEvent("C:\\x\\a.txt"))
        h.get_and_reset_counts()
        assert h.get_and_reset_counts() == {"nc": 0, "nr": 0, "nu": 0, "nm": 0}

    def test_directories_ignored(self):
        h = FileCountHandler()
        h.on_created(FakeEvent("C:\\x\\sub", is_directory=True))
        h.on_deleted(FakeEvent("C:\\x\\sub", is_directory=True))
        counts = h.get_and_reset_counts()
        assert counts["nc"] == 0 and counts["nu"] == 0

    def test_drain_returns_events_exactly_once(self):
        h = FileCountHandler()
        h.on_created(FakeEvent("C:\\x\\a.txt"))
        h.on_created(FakeEvent("C:\\x\\b.txt"))
        first = h.drain_new_events()
        second = h.drain_new_events()
        assert len(first) == 2
        assert second == []

    def test_drain_buffer_stays_bounded(self):
        h = FileCountHandler()
        for i in range(250):
            h.on_created(FakeEvent(f"C:\\x\\f{i}.txt"))
        h.drain_new_events()
        assert len(h._recent_events) <= 150


class TestRealFolderMonitor:
    """Test monitor construction and tick aggregation."""

    def test_default_folders_resolve_and_exist(self):
        m = RealFolderMonitor()
        names = {f["name"]: f for f in m.get_default_folders()}
        assert {"Desktop", "Downloads", "Documents", "Pictures"} <= set(names)
        for f in names.values():
            if f["exists"]:
                assert os.path.isdir(f["path"])

    def test_missing_folders_filtered_out(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        m = RealFolderMonitor(folders=[str(good), str(tmp_path / "missing")])
        assert m.folders == [str(good)]
        assert len(m.handlers) == 1

    def test_get_tick_aggregates_and_drains(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        m = RealFolderMonitor(folders=[str(good)])
        m.handlers[0].on_created(FakeEvent(str(good / "z.txt")))

        first = m.get_tick()
        second = m.get_tick()

        assert first["source"] == "real"
        assert first["nc"] == 1
        assert len(first["file_events"]) == 1
        assert second["nc"] == 0 and second["file_events"] == []
