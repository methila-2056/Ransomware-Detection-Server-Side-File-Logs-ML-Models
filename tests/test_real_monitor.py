"""
Tests for the real folder monitor module.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from real_monitor import (
    FileCountHandler,
    RealFolderMonitor,
)


class FakeEvent:
    """Minimal stand-in for watchdog file events (duck-typed)."""

    def __init__(self, src, dest=None, is_directory=False):
        self.src_path = src
        self.dest_path = dest
        self.is_directory = is_directory


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Registry-based known folders only resolve on Windows")
class TestResolveKnownFolders:
    """Test Windows known-folder resolution (OneDrive-aware)."""

    def test_returns_common_folders(self):
        m = RealFolderMonitor()
        folders = m.get_default_folders()
        names = {f["name"] for f in folders}
        assert "Desktop" in names
        assert "Downloads" in names

    def test_resolved_paths_exist(self):
        m = RealFolderMonitor()
        for f in m.get_default_folders():
            if f.get("exists"):
                assert os.path.isdir(f["path"]), f"{f['name']} -> {f['path']} does not exist"

    def test_paths_under_user_profile(self):
        home = os.path.expanduser("~")
        m = RealFolderMonitor()
        for f in m.get_default_folders():
            if f.get("exists"):
                assert f["path"].lower().startswith(home.lower())

    def test_returns_dict_of_strings(self):
        m = RealFolderMonitor()
        folders = m.get_default_folders()
        assert isinstance(folders, list)
        for f in folders:
            assert isinstance(f.get("name"), str)
            assert isinstance(f.get("path"), str)


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
        assert counts["nw"] == 1
        assert counts["nu"] == 1
        assert counts["nm"] == 1

    def test_reset_after_read(self):
        h = FileCountHandler()
        h.on_created(FakeEvent("C:\\x\\a.txt"))
        h.get_and_reset_counts()
        assert h.get_and_reset_counts() == {"nc": 0, "nw": 0, "nr": 0, "nm": 0, "nu": 0}

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
        first = h.get_recent_events()
        second = h.get_recent_events()
        assert len(first) == 2
        assert second == first


    def test_drain_buffer_stays_bounded(self):
        h = FileCountHandler()
        for i in range(250):
            h.on_created(FakeEvent(f"C:\\x\\f{i}.txt"))
        h.get_recent_events()
        assert len(h._recent_events) <= 50


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
        assert second["nc"] == 0
        assert len(second["file_events"]) == 1
