"""
Ransomware Detection - Real Folder Monitor

Uses the watchdog library to observe real file system events on
Desktop and Downloads folders. Completely read-only - only counts
file operations per second, never modifies any files.

Closely follows the Aranyi et al. (2026) methodology of capturing
server-side file operations with nanosecond timestamps and aggregating
them into one-second windows of 5 operation types (create/write/read/
rename/delete).

Feature mapping per 1-second window:
    nc  : file creation events
    nw  : file write (modify) events
    nr  : file read/open (access) events
    nm  : file rename/move events
    nu  : file unlink/delete events
"""

import os
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Callable

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    FileModifiedEvent,
    FileOpenedEvent,
)


class FileCountHandler(FileSystemEventHandler):
    """
    Event handler that counts file operations with nanosecond timestamps.

    Uses Windows ReadDirectoryChangesW API under the hood.
    Completely passive - only receives notifications, never modifies anything.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._counts = {"nc": 0, "nw": 0, "nr": 0, "nm": 0, "nu": 0}
        self._recent_events = []

    def on_created(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nc"] += 1
                self._log_event("create", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nu"] += 1
                self._log_event("delete", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nm"] += 1
                self._log_event("rename", f"{event.src_path} -> {event.dest_path}")

    def on_modified(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nw"] += 1
                self._log_event("write", event.src_path)

    def on_opened(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nr"] += 1
                self._log_event("read", event.src_path)

    def _log_event(self, event_type, path):
        """Store last 50 events for display."""
        self._recent_events.append({
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "path": os.path.basename(path),
            "full_path": path,
            "folder": os.path.dirname(path),
            "ns": time.time_ns(),
        })
        if len(self._recent_events) > 50:
            self._recent_events = self._recent_events[-50:]

    def get_and_reset_counts(self) -> Dict:
        """Get current counts and reset. Thread-safe."""
        with self._lock:
            counts = self._counts.copy()
            self._counts = {"nc": 0, "nw": 0, "nr": 0, "nm": 0, "nu": 0}
            return counts

    def get_recent_events(self) -> List[Dict]:
        """Get recent file events."""
        with self._lock:
            return list(self._recent_events)


class RealFolderMonitor:
    """
    Monitors real folders (Desktop, Downloads) for file system events.

    Uses watchdog's Observer which hooks into Windows ReadDirectoryChangesW.
    This is a kernel-level notification mechanism - completely passive observation.

    Features:
    - Watches multiple folders simultaneously
    - Aggregates events per second into nc/nw/nr/nm/nu counts
    - Nanosecond timestamps on events
    - Thread-safe event counting
    - No file modification, no file reads, no file creation
    """

    def __init__(self, folders: List[str] = None):
        if folders is None:
            home = os.path.expanduser("~")
            folders = [
                os.path.join(home, "Desktop"),
                os.path.join(home, "Downloads"),
            ]

        self.folders = [f for f in folders if os.path.isdir(f)]
        self.observers = []
        self.handlers = []
        self.is_running = False
        self._lock = threading.Lock()
        self._event_log = []
        self._total_events = 0

        for folder in self.folders:
            handler = FileCountHandler()
            self.handlers.append(handler)

    def start(self) -> Dict:
        """Start monitoring all configured folders."""
        if self.is_running:
            return {"success": False, "message": "Already running"}

        self.observers = []
        results = []

        for folder, handler in zip(self.folders, self.handlers):
            try:
                observer = Observer()
                observer.schedule(handler, folder, recursive=True)
                observer.start()
                self.observers.append(observer)
                results.append({"folder": folder, "status": "started"})
                print(f"[MONITOR] Watching: {folder}")
            except Exception as e:
                results.append({"folder": folder, "status": f"error: {str(e)}"})
                print(f"[MONITOR] Error watching {folder}: {e}")

        self.is_running = len(self.observers) > 0

        return {
            "success": self.is_running,
            "folders": results,
            "message": f"Monitoring {len(self.observers)} folders",
        }

    def stop(self) -> Dict:
        """Stop all monitoring."""
        if not self.is_running:
            return {"success": False, "message": "Not running"}

        for observer in self.observers:
            observer.stop()

        for observer in self.observers:
            observer.join(timeout=2)

        self.observers = []
        self.is_running = False
        print("[MONITOR] Stopped all folder monitoring")

        return {"success": True, "message": "Monitoring stopped"}

    def get_tick(self) -> Dict:
        """Get aggregated file operation counts for the current second."""
        total_nc = 0
        total_nw = 0
        total_nr = 0
        total_nm = 0
        total_nu = 0
        tick_events = []

        for handler in self.handlers:
            counts = handler.get_and_reset_counts()
            total_nc += counts["nc"]
            total_nw += counts["nw"]
            total_nr += counts["nr"]
            total_nm += counts["nm"]
            total_nu += counts["nu"]
            tick_events.extend(handler.get_recent_events())

        self._total_events += total_nc + total_nw + total_nr + total_nm + total_nu

        tick_events.sort(key=lambda x: x.get("ns", 0), reverse=True)
        tick_events = tick_events[:20]

        return {
            "nc": total_nc,
            "nw": total_nw,
            "nr": total_nr,
            "nm": total_nm,
            "nu": total_nu,
            "att": 0,
            "user": "Real System",
            "family": None,
            "is_attack": False,
            "source": "real",
            "file_events": tick_events,
        }

    def get_recent_events(self) -> List[Dict]:
        """Get recent file events from all monitored folders."""
        events = []
        for handler in self.handlers:
            events.extend(handler.get_recent_events())

        events.sort(key=lambda x: x.get("ns", 0), reverse=True)
        return events[:30]

    def get_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            "is_running": self.is_running,
            "folders": self.folders,
            "total_events": self._total_events,
            "observer_count": len(self.observers),
        }

    def get_default_folders(self) -> List[Dict]:
        """Get default monitoring folders with existence check."""
        home = os.path.expanduser("~")
        candidates = [
            ("Desktop", os.path.join(home, "Desktop")),
            ("Downloads", os.path.join(home, "Downloads")),
            ("Documents", os.path.join(home, "Documents")),
            ("Pictures", os.path.join(home, "Pictures")),
        ]

        folders = []
        for name, path in candidates:
            folders.append({
                "name": name,
                "path": path,
                "exists": os.path.isdir(path),
                "selected": name in ("Desktop", "Downloads"),
            })

        return folders


def get_monitor(folders: List[str] = None) -> RealFolderMonitor:
    """Factory function to get monitor instance."""
    return RealFolderMonitor(folders)
