"""
Ransomware Detection - Real Folder Monitor

Uses the watchdog library to observe real file system events on
Desktop and Downloads folders. Completely read-only - only counts
create/rename/delete events per second, never modifies any files.

Based on Aranyi et al. (2026):
- Watches centralized file operations using OS-level notifications
- Aggregates events into 1-second windows (nc, nr, nu)
- Feeds aggregated counts to ML models for classification
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
)


# Registry values for Windows known folders (handles OneDrive redirection)
_WINREG_KNOWN_FOLDERS = {
    "Desktop": "Desktop",
    "Documents": "Personal",
    "Pictures": "My Pictures",
    "{374DE290-123F-4565-9164-39C4925E467B}": "Downloads",
}

_FOLDER_FALLBACKS = {
    "Desktop": ["Desktop", os.path.join("OneDrive", "Desktop")],
    "Downloads": ["Downloads", os.path.join("OneDrive", "Downloads")],
    "Documents": ["Documents", os.path.join("OneDrive", "Documents")],
    "Pictures": ["Pictures", os.path.join("OneDrive", "Pictures")],
}


def resolve_known_folders() -> Dict[str, str]:
    """
    Resolve the real paths of user folders (Desktop, Downloads, Documents,
    Pictures). Reads HKCU User Shell Folders from the registry first so
    OneDrive-redirected folders are detected correctly; falls back to
    common paths under the user profile.
    """
    resolved = {}

    try:
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            for reg_value, name in _WINREG_KNOWN_FOLDERS.items():
                try:
                    raw_value, _ = winreg.QueryValueEx(key, reg_value)
                    if not isinstance(raw_value, str):
                        continue
                    path = os.path.expandvars(raw_value.strip())
                    if name not in resolved and os.path.isdir(path):
                        resolved[name] = path
                except OSError:
                    continue
    except (ImportError, OSError):
        pass

    home = os.path.expanduser("~")
    for name, rel_candidates in _FOLDER_FALLBACKS.items():
        if name in resolved:
            continue
        for rel in rel_candidates:
            candidate = os.path.join(home, rel)
            if os.path.isdir(candidate):
                resolved[name] = candidate
                break

    return resolved


class FileCountHandler(FileSystemEventHandler):
    """
    Event handler that counts file operations.
    
    Uses Windows ReadDirectoryChangesW API under the hood.
    Completely passive - only receives notifications, never modifies anything.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._counts = {"nc": 0, "nr": 0, "nu": 0, "nm": 0}
        self._recent_events = []
        self._drain_index = 0

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
                self._counts["nr"] += 1
                self._log_event("rename", f"{event.src_path} -> {event.dest_path}")

    def on_modified(self, event):
        if not event.is_directory:
            with self._lock:
                self._counts["nm"] += 1

    def _log_event(self, event_type, path):
        """Store last 50 events for display."""
        self._recent_events.append({
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "path": os.path.basename(path),
            "full_path": path,
            "folder": os.path.dirname(path),
        })
        if len(self._recent_events) > 50:
            self._recent_events = self._recent_events[-50:]

    def get_and_reset_counts(self) -> Dict:
        """Get current counts and reset. Thread-safe."""
        with self._lock:
            counts = self._counts.copy()
            self._counts = {"nc": 0, "nr": 0, "nu": 0, "nm": 0}
            return counts

    def get_recent_events(self) -> List[Dict]:
        """Get recent file events."""
        with self._lock:
            return list(self._recent_events)

    def drain_new_events(self) -> List[Dict]:
        """
        Get events logged since the last drain and mark them consumed.
        Thread-safe. Keeps the internal buffer bounded.
        """
        with self._lock:
            new_events = list(self._recent_events[self._drain_index:])
            self._drain_index = len(self._recent_events)

            # Keep the buffer bounded; reset the cursor after a trim
            if len(self._recent_events) > 200:
                self._recent_events = self._recent_events[-100:]
                self._drain_index = len(self._recent_events)

            return new_events


class RealFolderMonitor:
    """
    Monitors real folders (Desktop, Downloads) for file system events.
    
    Uses watchdog's Observer which hooks into Windows ReadDirectoryChangesW.
    This is a kernel-level notification mechanism - completely passive observation.
    
    Features:
    - Watches multiple folders simultaneously
    - Aggregates events per second into nc/nr/nu counts
    - Thread-safe event counting
    - No file modification, no file reads, no file creation
    """

    def __init__(self, folders: List[str] = None):
        """
        Initialize the real folder monitor.
        
        Args:
            folders: List of folder paths to monitor.
                     Defaults to user's Desktop and Downloads.
        """
        if folders is None:
            known = resolve_known_folders()
            home = os.path.expanduser("~")
            fallback_desktop = os.path.join(home, "OneDrive", "Desktop")
            fallback_downloads = os.path.join(home, "Downloads")
            folders = [
                known.get("Desktop") or (fallback_desktop if os.path.isdir(fallback_desktop) else os.path.join(home, "Desktop")),
                known.get("Downloads") or fallback_downloads,
            ]

        self.folders = [f for f in folders if os.path.isdir(f)]
        self.observers = []
        self.handlers = []
        self.is_running = False
        self._lock = threading.Lock()
        self._event_log = []
        self._total_events = 0

        # Create a handler for each folder
        for folder in self.folders:
            handler = FileCountHandler()
            self.handlers.append(handler)

    def start(self) -> Dict:
        """
        Start monitoring all configured folders.
        
        Returns:
            Status dict with folder list and success flag.
        """
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
        """
        Get aggregated file operation counts for the current second.
        
        This is called every second by the main loop.
        Returns counts from ALL monitored folders combined,
        plus recent file-level events for display.
        """
        total_nc = 0
        total_nr = 0
        total_nu = 0
        tick_events = []

        for handler in self.handlers:
            counts = handler.get_and_reset_counts()
            total_nc += counts["nc"]
            total_nr += counts["nr"]
            total_nu += counts["nu"]
            # Collect only events that occurred since the previous tick
            tick_events.extend(handler.drain_new_events())

        self._total_events += total_nc + total_nr + total_nu

        # Keep only the most recent events from this tick
        tick_events.sort(key=lambda x: x["time"], reverse=True)
        tick_events = tick_events[:20]

        return {
            "nc": total_nc,
            "nr": total_nr,
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

        # Sort by time, most recent first
        events.sort(key=lambda x: x["time"], reverse=True)
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
        known = resolve_known_folders()
        candidates = [
            ("Desktop", known.get("Desktop")),
            ("Downloads", known.get("Downloads")),
            ("Documents", known.get("Documents")),
            ("Pictures", known.get("Pictures")),
        ]

        folders = []
        for name, path in candidates:
            if path is None:
                home = os.path.expanduser("~")
                path = os.path.join(home, name)
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
