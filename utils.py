"""
Ransomware Detection System - Utility Functions
Shared helpers used across the application.
"""
import time
import hashlib
from typing import Dict


def format_bytes(size: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def format_timestamp(ts: float = None) -> str:
    """Format timestamp to HH:MM:SS string."""
    if ts is None:
        ts = time.time()
    return time.strftime("%H:%M:%S", time.localtime(ts))


def file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for integrity checking."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (OSError, IOError):
        return ""


def safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.strip()


def calculate_ratios(nc: int, nr: int, nu: int) -> Dict[str, float]:
    """Calculate operation ratios for analysis."""
    total = nc + nr + nu
    if total == 0:
        return {"create_ratio": 0, "rename_ratio": 0, "delete_ratio": 0}
    return {
        "create_ratio": nc / total,
        "rename_ratio": nr / total,
        "delete_ratio": nu / total,
    }


def classify_risk(nc: int, nr: int, nu: int) -> str:
    """Simple heuristic risk classification based on operation counts."""
    total = nc + nr + nu
    if total == 0:
        return "idle"
    if nr > total * 0.4 and nc > 10:
        return "critical"
    if nr > total * 0.3 and nc > 5:
        return "warning"
    if total > 50:
        return "elevated"
    return "normal"
