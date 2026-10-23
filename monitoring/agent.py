"""Operational monitoring: hysteresis, abstention, canary ground truth, multi-share.

Four gaps between the base paper's evaluation and a deployable monitor.

Hysteresis
    The paper classifies each one-second window independently at a fixed
    threshold. A deployment that does the same flips its state on any single
    quiet second, so an operator watching the console sees an alarm clear and
    re-raise within one window of a real attack. Confirmation counts on the way
    up and on the way down remove that flicker without changing which windows are
    scored.

Three-state output with abstention
    Every window is forced into benign or attack, even when the model is
    undecided. The paper's fixed 0.5 threshold guarantees a decision is always
    emitted. For a cost-sensitive system an explicit "insufficient evidence" state
    is more honest and can be routed to a slower review queue instead of either
    alarming or dismissing.

Canary-based ground truth
    The paper derives its ``att`` field from a canary file: a bash script checks
    every second whether ``b.docx`` still exists, and treats its disappearance as
    proof that ransomware is active (Sect. 4.2). That mechanism is reproducible
    here so the label is grounded in an observable fact rather than in whoever was
    pressing the simulator button.

Multiple shares
    The paper evaluates a single file server and claims enterprise applicability
    without evidence. Per-share monitors with independent histories are needed for
    that claim to be testable at all.
"""

import hashlib
import os
import time
from collections import deque
from pathlib import Path

CANARY_NAME = "b.docx"
DEFAULT_HISTORY = 30


class HysteresisState:
    """Confirm-before-transition alarm state.

    ``raise_after`` positive windows in a row promote CLEAR -> ALARMING, and
    ``clear_after`` consecutive quiet windows demote it. The asymmetry is
    deliberate: raising late is worse operationally than clearing late, because a
    missed attack costs 10x a false alarm under the paper's own cost matrix.
    """

    def __init__(self, raise_after=2, clear_after=3):
        if raise_after < 1 or clear_after < 1:
            raise ValueError("confirmation counts must be at least 1")
        self.raise_after = raise_after
        self.clear_after = clear_after
        self.state = "CLEAR"
        self._pos = 0
        self._neg = 0
        self.transitions = []

    def update(self, is_positive, timestamp=None):
        if is_positive:
            self._pos += 1
            self._neg = 0
        else:
            self._neg += 1
            self._pos = 0

        previous = self.state
        if self.state == "CLEAR":
            if self._pos >= self.raise_after:
                self.state = "ALARMING"
        else:
            if self._neg >= self.clear_after:
                self.state = "CLEAR"

        if self.state != previous:
            self.transitions.append({
                "From": previous, "To": self.state,
                "Timestamp": timestamp,
                "Confirmed Positive Run": self._pos,
                "Confirmed Negative Run": self._neg,
            })
        return self.state

    def reset(self):
        self.state = "CLEAR"
        self._pos = 0
        self._neg = 0


class AbstainingClassifier:
    """Three-state decision with an explicit undecided band.

    The paper fixes the threshold at 0.5 for every model (Sect. 3.6). This wraps
    that fixed point with a band around it so a probability inside the band is
    reported as ``UNSURE`` instead of being forced to a binary call. The band is
    the paper's decision point widened, not a replacement for it: 0.5 remains the
    centre.
    """

    NORMAL = "NORMAL"
    RANSOMWARE = "RANSOMWARE"
    UNSURE = "UNSURE"

    def __init__(self, threshold=0.5, margin=0.10):
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self.threshold = threshold
        self.margin = margin
        self.history = deque(maxlen=64)
        self.counts = {self.NORMAL: 0, self.RANSOMWARE: 0, self.UNSURE: 0}

    def classify(self, proba, timestamp=None):
        p = float(proba)
        low = self.threshold - self.margin
        high = self.threshold + self.margin

        if p < low:
            state = self.NORMAL
        elif p > high:
            state = self.RANSOMWARE
        else:
            state = self.UNSURE

        self.history.append({"Probability": round(p, 6), "State": state,
                             "Timestamp": timestamp})
        self.counts[state] = self.counts.get(state, 0) + 1
        return state

    def summary(self):
        total = sum(self.counts.values())
        return {
            "Threshold": self.threshold,
            "Margin": self.margin,
            "Total Windows": total,
            "Counts": dict(self.counts),
            "Abstention Rate": round(self.counts[self.UNSURE] / total, 6)
            if total else 0.0,
        }

    def reset(self):
        self.history.clear()
        for key in self.counts:
            self.counts[key] = 0


class CanaryMonitor:
    """Canary-file ground truth, following the paper's Sect. 4.2 mechanism.

    A file that benign activity never touches is placed in the monitored tree. Its
    disappearance can only be explained by the attack scenario touching it, so the
    label becomes an observation rather than an assertion. The paper restores the
    attacker host from a snapshot on detection; the filesystem-level equivalent is
    to rewrite the canary and record the transition.
    """

    def __init__(self, directory, canary_name=CANARY_NAME):
        self.directory = Path(directory)
        self.canary_name = canary_name
        self.canary_path = self.directory / canary_name
        self.present = False
        self.events = []
        self._last_check = None

    def ensure_canary(self, content="canary\n"):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.canary_path.write_text(content)
        self.present = True
        return self.canary_path

    def canary_hash(self):
        if not self.canary_path.exists():
            return None
        return hashlib.sha256(self.canary_path.read_bytes()).hexdigest()

    def check(self, timestamp=None):
        """Return ``True`` while the canary is intact, ``False`` once gone."""
        present = self.canary_path.exists()
        if self.present and not present:
            self.events.append({
                "Event": "canary_removed",
                "Timestamp": timestamp if timestamp is not None else time.time(),
                "Previous Hash": self._last_check,
            })
        elif not self.present and present:
            self.events.append({
                "Event": "canary_restored",
                "Timestamp": timestamp if timestamp is not None else time.time(),
                "New Hash": self.canary_hash(),
            })
        self.present = present
        if present:
            self._last_check = self.canary_hash()
        return present

    def restore(self):
        """Rewrite the canary after an attack touched it.

        The internal presence flag is deliberately left as-is. ``ensure_canary``
        sets it to True, which would suppress the canary_restored event that the
        next ``check`` is supposed to record, and the recovery would go unlogged.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self.canary_path.write_text("canary\n")
        return self.canary_path

    def status(self):
        return {
            "Canary Path": str(self.canary_path),
            "Present": self.canary_path.exists(),
            "Hash": self.canary_hash(),
            "Events": self.events,
        }


class ShareMonitor:
    """One monitored share: its own event buffer, history and alarm state.

    A deployment watching several file servers cannot share a single rolling
    history between them, because activity on one share would populate the
    temporal features of another and produce phantom detections. Each share
    therefore keeps its own buffers.
    """

    def __init__(self, name, directory, observer=None, history_length=DEFAULT_HISTORY,
                 raise_after=2, clear_after=3, margin=0.10):
        self.name = name
        self.directory = Path(directory)
        self.events = deque()
        self.counts = deque(maxlen=history_length)
        self.hysteresis = HysteresisState(raise_after, clear_after)
        self.abstainer = AbstainingClassifier(margin=margin)
        self.observer = observer
        self.last_counts = {"create": 0, "rename": 0, "delete": 0}

    def record(self, operation, path=None, timestamp=None):
        self.events.append({
            "op": operation,
            "path": str(path) if path else None,
            "t": timestamp if timestamp is not None else time.time(),
        })

    def aggregate(self, now=None):
        """Roll the buffered events into one window and clear the buffer."""
        now = now if now is not None else time.time()
        cutoff = now - 1.0
        recent = [e for e in self.events if e["t"] >= cutoff]
        self.events.clear()

        counts = {
            "create": sum(1 for e in recent if e["op"] == "create"),
            "rename": sum(1 for e in recent if e["op"] == "rename"),
            "delete": sum(1 for e in recent if e["op"] == "delete"),
        }
        self.counts.append(counts)
        self.last_counts = counts
        return counts, recent

    def feature_row(self):
        """Count history in the form ``ml.streaming`` expects."""
        return [{"create_count": c["create"], "rename_count": c["rename"],
                 "delete_count": c["delete"]} for c in self.counts]

    def evaluate(self, proba, timestamp=None):
        state = self.abstainer.classify(proba, timestamp)
        alarm = self.hysteresis.update(proba >= 0.5, timestamp)
        return {
            "Share": self.name,
            "Probability": round(float(proba), 6),
            "Three State": state,
            "Alarm": alarm,
            "Counts": dict(self.last_counts),
        }

    def status(self):
        return {
            "Share": self.name,
            "Directory": str(self.directory),
            "Buffered Events": len(self.events),
            "History Depth": len(self.counts),
            "Alarm State": self.hysteresis.state,
            "Abstention": self.abstainer.summary(),
        }


class MonitorFleet:
    """Collection of shares monitored together."""

    def __init__(self, shares):
        self.shares = list(shares)

    def add(self, share):
        self.shares.append(share)
        return share

    def tick(self, now=None):
        """Aggregate and classify every share.

        Returns one record per share. Shares that fail to classify are reported
        with their error rather than being dropped, so a fault on one share is
        visible rather than hidden behind a healthy aggregate.
        """
        now = now if now is not None else time.time()
        records = []
        for share in self.shares:
            counts, _ = share.aggregate(now)
            try:
                record = share.evaluate(self._predict(share, counts), now)
            except Exception as exc:  # noqa: BLE001
                record = {"Share": share.name, "Error": str(exc),
                          "Counts": dict(counts)}
            records.append(record)
        return records

    def _predict(self, share, counts):
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("no model attached to the fleet")

        from ml.streaming import add_sequence_features
        import pandas as pd

        history = share.feature_row()
        current = [{"create_count": counts["create"],
                    "rename_count": counts["rename"],
                    "delete_count": counts["delete"]}]
        frame = pd.DataFrame(history + current)

        feature_set = getattr(self, "feature_set", "sequence")
        if feature_set == "causal":
            from ml.streaming import add_causal_features
            enriched = add_causal_features(frame)
        else:
            enriched = add_sequence_features(frame)

        columns = getattr(self, "feature_columns", None) or \
            [c for c in enriched.columns
             if c not in ("label", "episode_id", "family", "is_stealth",
                          "benign_template", "window_index", "ops_total",
                          "is_active_window", "ops_delta")]

        row = enriched[columns].tail(1).values.astype(float)
        # tail(1) yields a single row, but numpy can collapse that to 1-D
        # depending on the pandas version. Reshape so the (1, n_features) shape
        # a classifier expects is guaranteed rather than assumed.
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if getattr(self, "scaler", None) is not None:
            row = self.scaler.transform(row)
        return float(self.model.predict_proba(row)[0, 1])

    def attach(self, model, feature_columns=None, feature_set="sequence",
               scaler=None):
        self.model = model
        self.feature_columns = feature_columns
        self.feature_set = feature_set
        self.scaler = scaler
        return self

    def status(self):
        return {
            "Shares": [s.status() for s in self.shares],
            "Total Shares": len(self.shares),
            "Any Alarm": any(s.hysteresis.state == "ALARMING" for s in self.shares),
        }
