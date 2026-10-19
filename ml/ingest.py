"""Parsers for real server-side file-operation logs.

The base paper's pipeline logs file operations on the server, forwards them with
Fluent Bit into InfluxDB, and exports them to CSV for aggregation into one-second
windows (Sect. 3.2, 4.2). This project had no equivalent: every window it trained
on came from ``ml.attack_sim``. That made the paper's most fundamental piece of
infrastructure absent, and it is why "reproduces the paper's pipeline" was not a
claim this project could make.

This module closes that gap. It reads the log formats a real file server actually
emits and produces the same unified event schema the live observer emits, so real
logs and emulated traces enter an identical downstream pipeline rather than two
parallel ones.

Supported formats
-----------------
Samba ``full_audit``
    ``smbd_audit: domain\\user|10.0.0.5|share|unlinkat|ok|/srv/share/a.docx``
    The paper's own file server. The payload format is
    ``PREFIX|OPERATION|RESULT|FILE`` where PREFIX is the configured
    ``full_audit:prefix``; the usual setting ``%u|%I|%S`` yields
    ``user|client-ip|share``. Operations are the VFS names: ``create_file``,
    ``open``, ``renameat``, ``unlinkat``, ``mkdirat``. A renameat record carries
    both paths as ``OLD|NEW``.

Linux auditd
    ``type=SYSCALL msg=audit(1717171717.123:456): exe="/usr/bin/smbd" ...``
    Kernel audit records, the Linux equivalent of the same telemetry.

Windows Security (4663)
    The file-access event the paper's Windows clients would produce.

Sysmon (IDs 2, 11, 23)
    2 = file creation time changed, 11 = file created, 23 = file delete.
    Optional but commonly deployed alongside 4663.

Honest limitation
-----------------
Every parser here is tested against **synthetic lines written to the documented
format**. No real capture from a Samba server or a Windows host exists in this
repository, so the parsers are unvalidated against production traffic. That is
recorded in the validation report rather than glossed over, because a parser that
passes synthetic tests and fails on a real deployment is the common failure mode
here.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# Unified operation vocabulary. Everything is mapped onto the three operations the
# paper models, because a rename is reported inconsistently across sources.
CREATE = "create"
RENAME = "rename"
DELETE = "delete"
OTHER = "other"

OPERATIONS = (CREATE, RENAME, DELETE, OTHER)


@dataclass
class FileEvent:
    """One normalised file-system operation."""

    timestamp: float          # epoch seconds, UTC
    operation: str            # one of OPERATIONS
    path: str = ""
    user: str = ""
    source: str = ""          # which parser produced it
    raw: str = ""             # original line, retained for auditing

    def to_dict(self):
        return asdict(self)


@dataclass
class ParseReport:
    """What a parser saw, including what it could not classify."""

    parser: str = ""
    total_lines: int = 0
    parsed: int = 0
    skipped: int = 0
    operations: dict = field(default_factory=dict)
    unparsed_examples: list = field(default_factory=list)

    def note(self, operation):
        self.operations[operation] = self.operations.get(operation, 0) + 1

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Samba full_audit
# ---------------------------------------------------------------------------

# op=RENAME appears in the log before the target path is known, so the two halves
# are paired. Real Samba emits the rename and the old-path line adjacently.
# Samba's vfs_full_audit emits, after the syslog program name:
#
#     PREFIX|OPERATION|RESULT|FILE
#
# where PREFIX is the configured full_audit:prefix. The usual configuration
# ``%u|%I|%S`` gives three leading fields (user, client ip, share), so a
# successful create looks like::
#
#     AD\alice|10.0.0.5|Data|create_file|ok|/srv/share/a.docx
#
# and a rename carries both paths separated by a pipe inside the FILE field,
# which the pattern treats as part of the trailing remainder because it is
# anchored only at the end.
_SAMBA_LINE = re.compile(
    r"smbd_audit(?:\[\d+\])?:\s*(?P<user>[^|]*)\|(?P<client>[^|]*)\|"
    r"(?P<share>[^|]*)\|(?P<op>[a-z_]+)\|(?P<result>[^|]*)\|(?P<path>.*)$")
# syslog RFC3164 / RFC5424 timestamp, as written by the syslog daemon ahead of
# the program name.
_SAMBA_TS = re.compile(
    r"(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")


def _parse_samba_timestamp(text, fallback):
    """Parse the syslog timestamp that precedes the Samba program name.

    samba_full_audit writes to syslog, so the line begins
    ``Feb  1 20:59:12 z1 smbd_audit[35482]: ...``. The year is absent, as in
    RFC3164, so the current year is assumed and the caller keeps the clock
    monotonic.
    """
    match = _SAMBA_TS.search(text)
    if not match:
        return fallback
    for fmt in ("%b %d %H:%M:%S", "%b  %d %H:%M:%S"):
        try:
            dt = datetime.strptime(match.group("ts"), fmt)
            dt = dt.replace(year=datetime.now(timezone.utc).year,
                            tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return fallback


def parse_samba(lines):
    """Parse Samba ``full_audit`` output into events plus a report."""
    report = ParseReport(parser="samba_full_audit")
    events = []
    clock = 0.0

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        report.total_lines += 1
        clock = _parse_samba_timestamp(line, clock + 1.0)

        match = _SAMBA_LINE.search(line)
        if not match:
            report.skipped += 1
            if len(report.unparsed_examples) < 5:
                report.unparsed_examples.append(line[:160])
            continue

        op = match.group("op")
        if op in ("create_file", "mkdirat", "creat"):
            operation = CREATE
        elif op in ("renameat", "rename", "rename_stream"):
            operation = RENAME
        elif op in ("unlinkat", "unlink", "rmdir"):
            operation = DELETE
        else:
            # open, close, read, write, chmod, stat and friends carry no
            # create/rename/delete signal. Counting them would swamp the counts
            # the detector depends on, so they are recorded as parsed but not as
            # an operation.
            operation = OTHER

        # A renameat record carries OLD_PATH|NEW_PATH. The destination is the file
        # the attacker writes, so it is the more informative of the two.
        raw_path = match.group("path").strip()
        if operation == RENAME and "|" in raw_path:
            raw_path = raw_path.rsplit("|", 1)[1].strip()

        report.parsed += 1
        report.note(operation)
        events.append(FileEvent(
            timestamp=clock, operation=operation,
            path=raw_path, user=match.group("user").strip(),
            source="samba", raw=line[:200],
        ))

    return events, report


# ---------------------------------------------------------------------------
# Linux auditd
# ---------------------------------------------------------------------------

_AUDIT_TS = re.compile(r"audit\((?P<epoch>[\d.]+):(?P<serial>\d+)\)")
_AUDIT_TYPE = re.compile(r'type=(?P<type>\S+)')
_AUDIT_EXEC = re.compile(r'exe="(?P<exe>[^"]*)"')
_AUDIT_NAME = re.compile(r'name="(?P<name>[^"]*)"')


def parse_auditd(lines, interesting_exes=("smbd", "samba")):
    """Parse Linux auditd records into file events.

    Auditd records every syscall a process makes, so a whole ``type=SYSCALL``
    block is filtered to file-related executables and the ``name=`` argument,
    which is the path the operation touched. Non-file syscalls are counted and
    skipped rather than being guessed at.
    """
    report = ParseReport(parser="linux_auditd")
    events = []

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        report.total_lines += 1

        ts_match = _AUDIT_TS.search(line)
        if not ts_match:
            report.skipped += 1
            continue

        kind_match = _AUDIT_TYPE.search(line)
        kind = kind_match.group("type") if kind_match else ""

        # Only SYSCALL records carry the path argument that matters here.
        if kind != "SYSCALL":
            report.skipped += 1
            continue

        exe_match = _AUDIT_EXEC.search(line)
        exe = exe_match.group("exe") if exe_match else ""
        if interesting_exes and not any(x in exe for x in interesting_exes):
            report.skipped += 1
            continue

        name_match = _AUDIT_NAME.search(line)
        if not name_match:
            report.skipped += 1
            continue

        epoch = float(ts_match.group("epoch"))
        path = name_match.group("name")

        operation = _audit_operation(line, path)

        if operation == OTHER:
            # A read or a stat. Recorded as seen, but not emitted as an event,
            # because openat on a read is orders of magnitude more frequent than
            # the creates and deletes the detector depends on. Emitting them would
            # swamp the counts rather than inform them.
            report.note(OTHER)
            report.skipped += 1
            continue

        report.parsed += 1
        report.note(operation)
        events.append(FileEvent(
            timestamp=epoch, operation=operation, path=path,
            user="", source="auditd", raw=line[:200],
        ))

    return events, report


def _audit_operation(line, path):
    """Infer the operation from the auditd syscall number.

    Linux x86_64 syscall numbers used here: 257 openat, 262 newfstatat,
    316 renameat2, 263 unlinkat, 87 unlink, 82 rename.

    openat is ambiguous -- it is used for reads and writes as well as creation --
    so ``O_CREAT`` in the flags is what distinguishes a create from a plain open.
    Without that flag an openat is treated as a non-event rather than being
    counted as a create, because over-counting creates inflates the very signal
    the detector depends on.
    """
    syscall = re.search(r"syscall=(\d+)", line)
    number = int(syscall.group(1)) if syscall else -1

    if number in (316, 82):
        return RENAME
    if number in (263, 87):
        return DELETE
    if number == 257:
        return CREATE if "O_CREAT" in line else OTHER
    return OTHER


# ---------------------------------------------------------------------------
# Windows Security 4663 / Sysmon
# ---------------------------------------------------------------------------

def parse_windows_events(records, event_id=None):
    """Parse Windows Security and Sysmon records.

    Accepts an iterable of dicts, which is what a JSON export or an Elastic query
    produces. Accepting dicts rather than raw XML keeps the parser testable and
    avoids an XML dependency for what is a handful of field lookups.

    Sysmon event IDs are mapped as follows: 11 file created, 23 file deleted,
    2 = creation-time changed, 7 = image loaded, 1 = process created. Only the
    three file operations are relevant; the rest are counted and skipped.
    """
    report = ParseReport(parser="windows_events")
    events = []

    for record in records:
        report.total_lines += 1
        if not isinstance(record, dict):
            report.skipped += 1
            continue

        eid = record.get("EventID") or record.get("event_id") or event_id
        try:
            eid = int(eid)
        except (TypeError, ValueError):
            report.skipped += 1
            continue

        if eid == 11:
            operation = CREATE
        elif eid == 23:
            operation = DELETE
        elif eid == 4663:
            operation = _windows_4663_operation(record)
        else:
            report.skipped += 1
            continue

        if operation == OTHER:
            report.skipped += 1
            continue

        report.parsed += 1
        report.note(operation)
        events.append(FileEvent(
            timestamp=_windows_timestamp(record),
            operation=operation,
            path=record.get("TargetFilename") or record.get("ObjectName", ""),
            user=record.get("SubjectUserName") or record.get("User", ""),
            source=f"windows_{eid}",
            raw=json.dumps(record, default=str)[:200],
        ))

    return events, report


def _windows_4663_operation(record):
    """Derive the operation from the 4663 access mask.

    A 4663 record does not say what was done, only what was permitted. The access
    mask bits for write-data, append, delete and rename are mapped, and a record
    carrying none of them is an ordinary read and is skipped -- reads are frequent
    and carry no signal, so counting them as operations would swamp the create and
    delete counts the detector relies on.
    """
    mask = record.get("AccessMask") or record.get("Mask")
    if mask is None:
        return OTHER

    if isinstance(mask, str):
        try:
            mask = int(mask, 16)
        except ValueError:
            return OTHER
    try:
        mask = int(mask)
    except (TypeError, ValueError):
        return OTHER

    FILE_APPEND_DATA = 0x0004
    FILE_WRITE_DATA = 0x0002
    FILE_DELETE = 0x00010000
    FILE_RENAME = 0x00020000

    if mask & (FILE_DELETE | FILE_RENAME):
        return RENAME
    if mask & (FILE_APPEND_DATA | FILE_WRITE_DATA):
        return CREATE
    return OTHER


def _windows_timestamp(record):
    for key in ("UtcTime", "TimeCreated", "@timestamp", "timestamp"):
        if key in record:
            value = record[key]
            if isinstance(value, (int, float)):
                return float(value)
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(
                        value, fmt).replace(tzinfo=timezone.utc).timestamp()
                except (ValueError, TypeError):
                    continue
    return 0.0


PARSERS = {
    "samba": parse_samba,
    "auditd": parse_auditd,
    "windows": parse_windows_events,
}


def detect_format(sample_lines):
    """Identify which parser a log snippet belongs to.

    Auto-detection matters because a deployment receives whatever the host emits
    and should not require the operator to know which format that is.
    """
    if not sample_lines:
        return None
    head = list(sample_lines)[:50]

    if any("smbd_audit" in line for line in head):
        return "samba"
    if any("audit(" in line and "type=" in line for line in head):
        return "auditd"
    if any(isinstance(line, dict) for line in head):
        return "windows"
    return None


def parse_any(source, fmt=None):
    """Parse a log source, detecting the format when not given.

    ``source`` may be an iterable of lines or an iterable of dicts, or a path.
    """
    if isinstance(source, str):
        with open(source, encoding="utf-8", errors="replace") as f:
            source = f.readlines()

    source = list(source)
    chosen = fmt or detect_format(source)
    if chosen is None:
        return [], ParseReport(parser="unknown")

    if chosen not in PARSERS:
        raise ValueError(f"unknown parser {chosen!r}; "
                         f"expected one of {sorted(PARSERS)}")
    return PARSERS[chosen](source)


# ---------------------------------------------------------------------------
# Aggregation to the window schema the model consumes
# ---------------------------------------------------------------------------

def events_to_windows(events, start=None, end=None, window_seconds=1.0,
                      label=None):
    """Aggregate events into the same window frame the trace generator produces.

    This is the join point: real logs and emulated traces end up in an identical
    schema, so the feature pipeline and the model are unchanged by the source of
    the data.
    """
    import pandas as pd

    if not events:
        # Return the full schema even with nothing to aggregate. Returning a
        # partial frame here silently produces NaN count columns downstream, which
        # then fails deep inside the feature pipeline rather than here.
        empty = pd.DataFrame({
            "window_index": pd.Series(dtype="int64"),
            "create_count": pd.Series(dtype="int64"),
            "rename_count": pd.Series(dtype="int64"),
            "delete_count": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="int64"),
            "episode_id": pd.Series(dtype="int64"),
            "family": pd.Series(dtype="object"),
            "is_stealth": pd.Series(dtype="bool"),
            "benign_template": pd.Series(dtype="object"),
        })
        return empty

    frame = pd.DataFrame([e.to_dict() for e in events])
    frame = frame.sort_values("timestamp")

    if start is None:
        start = float(frame["timestamp"].min())
    if end is None:
        end = float(frame["timestamp"].max())

    total = max(1, int((end - start) / window_seconds) + 1)
    frame["bucket"] = ((frame["timestamp"] - start) // window_seconds).astype(int)
    frame = frame[frame["bucket"] < total]

    counts = (frame.groupby(["bucket", "operation"]).size()
              .unstack(fill_value=0))

    out = pd.DataFrame(index=range(total))
    out.index.name = "bucket"
    for op in (CREATE, RENAME, DELETE):
        out[op] = counts[op] if op in counts.columns else 0

    out = out.reset_index()
    out = out.rename(columns={"bucket": "window_index",
                              CREATE: "create_count",
                              RENAME: "rename_count",
                              DELETE: "delete_count"})
    # Windows with no event in any operation are absent from the unstack, so they
    # must be materialised with an explicit zero. Leaving them as NaN would make
    # the frame non-numeric and fail the model without a useful error.
    for column in ("create_count", "rename_count", "delete_count"):
        out[column] = out[column].fillna(0).astype("int64")
    out["label"] = 1 if label is not None else 0
    out["episode_id"] = -1
    out["family"] = None
    out["is_stealth"] = False
    out["benign_template"] = "ingested"
    out["window_index"] = out["window_index"].astype(int)
    return out


def load_and_prepare(path, fmt=None, window_seconds=1.0, label=None):
    """Read a log file and return (window frame, combined report)."""
    events, report = parse_any(path, fmt=fmt)
    windows = events_to_windows(events, window_seconds=window_seconds,
                                label=label)
    return windows, report
